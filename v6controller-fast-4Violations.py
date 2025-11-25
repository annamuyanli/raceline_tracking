import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

# ---------------------------------------------------------------------------
# Simple global flag to track whether we have "started" the lap (for finish logic)
# ---------------------------------------------------------------------------
_lap_started = False


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _ensure_avg_ds(racetrack: RaceTrack):
    """
    Precompute average spacing between centerline points and curvature.
    Used to convert a desired lookahead distance into an index offset, and
    to reason about how tight the track is at each point.
    """
    if getattr(racetrack, "_avg_ds_computed", False):
        return

    cl = racetrack.centerline
    n = cl.shape[0]

    # Differences between consecutive points (with wrap-around)
    next_idx = (np.arange(n) + 1) % n
    diffs = cl[next_idx] - cl
    ds = np.linalg.norm(diffs, axis=1)
    ds[ds < 1e-6] = 1e-6

    racetrack.avg_ds = float(np.mean(ds))

    # Approximate curvature at each centerline point using heading change
    kappas = np.zeros(n)
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n

        p_prev = cl[i_prev, 0:2]
        p_cur = cl[i, 0:2]
        p_next = cl[i_next, 0:2]

        v1 = p_cur - p_prev
        v2 = p_next - p_cur

        # Avoid degenerate segments
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            kappas[i] = 0.0
            continue

        ang1 = np.arctan2(v1[1], v1[0])
        ang2 = np.arctan2(v2[1], v2[0])
        dtheta = _wrap_angle(ang2 - ang1)
        ds_center = 0.5 * (np.linalg.norm(v1) + np.linalg.norm(v2)) + 1e-6

        kappas[i] = dtheta / ds_center

    racetrack.curvature = kappas
    racetrack._avg_ds_computed = True


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Low level controller: track desired steering angle and speed.

    state   = [sx, sy, delta, v, phi]
    desired = [delta_ref, v_ref]
    returns = [steering_rate, acceleration]
    """
    assert desired.shape == (2,)

    delta_ref, v_ref = float(desired[0]), float(desired[1])
    delta = float(state[2])
    v = float(state[3])

    # Errors
    e_delta = _wrap_angle(delta_ref - delta)
    e_v = v_ref - v

    # P gains (tuned for responsiveness without oscillation)
    k_delta = 5.0   # steering rate gain
    k_v = 5.0       # acceleration gain

    steer_rate = k_delta * e_delta
    accel = k_v * e_v

    # Traction control: reduce acceleration when steering angle is large
    # This prevents the car from accelerating too fast while turning or
    # immediately after, allowing for a smoother exit.
    if accel > 0:
        delta_max = float(parameters[4])
        steer_ratio = abs(delta) / max(delta_max, 1e-3)
        # Reduce acceleration based on steering (traction circle concept)
        # 1.5 factor means at ~67% steering lock, acceleration is cut to 0.
        throttle_factor = max(0.0, 1.0 - 1.5 * steer_ratio)
        accel *= throttle_factor

    # Actuator limits
    steer_rate = np.clip(steer_rate, parameters[7], parameters[9])
    accel = np.clip(accel, parameters[8], parameters[10])

    return np.array([steer_rate, accel])


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    High level controller: pure pursuit steering with speed planning for
    F1-style tracks.

    - Lookahead distance depends on speed and local curvature.
    - Speed is limited by lateral acceleration from curvature and steering.
    - Extra slowdown in tight corners and near the finish.
    - Finish logic drives the car through the start point and then stops.
    """
    global _lap_started

    _ensure_avg_ds(racetrack)

    pos = state[0:2]          # [sx, sy]
    v = float(state[3])
    phi = float(state[4])

    cl = racetrack.centerline
    kappas = getattr(racetrack, "curvature")
    n = cl.shape[0]

    # -----------------------------------------------------------------------
    # Finish line logic: track distance to start and mark lap started
    # -----------------------------------------------------------------------
    start_pos = cl[0, 0:2]
    progress = float(np.linalg.norm(pos - start_pos))

    START_LEAVE_DIST = 25.0   # meters
    FINISH_WINDOW = 40.0      # meters radius around the start to begin aiming for it

    if (not _lap_started) and progress > START_LEAVE_DIST:
        _lap_started = True

    # -----------------------------------------------------------------------
    # 1) Find closest point on centerline
    # -----------------------------------------------------------------------
    diff = cl[:, 0:2] - pos
    dists_sq = np.einsum("ij,ij->i", diff, diff)
    idx_closest = int(np.argmin(dists_sq))

    kappa_here = float(kappas[idx_closest])

    # -----------------------------------------------------------------------
    # 2) Choose lookahead point based on speed and curvature
    # -----------------------------------------------------------------------
    v_abs = abs(v)

    # Base lookahead (roughly seconds of lookahead along the path)
    L0 = 12.0    # base lookahead [m]
    L1 = 0.7     # extra lookahead per m/s

    # Bounds on lookahead
    Lmin = 8.0
    Lmax = 45.0

    lookahead_distance = L0 + L1 * v_abs

    # Shorten lookahead in tighter corners so we react more aggressively
    KAPPA_MED = 0.02
    KAPPA_TIGHT = 0.04

    if abs(kappa_here) > KAPPA_TIGHT:
        lookahead_distance *= 0.5
    elif abs(kappa_here) > KAPPA_MED:
        lookahead_distance *= 0.7

    # Near the finish we want fine control and a tighter line
    if _lap_started and progress < FINISH_WINDOW:
        lookahead_distance = min(lookahead_distance, 18.0)

    lookahead_distance = max(Lmin, min(lookahead_distance, Lmax))

    avg_ds = getattr(racetrack, "avg_ds", 5.0)
    index_offset = max(1, int(lookahead_distance / max(avg_ds, 1e-3)))
    idx_target = (idx_closest + index_offset) % n
    target = cl[idx_target, 0:2]

    # -----------------------------------------------------------------------
    # 3) Finish line aiming: blend target toward the start position, and
    #    very close to the start just aim directly at it.
    # -----------------------------------------------------------------------
    if _lap_started and progress < FINISH_WINDOW:
        # Blend between normal target and the start position.
        w = (FINISH_WINDOW - progress) / FINISH_WINDOW
        w = max(0.0, min(1.0, w))
        target = (1.0 - w) * target + w * start_pos

        # Within a small radius, just aim directly at the start
        if progress < 5.0:
            target = start_pos.copy()

    # -----------------------------------------------------------------------
    # 4) Pure pursuit steering toward the chosen target
    # -----------------------------------------------------------------------
    vec_to_target = target - pos
    Ld_actual = max(np.linalg.norm(vec_to_target), 1.0)
    angle_to_target = np.arctan2(vec_to_target[1], vec_to_target[0])
    alpha = _wrap_angle(angle_to_target - phi)

    wheelbase = float(parameters[0])

    # Pure pursuit steering law
    delta_ref = np.arctan2(2.0 * wheelbase * np.sin(alpha), Ld_actual)

    # Steering angle limits
    delta_min = float(parameters[1])
    delta_max = float(parameters[4])
    delta_ref = float(np.clip(delta_ref, delta_min, delta_max))

    # -----------------------------------------------------------------------
    # 5) Speed planning
    #
    # Combine:
    # - lateral accel limit from steering angle,
    # - lateral accel limit from track curvature,
    # - extra slowdown when heading error is large,
    # - lower minimum speeds in tight corners.
    # -----------------------------------------------------------------------
    v_max_global = float(parameters[5])
    ay_limit = 11.0  # lateral accel limit [m/s^2], conservative for F1-like turns
    eps = 1e-4

    # Limit from steering angle
    abs_tan_delta = max(abs(np.tan(delta_ref)), eps)
    v_from_delta = np.sqrt(ay_limit * wheelbase / abs_tan_delta)

    # Limit from geometric curvature of the centerline
    if abs(kappa_here) > 1e-6:
        v_from_kappa = np.sqrt(ay_limit / max(abs(kappa_here), eps))
    else:
        v_from_kappa = v_max_global

    v_corner = min(v_from_delta, v_from_kappa, v_max_global)

    # Extra slowdown if we are turning a lot relative to current heading
    alpha_abs = abs(alpha)
    scale_alpha = 1.0 / (1.0 + 2.5 * alpha_abs)

    v_target = v_corner * scale_alpha

    # Minimum desired speed depends on how tight the corner is
    if abs(kappa_here) > KAPPA_TIGHT:
        v_min_des = 3.0
    elif abs(kappa_here) > KAPPA_MED:
        v_min_des = 4.5
    else:
        v_min_des = 6.0

    v_ref = float(np.clip(v_target, v_min_des, v_max_global))

    # -----------------------------------------------------------------------
    # 6) Extra slowdown and stop near finish
    # -----------------------------------------------------------------------
    
    # Calculate distance to finish along the centerline
    # This avoids issues with hairpins near the finish where Euclidean distance is small
    avg_ds = getattr(racetrack, "avg_ds", 5.0)
    dist_along_track = (n - idx_closest) * avg_ds
    
    # Start slowing down well in advance (e.g., 300m) to handle high speeds (like 100m/s)
    SLOWDOWN_DIST = 300.0
    if _lap_started and idx_closest > n / 2 and dist_along_track < SLOWDOWN_DIST:
        v_arrival = 1.5
        a_brake = 6.0  # Decent braking (well within 20m/s^2 limit)
        v_limit = np.sqrt(v_arrival**2 + 2.0 * a_brake * dist_along_track)
        v_ref = min(v_ref, v_limit)

    if _lap_started and progress < FINISH_WINDOW:
        # As we get closer to the start, enforce the low speed cap strictly
        frac = progress / FINISH_WINDOW
        v_cap_finish = 1.5 + 18.5 * frac
        v_ref = min(v_ref, v_cap_finish)

    # When we are really close to the start, command a full stop
    STOP_RADIUS = 1.0  # meters
    if _lap_started and progress < STOP_RADIUS:
        v_ref = 0.0

    return np.array([delta_ref, v_ref])

