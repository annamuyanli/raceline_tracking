import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
# Tracks whether the car has officially started the lap (left the start zone).
# This is used to enable the finish-line logic (stopping at the end).
_lap_started = False


def _wrap_angle(angle: float) -> float:
    """
    Wraps an angle to the range [-pi, pi].
    Useful for calculating the shortest steering or heading error.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _ensure_avg_ds(racetrack: RaceTrack):
    """
    Calculates and caches the average distance between centerline points
    and precomputes curvature at each point.
    """
    if getattr(racetrack, "_avg_ds_computed", False):
        return

    cl = racetrack.centerline
    n = cl.shape[0]

    # Calculate distance between consecutive points (wrapping around at the end)
    next_idx = (np.arange(n) + 1) % n
    diffs = cl[next_idx] - cl
    ds = np.linalg.norm(diffs, axis=1)
    
    # Avoid division by zero
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
    Low-level controller: Computes actuation commands to track desired state.
    
    Inputs:
        state: [x, y, steering_angle, velocity, heading]
        desired: [desired_steering_angle, desired_velocity]
        parameters: Car parameters (wheelbase, limits, etc.)
        
    Outputs:
        [steering_rate, acceleration]
    """
    delta_ref, v_ref = float(desired[0]), float(desired[1])
    delta = float(state[2])
    v = float(state[3])

    # Calculate errors
    e_delta = _wrap_angle(delta_ref - delta)
    e_v = v_ref - v

    # Proportional gains (P-controller)
    k_delta = 5.0   # Gain for steering rate
    k_v = 5.0       # Gain for acceleration

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

    # Apply physical actuator limits
    # parameters[7/9] are min/max steering rate
    # parameters[8/10] are min/max acceleration
    steer_rate = np.clip(steer_rate, parameters[7], parameters[9])
    accel = np.clip(accel, parameters[8], parameters[10])

    return np.array([steer_rate, accel])


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    High-level controller: Pure Pursuit with Adaptive Speed Control.
    
    Strategy:
    1. Find the closest point on the track.
    2. Determine a "lookahead" point based on current speed and track curvature.
    3. Steer towards that lookahead point (Pure Pursuit).
    4. Adjust speed based on how sharp the turn is (lateral acceleration limits).
    5. Handle finish line logic to stop near the start point after a lap.
    
    Returns:
        [desired_steering_angle, desired_velocity]
    """
    global _lap_started

    _ensure_avg_ds(racetrack)

    # Unpack state
    pos = state[0:2]          # [x, y]
    v = float(state[3])       # velocity
    phi = float(state[4])     # heading

    cl = racetrack.centerline
    kappas = getattr(racetrack, "curvature")
    n = cl.shape[0]

    # -----------------------------------------------------------------------
    # 1. Track Position & Lap Logic
    # -----------------------------------------------------------------------
    
    # Find closest point on centerline
    diff = cl[:, 0:2] - pos
    dists_sq = np.einsum("ij,ij->i", diff, diff)
    idx_closest = int(np.argmin(dists_sq))
    
    kappa_here = float(kappas[idx_closest])

    # Finish line handling
    start_pos = cl[0, 0:2]
    dist_from_start = float(np.linalg.norm(pos - start_pos))
    
    START_LEAVE_DIST = 25.0   # meters
    if (not _lap_started) and dist_from_start > START_LEAVE_DIST:
        _lap_started = True

    # -----------------------------------------------------------------------
    # 2. Determine Lookahead Distance
    # -----------------------------------------------------------------------
    
    v_abs = abs(v)

    # Base lookahead (roughly seconds of lookahead along the path)
    L0 = 12.0    # base lookahead [m]
    L1 = 0.7     # extra lookahead per m/s

    lookahead_distance = L0 + L1 * v_abs

    # Shorten lookahead in tighter corners so we react more aggressively
    KAPPA_MED = 0.02
    KAPPA_TIGHT = 0.04

    if abs(kappa_here) > KAPPA_TIGHT:
        lookahead_distance *= 0.5
    elif abs(kappa_here) > KAPPA_MED:
        lookahead_distance *= 0.7

    # Near the finish we want fine control and a tighter line
    FINISH_WINDOW = 40.0      # meters radius around the start to begin aiming for it
    
    if _lap_started and dist_from_start < FINISH_WINDOW:
        lookahead_distance = min(lookahead_distance, 18.0)

    # Bounds on lookahead
    Lmin = 8.0
    Lmax = 45.0
    lookahead_distance = max(Lmin, min(lookahead_distance, Lmax))

    # Convert distance to index offset
    avg_ds = getattr(racetrack, "avg_ds", 5.0)
    index_offset = max(1, int(lookahead_distance / max(avg_ds, 1e-3)))
    
    # -----------------------------------------------------------------------
    # 3. Finish Line Aiming
    # -----------------------------------------------------------------------
    
    idx_target = (idx_closest + index_offset) % n
    target = cl[idx_target, 0:2]

    if _lap_started and dist_from_start < FINISH_WINDOW:
        # Blend between normal target and the start position.
        w = (FINISH_WINDOW - dist_from_start) / FINISH_WINDOW
        w = max(0.0, min(1.0, w))
        target = (1.0 - w) * target + w * start_pos

        # Within a small radius, just aim directly at the start
        if dist_from_start < 5.0:
            target = start_pos.copy()

    # -----------------------------------------------------------------------
    # 4. Pure Pursuit Steering
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
    # 5. Speed Planning
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
    # 6. Final Constraints (Finish Logic)
    # -----------------------------------------------------------------------
    
    # Calculate distance to finish along the centerline
    dist_along_track = (n - idx_closest) * avg_ds
    
    # Start slowing down well in advance (e.g., 300m) to handle high speeds
    SLOWDOWN_DIST = 300.0
    if _lap_started and idx_closest > n / 2 and dist_along_track < SLOWDOWN_DIST:
        v_arrival = 1.5
        a_brake = 6.0  # Decent braking (well within 20m/s^2 limit)
        v_limit = np.sqrt(v_arrival**2 + 2.0 * a_brake * dist_along_track)
        v_ref = min(v_ref, v_limit)

    if _lap_started and dist_from_start < FINISH_WINDOW:
        # As we get closer to the start, enforce the low speed cap strictly
        frac = dist_from_start / FINISH_WINDOW
        v_cap_finish = 1.5 + 18.5 * frac
        v_ref = min(v_ref, v_cap_finish)

    # When we are really close to the start, command a full stop
    STOP_RADIUS = 1.0  # meters
    if _lap_started and dist_from_start < STOP_RADIUS:
        v_ref = 0.0

    return np.array([delta_ref, v_ref])

