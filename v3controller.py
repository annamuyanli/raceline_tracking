import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

# ---------------------------------------------------------------------------
# Simple global flag to track whether we've "started" the lap (for finish logic)
# ---------------------------------------------------------------------------
_lap_started = False


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _ensure_avg_ds(racetrack: RaceTrack):
    """
    Precompute average spacing between centerline points.
    We only need this to convert a desired look-ahead distance
    into an index offset along the track.

    Also precompute per-segment length and curvature so we can
    look ahead for sharp turns.  (Curvature is a rough estimate.)
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
    racetrack._avg_ds_computed = True

    # NEW: cache segment lengths and a simple curvature estimate
    racetrack._ds = ds

    # Direction vectors and headings for each segment
    dirs = diffs / ds[:, None]
    headings = np.arctan2(dirs[:, 1], dirs[:, 0])

    # Heading change between consecutive segments
    heading_next = headings[next_idx]
    dpsi = _wrap_angle(heading_next - headings)

    # Approximate unsigned curvature |kappa| ≈ |dpsi| / ds
    curvature = np.abs(dpsi) / ds
    racetrack._curvature = curvature


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Low-level controller: track desired steering angle and speed.

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
    k_delta = 3.0   # steering rate gain
    k_v = 0.8       # acceleration gain

    steer_rate = k_delta * e_delta
    accel = k_v * e_v

    # Actuator limits
    steer_rate = np.clip(steer_rate, parameters[7], parameters[9])
    accel = np.clip(accel, parameters[8], parameters[10])

    return np.array([steer_rate, accel])


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    High-level controller: pure-pursuit steering + steering-based speed limit
    with additional finish-line aiming so that the car passes close to the
    starting point after one lap, and a 'look-ahead' speed planner that
    slows down before sharp turns.

    Returns desired [delta_ref, v_ref].
    """
    global _lap_started

    _ensure_avg_ds(racetrack)

    pos = state[0:2]          # [sx, sy]
    v = float(state[3])
    phi = float(state[4])

    cl = racetrack.centerline
    n = cl.shape[0]

    # -----------------------------------------------------------------------
    # Finish-line logic: track distance to start and mark lap started
    # -----------------------------------------------------------------------
    start_pos = cl[0, 0:2]
    progress = float(np.linalg.norm(pos - start_pos))

    # Treat lap as "started" once we're sufficiently far from the start
    START_LEAVE_DIST = 25.0   # meters
    FINISH_WINDOW = 40.0      # meters radius around the start to aim at it

    if (not _lap_started) and progress > START_LEAVE_DIST:
        _lap_started = True

    # 1) Find closest point on centerline
    diff = cl - pos
    dists_sq = np.einsum("ij,ij->i", diff, diff)
    idx_closest = int(np.argmin(dists_sq))

    # 2) Choose look-ahead point based on speed (geometry unchanged)
    v_abs = abs(v)
    L0 = 20.0   # base look-ahead [m]
    L1 = 0.4    # extra look-ahead per m/s
    lookahead_distance = L0 + L1 * v_abs

    avg_ds = getattr(racetrack, "avg_ds", 5.0)
    index_offset = max(1, int(lookahead_distance / max(avg_ds, 1e-3)))
    idx_target = (idx_closest + index_offset) % n
    target = cl[idx_target]

    # -----------------------------------------------------------------------
    # Finish-line aiming: when we've done a lap and are back near the start,
    # gradually change the pure-pursuit target to the exact start point.
    # -----------------------------------------------------------------------
    if _lap_started and progress < FINISH_WINDOW:
        # Blend between normal target and the start position.
        # w = 0 at progress == FINISH_WINDOW, w -> 1 as progress -> 0.
        w = (FINISH_WINDOW - progress) / FINISH_WINDOW
        w = max(0.0, min(1.0, w))
        target = (1.0 - w) * target + w * start_pos

    # Pure pursuit towards "target"
    vec_to_target = target - pos
    Ld_actual = max(np.linalg.norm(vec_to_target), 1.0)
    angle_to_target = np.arctan2(vec_to_target[1], vec_to_target[0])
    alpha = _wrap_angle(angle_to_target - phi)

    # 3) Pure-pursuit steering law
    wheelbase = float(parameters[0])
    delta_ref = np.arctan2(2.0 * wheelbase * np.sin(alpha), Ld_actual)

    # Steering angle limits
    delta_min = float(parameters[1])
    delta_max = float(parameters[4])
    delta_ref = float(np.clip(delta_ref, delta_min, delta_max))

    # 4) Speed planning from steering (lateral-accel bound)
    v_max_global = float(parameters[5])
    ay_limit = 15.0  # m/s^2, heuristic lateral-accel limit

    abs_tan_delta = max(abs(np.tan(delta_ref)), 1e-3)
    v_curve = np.sqrt(ay_limit * wheelbase / abs_tan_delta)
    v_curve = min(v_curve, v_max_global)

    # Reduce speed further when look-ahead angle is large
    scale_alpha = 1.0 / (1.0 + 1.0 * abs(alpha))
    v_ref = v_curve * scale_alpha

    # Always keep some forward motion, but not exceed v_max
    v_min_des = 8.0
    v_ref = float(np.clip(v_ref, v_min_des, v_max_global))

    # -----------------------------------------------------------------------
    # NEW: anticipatory slowing when a sharp turn is coming up
    # -----------------------------------------------------------------------
    ds_arr = getattr(racetrack, "_ds", None)
    curv_arr = getattr(racetrack, "_curvature", None)

    if ds_arr is not None and curv_arr is not None:
        SHARP_KAPPA = 0.03      # ~ radius < 33 m treated as "sharp"
        MAX_SCAN_DIST = 150.0   # how far ahead we look [m]
        D_SLOW_START = 70.0     # start slowing this far before a sharp turn [m]
        D_SLOW_MIN = 10.0       # by this distance to the turn, be at corner speed [m]

        dist_ahead = 0.0
        idx_scan = idx_closest
        sharp_found = False
        dist_to_sharp = None

        # walk along the track forwards, accumulating distance
        for _ in range(n):  # safety bound: never more than 1 lap
            kappa = float(curv_arr[idx_scan])
            if kappa > SHARP_KAPPA:
                sharp_found = True
                dist_to_sharp = dist_ahead
                break

            dist_ahead += float(ds_arr[idx_scan])
            if dist_ahead >= MAX_SCAN_DIST:
                break

            idx_scan = (idx_scan + 1) % n

        if sharp_found and dist_to_sharp is not None:
            # Desired max speed based on distance to that sharp curve
            V_FAST = v_max_global      # straight / gentle region
            V_TURN = 18.0              # safe speed in sharp curve (tunable)

            if dist_to_sharp >= D_SLOW_START:
                v_ahead_limit = V_FAST
            elif dist_to_sharp <= D_SLOW_MIN:
                v_ahead_limit = V_TURN
            else:
                # linearly interpolate between FAST and TURN speed
                w = (dist_to_sharp - D_SLOW_MIN) / (D_SLOW_START - D_SLOW_MIN)
                w = max(0.0, min(1.0, w))
                v_ahead_limit = V_TURN + w * (V_FAST - V_TURN)

            v_ref = min(v_ref, v_ahead_limit)

    # Extra slowdown near the finish so we don't overshoot the 1 m radius
    if _lap_started and progress < FINISH_WINDOW:
        v_ref = min(v_ref, 20.0)

    return np.array([delta_ref, v_ref])
