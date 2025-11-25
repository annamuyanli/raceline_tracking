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

    Additionally, precompute per-segment length and curvature so
    we can look ahead for sharp curves in the controller.
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

    # NEW: cache segment lengths and simple curvature
    racetrack._ds = ds

    # Direction of each segment
    dirs = diffs / ds[:, None]
    headings = np.arctan2(dirs[:, 1], dirs[:, 0])

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
    with:
      - finish-line aiming so that the car passes close to the starting point
      - anticipatory slowing for upcoming sharp turns based on a long
        look-ahead along the raceline.

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

    # 2) Choose look-ahead point based on speed
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
    # NEW: long look-ahead for sharp turns, and anticipatory slowing
    # -----------------------------------------------------------------------
    ds_arr = getattr(racetrack, "_ds", None)
    curv_arr = getattr(racetrack, "_curvature", None)

    if ds_arr is not None and curv_arr is not None:
        # Treat segments with curvature above this as "sharp"
        SHARP_KAPPA = 0.04        # tunable: smaller → more segments counted as sharp
        MAX_SCAN_DIST = 250.0     # how far ahead we scan [m]
        D_SLOW_MARGIN = 40.0      # start slowing this far before curve start [m]

        # 1) scan forward to find start of next sharp region
        dist = 0.0
        idx_scan = idx_closest
        sharp_start_idx = None
        sharp_start_dist = None

        for _ in range(n):  # safety: never more than a lap
            # advance to next segment along track
            idx_scan = (idx_scan + 1) % n
            dist += float(ds_arr[idx_scan])

            if dist > MAX_SCAN_DIST:
                break

            if float(curv_arr[idx_scan]) > SHARP_KAPPA:
                sharp_start_idx = idx_scan
                sharp_start_dist = dist
                break

        # 2) if there is a sharp region ahead, also find where it ends
        sharp_end_dist = None
        if sharp_start_idx is not None:
            idx = sharp_start_idx
            dist_through = 0.0
            for _ in range(n):
                # stay in region while curvature is sharp
                if float(curv_arr[idx]) <= SHARP_KAPPA:
                    break
                dist_through += float(ds_arr[idx])
                idx = (idx + 1) % n
            sharp_end_dist = sharp_start_dist + dist_through  # distance from current pos

        # 3) decide speed limit based on where we are relative to the sharp region
        if sharp_start_dist is not None and sharp_end_dist is not None:
            # distance from current position to start of region
            d_start = sharp_start_dist
            # distance from current position to end of region
            d_end = sharp_end_dist

            # region spanning [d_start, d_end] ahead along the track
            # we are currently "before" the region (d=0). On later calls,
            # d_start will decrease / go to 0 as we approach / enter it.

            V_FAST = v_max_global   # desired fast speed on straights/gentle turns
            V_TURN = 16.0           # safe speed inside sharp curves (tunable)

            if d_start <= 0.0:
                # We're already inside sharp region (rare with this forward-only scan)
                v_ahead_limit = V_TURN
            elif d_start <= D_SLOW_MARGIN:
                # In the slowdown zone right BEFORE the sharp curve:
                # interpolate from V_TURN (at d_start = 0) to V_FAST (at d_start = D_SLOW_MARGIN)
                w = d_start / D_SLOW_MARGIN
                w = max(0.0, min(1.0, w))
                v_ahead_limit = V_TURN + w * (V_FAST - V_TURN)
            else:
                # We're far from the sharp region: no extra limit yet
                v_ahead_limit = V_FAST

            # If we're effectively at / inside the region, stay at corner speed
            if sharp_start_dist <= 1.0:
                v_ahead_limit = V_TURN

            # Apply anticipatory limit
            v_ref = min(v_ref, v_ahead_limit)
        # else: no sharp curve within MAX_SCAN_DIST -> no change to v_ref

    # Extra slowdown near the finish so we don't overshoot the 1 m radius
    if _lap_started and progress < FINISH_WINDOW:
        v_ref = min(v_ref, 20.0)

    return np.array([delta_ref, v_ref])
