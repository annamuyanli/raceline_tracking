import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

# ---------------------------------------------------------------------------
# Simple global state (per simulation) for lap tracking
# ---------------------------------------------------------------------------
_lap_started = False  # becomes True once the car moves away from the start


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _ensure_track_cache(racetrack: RaceTrack) -> None:
    """
    Precompute and cache:
      - _avg_ds   : average spacing between centerline points
      - _start_pos: starting centerline position (index 0)
    """
    if getattr(racetrack, "_cache_ready", False):
        return

    cl = racetrack.centerline  # shape (N, 2)
    n = cl.shape[0]

    next_idx = (np.arange(n) + 1) % n
    diffs = cl[next_idx] - cl
    ds = np.linalg.norm(diffs, axis=1)
    ds[ds < 1e-6] = 1e-6

    racetrack._avg_ds = float(np.mean(ds))
    racetrack._start_pos = cl[0, 0:2].copy()
    racetrack._cache_ready = True


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

    # P gains: fairly aggressive for responsive control
    k_delta = 5.0   # steering rate gain
    k_v = 1.5       # acceleration gain

    steer_rate = k_delta * e_delta
    accel = k_v * e_v

    # Actuator limits from parameters
    steer_rate = np.clip(steer_rate, parameters[7], parameters[9])
    accel = np.clip(accel, parameters[8], parameters[10])

    return np.array([steer_rate, accel])


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    High-level controller:
      - Pure-pursuit steering along centerline
      - Recovery mode when far off track (steer back to closest point)
      - Finish-line aiming near the start after one lap
      - Simple speed planning based on mode and steering magnitude

    Returns desired [delta_ref, v_ref].
    """
    global _lap_started

    _ensure_track_cache(racetrack)

    # Unpack state
    pos = np.array(state[0:2], dtype=float)  # [sx, sy]
    v = float(state[3])
    phi = float(state[4])

    cl = racetrack.centerline
    n = cl.shape[0]
    avg_ds = float(racetrack._avg_ds)
    start_pos = racetrack._start_pos

    # -----------------------------------------------------------------------
    # Lap progress relative to start (same as simulator's start point)
    # -----------------------------------------------------------------------
    d_start = float(np.linalg.norm(pos - start_pos))
    if (not _lap_started) and d_start > 30.0:
        _lap_started = True

    # -----------------------------------------------------------------------
    # 1) Find closest point on centerline
    # -----------------------------------------------------------------------
    diff = cl - pos  # shape (N, 2)
    dists_sq = np.einsum("ij,ij->i", diff, diff)
    idx_closest = int(np.argmin(dists_sq))
    dist_closest = float(np.sqrt(dists_sq[idx_closest]))

    # Compute a forward lookahead index along the centerline
    v_abs = abs(v)
    lookahead_distance = 15.0 + 0.8 * v_abs  # meters ahead
    index_offset = max(2, int(lookahead_distance / max(avg_ds, 1e-3)))
    idx_target = (idx_closest + index_offset) % n

    # -----------------------------------------------------------------------
    # 2) Choose a geometric target based on mode
    # -----------------------------------------------------------------------
    # Default: look ahead along centerline
    target = cl[idx_target]

    # Recovery mode: far away from track -> aim directly at closest centerline point
    if dist_closest > 30.0:
        target = cl[idx_closest]
    # Finish-line aiming: after we've gone around once, aim at the exact start
    elif _lap_started and d_start < 50.0:
        target = start_pos

    # -----------------------------------------------------------------------
    # 3) Pure-pursuit steering towards the chosen target
    # -----------------------------------------------------------------------
    vec = target - pos
    Ld = float(np.linalg.norm(vec))
    if Ld < 1e-3:
        Ld = 1e-3

    angle_to_target = float(np.arctan2(vec[1], vec[0]))
    alpha = _wrap_angle(angle_to_target - phi)

    wheelbase = float(parameters[0])
    delta_pp = np.arctan2(2.0 * wheelbase * np.sin(alpha), Ld)

    # Steering angle limits
    delta_min = float(parameters[1])
    delta_max = float(parameters[4])
    delta_ref = float(np.clip(delta_pp, delta_min, delta_max))

    # -----------------------------------------------------------------------
    # 4) Speed planning by mode + steering magnitude
    # -----------------------------------------------------------------------
    v_max_global = float(parameters[5])
    v_min_des = 10.0  # always keep some forward motion

    # Base speed by mode
    if dist_closest > 30.0:
        # Recovery: go slowly while we steer back to the track
        v_base = 12.0
    elif _lap_started and d_start < 50.0:
        # Near finish: don't overshoot the 1 m start radius
        v_base = 25.0
    else:
        # Normal running: fairly fast
        v_base = 40.0

    # Slow down in tight turns
    steer_mag = abs(delta_ref)
    curv_factor = 1.0 / (1.0 + 1.8 * steer_mag)
    v_ref = v_base * curv_factor

    # Clip to allowed range
    v_ref = float(np.clip(v_ref, v_min_des, v_max_global))

    return np.array([delta_ref, v_ref])
