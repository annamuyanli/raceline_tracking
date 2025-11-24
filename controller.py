# import numpy as np
# from numpy.typing import ArrayLike

# from simulator import RaceTrack

# def lower_controller(
#     state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
# ) -> ArrayLike:
#     # [steer angle, velocity]
#     assert(desired.shape == (2,))

#     return np.array([0, 100]).T

# def controller(
#     state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
# ) -> ArrayLike:
#     return np.array([0, 100]).T

import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _ensure_avg_ds(racetrack: RaceTrack):
    """
    Precompute average spacing between centerline points.
    We only need this to convert a desired look-ahead distance
    into an index offset along the track.
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
    High-level controller: pure-pursuit steering + steering-based speed limit.

    Returns desired [delta_ref, v_ref].
    """
    _ensure_avg_ds(racetrack)

    pos = state[0:2]          # [sx, sy]
    v = float(state[3])
    phi = float(state[4])

    cl = racetrack.centerline
    n = cl.shape[0]

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

    return np.array([delta_ref, v_ref])
