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
    Calculates and caches the average distance between centerline points.
    This allows us to convert physical distances (meters) into array indices.
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
    racetrack._avg_ds_computed = True


def _get_local_curvature(cl: np.ndarray, idx: int) -> float:
    """
    Estimates local curvature (k) at a specific index on the centerline.
    
    Curvature k = d(theta) / ds
    High curvature means a sharp turn.
    """
    n = cl.shape[0]
    prev_idx = (idx - 1) % n
    next_idx = (idx + 1) % n
    
    p_prev = cl[prev_idx, 0:2]
    p_cur = cl[idx, 0:2]
    p_next = cl[next_idx, 0:2]

    # Vectors to previous and next points
    v1 = p_cur - p_prev
    v2 = p_next - p_cur

    # Angles of segments
    ang1 = np.arctan2(v1[1], v1[0])
    ang2 = np.arctan2(v2[1], v2[0])
    
    # Change in angle
    dtheta = _wrap_angle(ang2 - ang1)
    
    # Average segment length
    ds_center = 0.5 * (np.linalg.norm(v1) + np.linalg.norm(v2)) + 1e-6
    
    return dtheta / ds_center


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
    k_delta = 4.5   # Gain for steering rate
    k_v = 0.8       # Gain for acceleration

    steer_rate = k_delta * e_delta
    accel = k_v * e_v

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
    4. Adjust speed based on how sharp the turn is (lateral acceleration limits)
       and look ahead for future turns to brake early.
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
    n = cl.shape[0]

    # -----------------------------------------------------------------------
    # 1. Track Position & Lap Logic
    # -----------------------------------------------------------------------
    
    # Find closest point on centerline
    diff = cl - pos
    dists_sq = np.einsum("ij,ij->i", diff, diff)
    idx_closest = int(np.argmin(dists_sq))
    
    # Finish line handling
    start_pos = cl[0, 0:2]
    dist_from_start = float(np.linalg.norm(pos - start_pos))
    
    # Mark lap as "started" once we leave the immediate start area (25m)
    START_LEAVE_DIST = 25.0
    if (not _lap_started) and dist_from_start > START_LEAVE_DIST:
        _lap_started = True

    # -----------------------------------------------------------------------
    # 2. Determine Lookahead Distance
    # -----------------------------------------------------------------------
    
    # Calculate local curvature to adjust lookahead
    kappa = _get_local_curvature(cl, idx_closest)
    
    # Dynamic lookahead: look farther when fast, closer when slow
    v_abs = abs(v)
    L0 = 15.0    # Base lookahead (meters)
    L1 = 0.6     # Increase per m/s of speed
    
    lookahead_distance = L0 + L1 * v_abs

    # Shorten lookahead in sharp turns (hairpins) for tighter cornering
    KAPPA_HAIRPIN = 0.025 
    if abs(kappa) > KAPPA_HAIRPIN:
        lookahead_distance *= 0.6

    # Clamp lookahead to reasonable bounds
    Lmin, Lmax = 10.0, 40.0
    lookahead_distance = max(Lmin, min(lookahead_distance, Lmax))

    # Convert distance to index offset
    avg_ds = getattr(racetrack, "avg_ds", 5.0)
    index_offset = max(1, int(lookahead_distance / max(avg_ds, 1e-3)))
    
    # -----------------------------------------------------------------------
    # 3. Finish Line Aiming
    # -----------------------------------------------------------------------
    # If we are finishing the lap, aim exactly at the start point to ensure
    # we stop within the required radius.
    
    # Estimate remaining track distance
    dist_remaining = (n - idx_closest) * avg_ds
    
    idx_target = (idx_closest + index_offset) % n
    target = cl[idx_target]
    
    # If very close to finish (last 40m), force target to start point
    if _lap_started and (dist_remaining < 40.0 or dist_from_start < 40.0):
        if dist_remaining < 300: # Double check we are actually at end of lap
             target = start_pos

    # -----------------------------------------------------------------------
    # 4. Pure Pursuit Steering
    # -----------------------------------------------------------------------
    
    vec_to_target = target - pos
    
    # Ensure minimum lookahead distance to avoid numerical instability
    min_ld = 1.0
    # Allow very small lookahead when docking at finish
    if _lap_started and (dist_remaining < 40.0):
        min_ld = 0.1
        
    Ld_actual = max(np.linalg.norm(vec_to_target), min_ld)
    
    # Calculate angle to target relative to car's heading
    angle_to_target = np.arctan2(vec_to_target[1], vec_to_target[0])
    alpha = _wrap_angle(angle_to_target - phi)

    # Compute desired steering angle using Pure Pursuit formula
    wheelbase = float(parameters[0])
    delta_ref = np.arctan2(2.0 * wheelbase * np.sin(alpha), Ld_actual)

    # Clamp to car's physical steering limits
    delta_min, delta_max = float(parameters[1]), float(parameters[4])
    delta_ref = float(np.clip(delta_ref, delta_min, delta_max))

    # -----------------------------------------------------------------------
    # 5. Speed Planning
    # -----------------------------------------------------------------------
    
    v_max_global = float(parameters[5])
    ay_limit = 15.0  # Lateral acceleration limit (m/s^2)

    # A. Steering-based limit: Slow down if steering angle is large
    #    v^2 / R = ay_limit  =>  v = sqrt(ay_limit * R)
    #    R approx wheelbase / tan(delta)
    abs_tan_delta = max(abs(np.tan(delta_ref)), 1e-3)
    v_curve = np.sqrt(ay_limit * wheelbase / abs_tan_delta)
    v_ref = min(v_curve, v_max_global)

    # B. Heading error penalty: Slow down if we are not facing the target
    scale_alpha = 1.0 / (1.0 + 2.0 * abs(alpha))
    v_ref *= scale_alpha

    # C. Future Curvature "Horizon": Look ahead for upcoming turns
    #    This allows the car to brake *before* entering a turn.
    kappa_target = _get_local_curvature(cl, idx_target)
    
    horizon_steps = 3    # Check every 3rd point
    horizon_range = 20   # Check ~20 points ahead
    max_weighted_kappa = 0.0
    
    for i in range(1, horizon_range + 1):
        future_idx = (idx_closest + i * horizon_steps) % n
        k_fut = abs(_get_local_curvature(cl, future_idx))
        
        # Weight decreases with distance (brake less for far turns)
        weight = 1.0 / (1.0 + 0.15 * i)
        k_weighted = k_fut * weight
        
        if k_weighted > max_weighted_kappa:
            max_weighted_kappa = k_weighted

    # Combine current, target, and future curvature info
    effective_kappa = max(abs(kappa), abs(kappa_target), max_weighted_kappa)

    # Apply speed limit based on effective curvature
    if effective_kappa > 1e-3:
        v_kappa = np.sqrt(ay_limit / effective_kappa)
        v_ref = min(v_ref, v_kappa)

    # -----------------------------------------------------------------------
    # 6. Final Constraints
    # -----------------------------------------------------------------------
    
    # Ensure minimum speed (unless finishing)
    v_min_des = 8.0
    
    # Finish line deceleration profile
    # Linearly ramp down speed from 50+ m/s to 3 m/s over the last 300m
    FINISH_PREP_DIST = 300.0
    v_finish_limit = float('inf')
    
    if _lap_started and dist_remaining < FINISH_PREP_DIST:
        v_finish_limit = 3.0 + (dist_remaining / FINISH_PREP_DIST) * 50.0
        
        # Respect finish limit
        v_ref = min(v_ref, v_finish_limit)
        
        # Allow slowing down below minimum speed near finish
        if dist_remaining < 40.0:
            v_min_des = 3.0 

    # Final clamp
    v_ref = float(np.clip(v_ref, v_min_des, v_max_global))

    return np.array([delta_ref, v_ref])
