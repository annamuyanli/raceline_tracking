import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """Wraps an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _get_local_curvature(cl: np.ndarray, idx: int) -> float:
    """
    Estimates local curvature (k) at a specific index on the centerline.
    Curvature k = d(theta) / ds. High curvature means a sharp turn.
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

    ang1 = np.arctan2(v1[1], v1[0])
    ang2 = np.arctan2(v2[1], v2[0])
    
    dtheta = _wrap_angle(ang2 - ang1)
    ds_center = 0.5 * (np.linalg.norm(v1) + np.linalg.norm(v2)) + 1e-6
    
    return dtheta / ds_center


def _ensure_track_state(racetrack: RaceTrack):
    """
    Initializes persistent state on the racetrack object if not present.
    This includes average segment distance and lap start status.
    """
    # Calculate and cache average distance between centerline points
    if not getattr(racetrack, "_avg_ds_computed", False):
        cl = racetrack.centerline
        n = cl.shape[0]
        next_idx = (np.arange(n) + 1) % n
        diffs = cl[next_idx] - cl
        ds = np.linalg.norm(diffs, axis=1)
        ds[ds < 1e-6] = 1e-6
        
        racetrack.avg_ds = float(np.mean(ds))
        racetrack._avg_ds_computed = True

    # Initialize lap started flag for controller logic
    if not hasattr(racetrack, "_controller_lap_started"):
        racetrack._controller_lap_started = False


def _calculate_lookahead(v: float, kappa: float) -> float:
    """Calculates dynamic lookahead distance based on speed and curvature."""
    v_abs = abs(v)
    L0 = 15.0    # Base lookahead (meters)
    L1 = 0.65    # Increase per m/s of speed
    
    lookahead = L0 + L1 * v_abs

    # Shorten lookahead in sharp turns (hairpins)
    KAPPA_HAIRPIN = 0.025 
    if abs(kappa) > KAPPA_HAIRPIN:
        lookahead *= 0.7
    
    # Clamp lookahead to reasonable bounds
    Lmin, Lmax = 10.0, 50.0
    return max(Lmin, min(lookahead, Lmax))


def _apply_corner_cutting(
    target: np.ndarray, 
    cl: np.ndarray, 
    idx_target: int, 
    kappa_here: float
) -> np.ndarray:
    """Adjusts target point to cut corners (move outside on entry, inside on apex)."""
    n = cl.shape[0]
    
    # Calculate approximate normal vector at target
    p_prev = cl[(idx_target - 1) % n, 0:2]
    p_next = cl[(idx_target + 1) % n, 0:2]
    v_t = p_next - p_prev
    
    # Rotate 90 degrees left: (x, y) -> (-y, x)
    normal_t = np.array([-v_t[1], v_t[0]])
    norm_len = np.linalg.norm(normal_t)
    
    if norm_len < 1e-6:
        return target

    normal_t /= norm_len
    kappa_t = _get_local_curvature(cl, idx_target)
    
    # Offset parameters
    CUT_OFFSET_ENTRY = 1.5   # Wide Entry (Outside)
    CUT_OFFSET_APEX = 0.8    # Tight Apex (Inside)
    
    K_STRAIGHT = 0.005
    K_TURN = 0.02
    
    if abs(kappa_here) < K_STRAIGHT and abs(kappa_t) > K_TURN:
        # Approaching turn: Move Outside (Opposite to turn direction)
        target = target - normal_t * np.sign(kappa_t) * CUT_OFFSET_ENTRY
        
    elif abs(kappa_here) > K_TURN:
        # In turn: Move Inside (Towards turn center)
        target = target + normal_t * np.sign(kappa_here) * CUT_OFFSET_APEX
        
    return target


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Low-level controller: Computes actuation commands to track desired state.
    
    Args:
        state: [x, y, steering_angle, velocity, heading]
        desired: [desired_steering_angle, desired_velocity]
        parameters: Car parameters (wheelbase, limits, etc.)
        
    Returns:
        [steering_rate, acceleration]
    """
    delta_ref, v_ref = float(desired[0]), float(desired[1])
    delta = float(state[2])
    v = float(state[3])

    # Proportional gains
    k_delta = 8.0   
    k_v = 2.0       

    steer_rate = k_delta * _wrap_angle(delta_ref - delta)
    accel = k_v * (v_ref - v)

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
    1. Find closest point on track.
    2. Determine lookahead point based on speed and curvature.
    3. Steer towards lookahead point using Pure Pursuit.
    4. Adjust speed based on lateral acceleration limits and future curvature.
    5. Handle finish line logic.
    
    Returns:
        [desired_steering_angle, desired_velocity]
    """
    _ensure_track_state(racetrack)

    # Unpack state
    pos = state[0:2]          # [x, y]
    v = float(state[3])       # velocity
    phi = float(state[4])     # heading

    cl = racetrack.centerline
    n = cl.shape[0]
    avg_ds = racetrack.avg_ds

    # -----------------------------------------------------------------------
    # 1. Track Position & Lap Logic
    # -----------------------------------------------------------------------
    
    # Find closest point on centerline
    diff = cl - pos
    dists_sq = np.einsum("ij,ij->i", diff, diff)
    idx_closest = int(np.argmin(dists_sq))
    
    start_pos = cl[0, 0:2]
    dist_from_start = float(np.linalg.norm(pos - start_pos))
    
    # Mark lap as "started" once we leave the immediate start area
    if (not racetrack._controller_lap_started) and dist_from_start > 25.0:
        racetrack._controller_lap_started = True

    lap_started = racetrack._controller_lap_started
    dist_remaining = (n - idx_closest) * avg_ds

    # -----------------------------------------------------------------------
    # 2. Target Point Selection
    # -----------------------------------------------------------------------
    
    kappa = _get_local_curvature(cl, idx_closest)
    lookahead_dist = _calculate_lookahead(v, kappa)
    
    # Convert distance to index offset
    index_offset = max(1, int(lookahead_dist / max(avg_ds, 1e-3)))
    idx_target = (idx_closest + index_offset) % n
    target = cl[idx_target]

    # Apply corner cutting strategy
    target = _apply_corner_cutting(target, cl, idx_target, kappa)
    
    # Finish line override: if very close to finish, aim at start point
    if lap_started and (dist_remaining < 40.0 or dist_from_start < 40.0):
        if dist_remaining < 300: # Double check we are actually at end of lap
             target = start_pos

    # -----------------------------------------------------------------------
    # 3. Pure Pursuit Steering
    # -----------------------------------------------------------------------
    
    vec_to_target = target - pos
    
    # Ensure minimum lookahead distance to avoid numerical instability
    min_ld = 0.1 if (lap_started and dist_remaining < 40.0) else 1.0
    Ld_actual = max(np.linalg.norm(vec_to_target), min_ld)
    
    # Calculate angle to target relative to car's heading
    angle_to_target = np.arctan2(vec_to_target[1], vec_to_target[0])
    alpha = _wrap_angle(angle_to_target - phi)

    # Compute desired steering angle
    wheelbase = float(parameters[0])
    delta_ref = np.arctan2(2.0 * wheelbase * np.sin(alpha), Ld_actual)

    # Clamp to car's physical steering limits
    delta_min, delta_max = float(parameters[1]), float(parameters[4])
    delta_ref = float(np.clip(delta_ref, delta_min, delta_max))

    # -----------------------------------------------------------------------
    # 4. Speed Planning
    # -----------------------------------------------------------------------
    
    v_max_global = float(parameters[5])
    
    # Check immediate future (next 5 points) to see if curve is ending
    k_future_max = 0.0
    for i in range(1, 6): 
        idx_next = (idx_closest + i) % n
        k_future_max = max(k_future_max, abs(_get_local_curvature(cl, idx_next)))

    # Dynamic lateral acceleration limit
    # Aggressive exit (15.0) vs Conservative entry/mid (12.5)
    ay_limit = 15.0 if abs(kappa) > k_future_max + 0.01 else 12.5

    # A. Steering-based limit: v = sqrt(ay_limit * R)
    abs_tan_delta = max(abs(np.tan(delta_ref)), 1e-3)
    v_curve = np.sqrt(ay_limit * wheelbase / abs_tan_delta)
    v_ref = min(v_curve, v_max_global)

    # B. Heading error penalty
    v_ref *= 1.0 / (1.0 + 2.0 * abs(alpha))

    # C. Future Curvature "Horizon"
    horizon_range = 50
    max_weighted_kappa = 0.0
    for i in range(1, horizon_range + 1):
        future_idx = (idx_closest + i) % n # Using step 1 for smoother check
        k_fut = abs(_get_local_curvature(cl, future_idx))
        weight = 1.0 / (1.0 + 0.5 * i)
        max_weighted_kappa = max(max_weighted_kappa, k_fut * weight)

    effective_kappa = max(abs(kappa), abs(_get_local_curvature(cl, idx_target)), max_weighted_kappa)
    
    if effective_kappa > 1e-3:
        v_ref = min(v_ref, np.sqrt(ay_limit / effective_kappa))

    # -----------------------------------------------------------------------
    # 5. Final Constraints
    # -----------------------------------------------------------------------
    
    v_min_des = 8.0
    
    # Finish line deceleration
    FINISH_PREP_DIST = 50.0
    if lap_started and dist_remaining < FINISH_PREP_DIST:
        v_finish_limit = 10.0 + (dist_remaining / FINISH_PREP_DIST) * 50.0
        v_ref = min(v_ref, v_finish_limit)
        if dist_remaining < 20.0:
            v_min_des = 5.0 

    v_ref = float(np.clip(v_ref, v_min_des, v_max_global))

    return np.array([delta_ref, v_ref])
