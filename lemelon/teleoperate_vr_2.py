#!/usr/bin/env python3
"""
How to make LeMelon VR control match Telegrip's accuracy

Key Telegrip Methods to Integrate:
1. Frame-by-frame delta tracking (not origin-based)
2. Separate position and orientation control paths
3. Direct wrist mapping without IK interference
4. Adaptive scaling based on robot state
5. Workspace boundary enforcement before IK
"""

# ============================================
# TELEGRIP METHOD 1: Frame-by-Frame Delta Control
# ============================================
# Instead of: origin → current position → delta
# Telegrip uses: previous frame → current frame → delta

# Replace the position control section in handle_vr_input (lines 730-780) with:

def handle_vr_input_telegrip_style(self, vr_goal):
    """Telegrip-accurate position control"""
    
    if not self.is_initialized or vr_goal is None:
        return
    
    # ... [Keep STEP 1: GRIPPER and STEP 2: WRIST sections unchanged] ...
    
    # ═══════════════════════════════════════════════════════════════
    # TELEGRIP METHOD: Direct Frame-to-Frame Position Delta
    # ═══════════════════════════════════════════════════════════════
    
    # Parse current VR position
    current_vr_pos = parse_vr_position(vr_goal)
    if current_vr_pos is None:
        return
    
    # Check grip state
    current_grip = detect_grip_state(vr_goal)
    
    # TELEGRIP APPROACH: Only need grip to be active, no origin tracking
    if not current_grip:
        # Reset tracking when grip released
        self.prev_vr_pos = None
        return
    
    # First frame with grip active - establish baseline
    if self.prev_vr_pos is None:
        self.prev_vr_pos = current_vr_pos.copy()
        logger.info(f"✅ {self.prefix.upper()}: Telegrip tracking started")
        return
    
    # CRITICAL: Calculate frame-to-frame delta (Telegrip method)
    vr_delta = current_vr_pos - self.prev_vr_pos
    
    # Update tracking for next frame
    self.prev_vr_pos = current_vr_pos.copy()
    
    # Scale factors (Telegrip uses adaptive scaling)
    # These should be tuned based on your robot's workspace
    POSITION_SCALE = 2.0  # Telegrip typically uses 1.5-3.0
    
    # Transform VR delta to robot frame
    robot_delta = vr_to_robot_coordinates(
        {'x': float(vr_delta[0]) * POSITION_SCALE,
         'y': float(vr_delta[1]) * POSITION_SCALE,
         'z': float(vr_delta[2]) * POSITION_SCALE},
        scale=1.0
    )
    
    # Apply per-frame limits (Telegrip uses generous limits)
    MAX_DELTA_PER_FRAME = 0.02  # 20mm per frame at 30Hz = 0.6 m/s max
    robot_delta = np.clip(robot_delta, -MAX_DELTA_PER_FRAME, MAX_DELTA_PER_FRAME)
    
    # Update workspace target (direct addition)
    self.current_x += float(robot_delta[0])
    self.current_y += float(robot_delta[1])
    self.current_z += float(robot_delta[2])
    
    # TELEGRIP METHOD: Enforce workspace bounds BEFORE IK
    # This prevents IK from trying to solve impossible positions
    self.current_x = float(np.clip(self.current_x, 0.08, 0.25))
    self.current_y = float(np.clip(self.current_y, -0.15, 0.15))
    self.current_z = float(np.clip(self.current_z, 0.05, 0.30))
    
    # Continue to IK solving...


# ============================================
# TELEGRIP METHOD 2: Separate Control Paths
# ============================================
# Telegrip keeps position (IK) and orientation (direct) completely separate

def handle_vr_input_separated_paths(self, vr_goal):
    """
    Telegrip separates position IK from wrist control.
    This prevents interference between the two control modes.
    """
    
    # PATH 1: Wrist orientation (DIRECT - no IK)
    # This runs independently and never touches IK solver
    if hasattr(vr_goal, 'wrist_roll_deg') and vr_goal.wrist_roll_deg is not None:
        # Direct assignment (light smoothing only)
        target = clamp_joint_value("wrist_roll", vr_goal.wrist_roll_deg)
        self.target_positions["wrist_roll"] = (
            0.8 * target + 0.2 * self.target_positions["wrist_roll"]
        )
    
    if hasattr(vr_goal, 'wrist_flex_deg') and vr_goal.wrist_flex_deg is not None:
        target = clamp_joint_value("wrist_flex", vr_goal.wrist_flex_deg)
        self.target_positions["wrist_flex"] = (
            0.8 * target + 0.2 * self.target_positions["wrist_flex"]
        )
    
    # PATH 2: Position control (IK - ONLY shoulder/elbow)
    # IK solver NEVER touches wrist joints
    if self.ik_solver and current_vr_pos is not None:
        try:
            target_pos = np.array([self.current_x, self.current_y, self.current_z])
            
            # TELEGRIP KEY: Pass current angles as seed for stability
            current_angles = np.array([
                self.target_positions["shoulder_pan"],
                self.target_positions["shoulder_lift"],
                self.target_positions["elbow_flex"],
                0.0, 0.0, 0.0  # Wrist joints don't matter for position-only IK
            ])
            
            ik_solution = self.ik_solver.solve(
                target_position=target_pos,
                target_orientation_quat=None,  # Position-only
                current_angles_deg=current_angles
            )
            
            # CRITICAL: Only update shoulder/elbow (indices 0-2)
            # NEVER update wrist from IK
            joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
            for i, joint in enumerate(joint_names):
                if i < len(ik_solution):
                    ik_val = ensure_degrees(ik_solution[i])
                    clamped = clamp_joint_value(joint, ik_val)
                    
                    # Light smoothing (Telegrip uses 0.7-0.9 alpha)
                    self.target_positions[joint] = (
                        0.8 * clamped + 0.2 * self.target_positions[joint]
                    )
        
        except Exception as e:
            logger.debug(f"IK failed: {e}")
            # Continue with previous targets


# ============================================
# TELEGRIP METHOD 3: Adaptive Velocity Scaling
# ============================================
# Telegrip scales sensitivity based on distance from target

def calculate_adaptive_scale(self, vr_delta, robot_state):
    """
    Telegrip uses adaptive scaling: 
    - Fast when far from target
    - Slow when close (for precision)
    """
    
    # Calculate delta magnitude
    delta_magnitude = np.linalg.norm(vr_delta)
    
    # Base scale factors
    BASE_SCALE = 2.0
    MIN_SCALE = 0.5
    MAX_SCALE = 5.0
    
    # Adaptive scaling curve
    # Small movements → precise control (low scale)
    # Large movements → fast response (high scale)
    if delta_magnitude < 0.01:  # Less than 10mm
        scale = MIN_SCALE
    elif delta_magnitude > 0.05:  # More than 50mm
        scale = MAX_SCALE
    else:
        # Linear interpolation between min and max
        scale = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * (
            (delta_magnitude - 0.01) / 0.04
        )
    
    return scale


# ============================================
# TELEGRIP METHOD 4: Velocity-Based Control
# ============================================
# Telegrip can use velocity mode for smoother control

def handle_vr_velocity_mode(self, vr_goal):
    """
    Alternative: Velocity-based control (Telegrip optional mode)
    Instead of position targets, send velocity commands
    """
    
    if not hasattr(vr_goal, 'target_velocity'):
        return
    
    # VR provides velocity vector [vx, vy, vz] in m/s
    vr_velocity = np.array(vr_goal.target_velocity)
    
    # Transform to robot frame
    robot_velocity = vr_to_robot_coordinates(
        {'x': vr_velocity[0], 'y': vr_velocity[1], 'z': vr_velocity[2]},
        scale=1.0
    )
    
    # Convert Cartesian velocity to joint velocities
    # This requires Jacobian calculation (advanced)
    jacobian = self._compute_jacobian(self.target_positions)
    joint_velocities = np.linalg.pinv(jacobian) @ robot_velocity
    
    # Apply velocity limits
    MAX_JOINT_VEL = 50.0  # deg/s
    joint_velocities = np.clip(joint_velocities, -MAX_JOINT_VEL, MAX_JOINT_VEL)
    
    # Integrate to get position (dt = 1/30 seconds at 30Hz)
    dt = 1.0 / 30.0
    for i, joint in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex"]):
        self.target_positions[joint] += joint_velocities[i] * dt


# ============================================
# TELEGRIP METHOD 5: Workspace Safety Monitor
# ============================================
# Telegrip validates workspace BEFORE sending to IK

def validate_workspace_position(self, target_pos):
    """
    Telegrip checks if position is reachable before IK.
    This prevents IK from struggling with impossible targets.
    """
    
    # Check if within physical workspace sphere
    distance_from_base = np.linalg.norm(target_pos[:2])  # X-Y plane
    
    MIN_REACH = 0.05  # 50mm minimum
    MAX_REACH = 0.30  # 300mm maximum
    
    if distance_from_base < MIN_REACH:
        logger.warning(f"Target too close to base: {distance_from_base:.3f}m")
        # Push target outward
        direction = target_pos[:2] / (distance_from_base + 1e-6)
        target_pos[:2] = direction * MIN_REACH
    
    if distance_from_base > MAX_REACH:
        logger.warning(f"Target too far from base: {distance_from_base:.3f}m")
        # Pull target inward
        direction = target_pos[:2] / (distance_from_base + 1e-6)
        target_pos[:2] = direction * MAX_REACH
    
    # Check height limits
    target_pos[2] = np.clip(target_pos[2], 0.0, 0.35)
    
    return target_pos


# ============================================
# TELEGRIP METHOD 6: Singularity Avoidance
# ============================================
# Telegrip detects and avoids kinematic singularities

def check_singularity(self, joint_angles):
    """
    Telegrip monitors for singularities and prevents them.
    Singularities occur when arm is fully extended or at special angles.
    """
    
    shoulder_lift = joint_angles[1]  # Joint 2
    elbow_flex = joint_angles[2]     # Joint 3
    
    # Check for fully extended (singularity)
    if abs(shoulder_lift + elbow_flex) < 5.0:
        logger.warning("Near singularity: arm too straight")
        return True
    
    # Check for elbow near zero (another singularity)
    if abs(elbow_flex) < 10.0:
        logger.warning("Near singularity: elbow too straight")
        return True
    
    return False


# ============================================
# COMPLETE TELEGRIP-STYLE handle_vr_input
# ============================================

def handle_vr_input_complete_telegrip(self, vr_goal):
    """
    Complete Telegrip-accurate control method.
    Combines all Telegrip techniques for maximum accuracy.
    """
    
    if not self.is_initialized or vr_goal is None:
        return
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 1: GRIPPER (unchanged)
    # ═══════════════════════════════════════════════════════════════
    if hasattr(vr_goal, 'metadata') and vr_goal.metadata:
        trigger = vr_goal.metadata.get('trigger', 0.0)
        self.target_positions["gripper"] = 45.0 if trigger > 0.5 else 0.0
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: WRIST CONTROL (Direct path - no IK)
    # ═══════════════════════════════════════════════════════════════
    if hasattr(vr_goal, 'wrist_roll_deg') and vr_goal.wrist_roll_deg is not None:
        target = clamp_joint_value("wrist_roll", vr_goal.wrist_roll_deg)
        # High alpha = responsive (Telegrip uses 0.8-0.9)
        self.target_positions["wrist_roll"] = (
            0.85 * target + 0.15 * self.target_positions["wrist_roll"]
        )
    
    if hasattr(vr_goal, 'wrist_flex_deg') and vr_goal.wrist_flex_deg is not None:
        target = clamp_joint_value("wrist_flex", vr_goal.wrist_flex_deg)
        self.target_positions["wrist_flex"] = (
            0.85 * target + 0.15 * self.target_positions["wrist_flex"]
        )
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: POSITION CONTROL (Telegrip frame-to-frame method)
    # ═══════════════════════════════════════════════════════════════
    current_vr_pos = parse_vr_position(vr_goal)
    if current_vr_pos is None:
        return
    
    current_grip = detect_grip_state(vr_goal)
    if not current_grip:
        self.prev_vr_pos = None
        return
    
    if self.prev_vr_pos is None:
        self.prev_vr_pos = current_vr_pos.copy()
        return
    
    # Calculate frame delta
    vr_delta = current_vr_pos - self.prev_vr_pos
    self.prev_vr_pos = current_vr_pos.copy()
    
    # Telegrip adaptive scaling
    delta_mag = np.linalg.norm(vr_delta)
    if delta_mag < 0.001:  # Dead zone
        return
    
    # Scale: 2.0 for normal movements
    scale = 2.0
    if delta_mag < 0.01:
        scale = 1.0  # Precision mode
    elif delta_mag > 0.05:
        scale = 4.0  # Fast mode
    
    # Transform to robot frame
    robot_delta = vr_to_robot_coordinates(
        {'x': vr_delta[0] * scale, 
         'y': vr_delta[1] * scale, 
         'z': vr_delta[2] * scale},
        scale=1.0
    )
    
    # Limit per-frame change
    robot_delta = np.clip(robot_delta, -0.02, 0.02)
    
    # Update workspace
    self.current_x += float(robot_delta[0])
    self.current_y += float(robot_delta[1])
    self.current_z += float(robot_delta[2])
    
    # Validate and clamp workspace
    target_pos = np.array([self.current_x, self.current_y, self.current_z])
    target_pos = validate_workspace_position(self, target_pos)
    self.current_x, self.current_y, self.current_z = target_pos
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 4: IK SOLVING (Telegrip careful method)
    # ═══════════════════════════════════════════════════════════════
    if not self.ik_solver:
        return
    
    try:
        current_angles = np.array([
            self.target_positions["shoulder_pan"],
            self.target_positions["shoulder_lift"],
            self.target_positions["elbow_flex"],
            0.0, 0.0, 0.0
        ])
        
        # Solve IK
        ik_solution = self.ik_solver.solve(
            target_position=target_pos,
            target_orientation_quat=None,
            current_angles_deg=current_angles
        )
        
        ik_solution = ensure_degrees(np.asarray(ik_solution))
        
        # Check singularity
        if check_singularity(self, ik_solution):
            logger.warning("Singularity detected - keeping previous targets")
            return
        
        # Apply to shoulder/elbow only
        for i, joint in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex"]):
            if i < len(ik_solution):
                clamped = clamp_joint_value(joint, ik_solution[i])
                
                # Check for large jumps
                old = self.target_positions[joint]
                if abs(clamped - old) > 30.0:
                    logger.warning(f"{joint}: Large jump detected, rejecting")
                    continue
                
                # Telegrip smoothing: 0.8 alpha (responsive but stable)
                self.target_positions[joint] = (
                    0.8 * clamped + 0.2 * old
                )
    
    except Exception as e:
        logger.debug(f"IK failed: {e}")


# ============================================
# KEY DIFFERENCES: Telegrip vs Current Code
# ============================================
"""
1. DELTA CALCULATION:
   Current: Origin-based (current_pos - origin)
   Telegrip: Frame-based (current_frame - prev_frame)
   → Telegrip is more stable and responsive

2. SCALING:
   Current: Fixed VR_POS_SCALE = 1.0
   Telegrip: Adaptive 1.0-4.0 based on movement speed
   → Telegrip allows both precision and speed

3. WRIST CONTROL:
   Current: Smoothing alpha = 0.7
   Telegrip: Smoothing alpha = 0.85
   → Telegrip is more responsive

4. IK LIMITS:
   Current: Limit after IK solution
   Telegrip: Validate workspace BEFORE IK
   → Telegrip prevents IK struggling

5. SINGULARITY:
   Current: No check
   Telegrip: Active monitoring and avoidance
   → Telegrip prevents arm lockup

6. SEPARATION:
   Current: Wrist and IK can interfere
   Telegrip: Complete separation of paths
   → Telegrip has cleaner control
"""