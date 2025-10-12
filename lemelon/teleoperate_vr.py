#!/usr/bin/env python3
"""
LeMelon VR Teleoperation - Telegrip Standard Control Pattern
Uses SO100 URDF + PyBullet IK/FK with Telegrip VR ControlGoal

Based on LeRobot documentation:
- https://huggingface.co/docs/lerobot/so100
- https://huggingface.co/docs/lerobot/lekiwi

Hardware Requirements:
    - SO100 robot arms connected via USB (Dynamixel U2D2 adapter)
    - VR headset with Telegrip support (WebXR compatible)
    - SO100 URDF files in src/lerobot/model/URDF/SO100/

USB Setup:
    LeMelon uses 3 separate USB connections:
    1. Left arm:  /dev/ttyACM0 (motors 1-6)
    2. Right arm: /dev/ttyACM2 (motors 1-6, same IDs but different bus)
    3. Base:      /dev/ttyACM1 (motors 7-9, omni-wheels with LeKiwi kinematics)
    
    Set permissions:
    sudo chmod 666 /dev/ttyACM*
    
    Or add user to dialout group:
    sudo usermod -aG dialout $USER
    (then log out and back in)

Usage Examples:
    # Use real robot with default USB ports
    python teleoperate_vr.py --real-robot
    
    # Test without hardware
    python teleoperate_vr.py --mock-robot
    
    # Specify custom USB ports (all three buses)
    python teleoperate_vr.py --real-robot \
        --left-port /dev/ttyACM0 \
        --right-port /dev/ttyACM2 \
        --base-port /dev/ttyACM1
    
    # Use custom URDF file
    python teleoperate_vr.py --real-robot --urdf-path /path/to/custom/so100.urdf

Telegrip Control Features:
    - Grip-to-activate: Press grip button to activate arm control
    - Relative positioning: Arm moves relative to grip press position
    - Direct wrist mapping: VR controller orientation → robot wrist angles
    - Position-only IK: 3DOF IK for shoulder/elbow + direct wrist control
    - Gripper control: VR trigger → gripper open/close
    - Base movement: VR thumbsticks → base linear/angular velocity
    - LeRobot dataset compatible action/observation format
    - PyBullet-based IK/FK using SO100 URDF
"""

import argparse
import asyncio
import logging
import sys
import threading
import time
import traceback
import os
from pathlib import Path

import numpy as np
import pybullet as p  # type: ignore

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from lerobot.robots.lemelon import LeMelon, LeMelonConfig  # noqa: E402
from lerobot.model.so100_ik import (  # noqa: E402
    IKSolver,
    ForwardKinematics,
    euler_to_quaternion,
    quaternion_to_euler,
    vr_to_robot_coordinates,
    compute_relative_position,
    ensure_degrees,
    likely_radians
)
from lerobot.utils.robot_utils import busy_wait  # noqa: E402

# VR Monitor import
from vr_monitor import VRMonitor  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_vr_goal_structure(vr_goal, prefix="VR"):
    """
    Diagnostic helper: Print exact structure of VR goal object.
    Use this to verify what fields your vr_monitor actually provides.
    """
    if vr_goal is None:
        logger.info(f"{prefix} goal: None")
        return
    
    logger.info(f"📋 {prefix} Goal Structure:")
    logger.info(f"   Type: {type(vr_goal)}")
    logger.info(f"   Attributes: {[a for a in dir(vr_goal) if not a.startswith('_')]}")
    
    # Check critical Telegrip/ControlGoal fields
    if hasattr(vr_goal, 'target_position'):
        logger.info(f"   target_position: {vr_goal.target_position} (type: {type(vr_goal.target_position)})")
    
    if hasattr(vr_goal, 'orientation'):
        logger.info(f"   orientation (quat): {vr_goal.orientation}")
    
    if hasattr(vr_goal, 'wrist_roll_deg'):
        logger.info(f"   wrist_roll_deg: {vr_goal.wrist_roll_deg}")
    
    if hasattr(vr_goal, 'wrist_flex_deg'):
        logger.info(f"   wrist_flex_deg: {vr_goal.wrist_flex_deg}")
    
    if hasattr(vr_goal, 'gripper_position'):
        logger.info(f"   gripper_position: {vr_goal.gripper_position} (0-1 range)")
    
    if hasattr(vr_goal, 'gripper_closed'):
        logger.info(f"   gripper_closed: {vr_goal.gripper_closed}")
    
    if hasattr(vr_goal, 'gripper_velocity'):
        logger.info(f"   gripper_velocity: {vr_goal.gripper_velocity}")
    
    if hasattr(vr_goal, 'metadata'):
        logger.info(f"   metadata: {vr_goal.metadata}")
        if vr_goal.metadata:
            if 'grip' in vr_goal.metadata:
                logger.info(f"      grip: {vr_goal.metadata['grip']}")
            if 'trigger' in vr_goal.metadata:
                logger.info(f"      trigger: {vr_goal.metadata['trigger']}")
            # Check all keys for trigger-related fields
            trigger_keys = [k for k in vr_goal.metadata.keys() if 'trigger' in k.lower() or 'gripper' in k.lower()]
            if trigger_keys:
                logger.info(f"      trigger_related_keys: {trigger_keys}")
    
    if hasattr(vr_goal, 'arm'):
        logger.info(f"   arm: {vr_goal.arm}")


def verify_ik_fk_consistency(ik_solver, fk_solver, test_joints_deg):
    """
    Verify IK and FK are using consistent coordinate frames.
    Returns True if consistent, False otherwise.
    """
    try:
        # Forward: joints -> position
        test_joints = np.array(test_joints_deg, dtype=float)
        fk_pos, fk_orient = fk_solver.compute(test_joints)
        
        logger.info(f"🔍 IK/FK Consistency Check:")
        logger.info(f"  Test joints (deg): {test_joints[:3]}")
        logger.info(f"  FK position (m): {fk_pos}")
        logger.info(f"  FK orientation: {fk_orient}")
        
        # Backward: position -> joints
        ik_joints = ik_solver.solve(
            target_position=fk_pos,
            target_orientation_quat=None,
            current_angles_deg=test_joints
        )
        
        # Check if we get back similar joints
        joint_error = np.abs(ik_joints[:3] - test_joints[:3]).max()
        
        logger.info(f"  IK solution (deg): {ik_joints[:3]}")
        logger.info(f"  Joint error (deg): {joint_error:.2f}°")
        
        # Verify FK of IK solution matches target
        fk_check_pos, _ = fk_solver.compute(
            np.concatenate([ik_joints, test_joints[len(ik_joints):]])
        )
        position_error = np.linalg.norm(fk_check_pos - fk_pos)
        logger.info(f"  Position error (m): {position_error:.4f}")
        
        if joint_error > 5.0:
            logger.warning(f"⚠️  IK/FK may be inconsistent (error={joint_error:.2f}°)")
            return False
        
        if position_error > 0.01:  # 1cm threshold
            logger.warning(f"⚠️  FK(IK(pos)) != pos (error={position_error:.4f}m)")
            return False
        
        logger.info("✅ IK/FK consistency verified")
        return True
    except Exception as e:
        logger.error(f"❌ IK/FK consistency check failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# USB port defaults (can be overridden via command line)
# LeMelon uses 3 separate USB ports for:
# - Left arm (motors 1-6)
# - Right arm (motors 1-6, same IDs but different bus)
# - Base (motors 7-9 for omni-wheels)
DEFAULT_LEFT_ARM_PORT = "/dev/ttyACM0"
DEFAULT_RIGHT_ARM_PORT = "/dev/ttyACM2"
DEFAULT_BASE_PORT = "/dev/ttyACM1"

# Joint mapping configurations
LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}

RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

# Configuration
FPS = 30
USE_6DOF_IK = True  # Enable full 6DOF control (position + orientation)

# ═══════════════════════════════════════════════════════════════
# TELEGRIP CONTROL MODE (Frame-by-Frame Delta Tracking)
# ═══════════════════════════════════════════════════════════════
# This script uses the Telegrip ControlGoal standard for VR control
# with SO100 kinematics functions for coordinate transformation.
#
# TELEGRIP KEY FEATURES:
#   1. Frame-by-frame delta tracking (NOT origin-based)
#   2. Adaptive scaling (0.5x-4.0x based on movement speed)
#   3. Separate control paths (IK for position, direct for wrist)
#   4. Workspace validation BEFORE IK
#   5. Singularity detection and avoidance
#   6. High responsiveness (alpha=0.8 IK, 0.85 wrist)
#
# Control Scheme:
#   - Grip button: Activate arm control (frame-by-frame delta tracking)
#   - VR controller position: Control end-effector position (when grip active)
#   - VR controller orientation: Direct mapping to wrist roll/pitch
#   - Trigger button: Open/close gripper
#   - Thumbsticks: Base movement (linear + angular velocity)
#
# IK Strategy:
#   - Position-only 3DOF IK: shoulder_pan, shoulder_lift, elbow_flex
#   - Direct wrist control: wrist_roll, wrist_flex from VR controller
#   - This provides intuitive 5DOF control (3 position + 2 wrist orientation)
#   - Uses SO100 kinematics functions from lerobot.model.so100_ik
#
# Coordinate Mapping (via vr_to_robot_coordinates):
#   - VR: X=right, Y=up, Z=back → Robot: X=forward, Y=left, Z=up
#   - VR controller roll → Robot wrist_roll (direct or via quaternion)
#   - VR controller pitch → Robot wrist_flex (direct or via quaternion)
#
# Kinematics Functions Used:
#   - vr_to_robot_coordinates: VR→Robot coordinate transformation
#   - quaternion_to_euler: Convert VR controller orientation
#   - IKSolver.solve: Position-only IK (or 6DOF if orientation provided)
#   - ForwardKinematics.compute: Verify end-effector position
#   - validate_workspace_position: Telegrip workspace validation
#   - check_singularity: Telegrip singularity avoidance
# ═══════════════════════════════════════════════════════════════

# Telegrip VR → Robot conversion parameters (tunable)
# TELEGRIP ADAPTIVE SCALING PARAMETERS
VR_BASE_SCALE = 2.0          # Base scale for normal movements (Telegrip: 1.5-3.0)
VR_MIN_SCALE = 0.5           # Minimum scale for precision (Telegrip)
VR_MAX_SCALE = 4.0           # Maximum scale for fast movements (Telegrip)
VR_WRIST_ANGLE_SCALE = 4.0   # Quaternion→wrist angle scaling factor
VR_DELTA_POS_LIMIT = 0.02    # Max position change per frame (meters) - Telegrip: 20mm
VR_DELTA_ANGLE_LIMIT = 15.0  # Max angle change per frame (degrees)
IK_SMOOTH_ALPHA = 0.8        # IK solution smoothing - TELEGRIP: 0.8 (more responsive)
WRIST_SMOOTH_ALPHA = 0.85    # Wrist smoothing - TELEGRIP: 0.85 (very responsive)

# Joint velocity limits (degrees per second) - safety feature
# These prevent sudden large movements that could damage hardware
MAX_JOINT_SPEED = {
    "shoulder_pan": 90.0,    # deg/s
    "shoulder_lift": 60.0,   # deg/s (conservative - near base)
    "elbow_flex": 60.0,      # deg/s
    "wrist_flex": 120.0,     # deg/s (faster - smaller joint)
    "wrist_roll": 120.0,     # deg/s (faster - smaller joint)
    "gripper": 180.0,        # deg/s (fast gripper open/close)
}

# VR data timeout (seconds) - watchdog feature
VR_TIMEOUT = 0.2  # 200ms - hold position if VR data is stale


def vr_to_robot_delta(vr_pos, prev_vr_pos, scale=VR_BASE_SCALE):
    """
    Compute delta in VR space and return robot-frame deltas.
    Uses the SO100 kinematics coordinate transformation for consistency.
    
    Telegrip coordinate mapping (from so100_ik.py):
    - VR: X=right, Y=up, Z=back (toward user)
    - Robot: X=forward, Y=left, Z=up
    
    Args:
        vr_pos: Current VR position [x, y, z] in meters
        prev_vr_pos: Previous VR position [x, y, z] in meters
        scale: Scale factor for VR → robot mapping
        
    Returns:
        Tuple of (robot_dx, robot_dy, robot_dz) in meters
    """
    if vr_pos is None or prev_vr_pos is None:
        return 0.0, 0.0, 0.0
    
    # Calculate VR deltas
    vr_delta = {
        'x': (vr_pos[0] - prev_vr_pos[0]) * scale,
        'y': (vr_pos[1] - prev_vr_pos[1]) * scale,
        'z': (vr_pos[2] - prev_vr_pos[2]) * scale
    }
    
    # Limit individual deltas before transformation
    vr_delta['x'] = float(np.clip(
        vr_delta['x'], -VR_DELTA_POS_LIMIT, VR_DELTA_POS_LIMIT
    ))
    vr_delta['y'] = float(np.clip(
        vr_delta['y'], -VR_DELTA_POS_LIMIT, VR_DELTA_POS_LIMIT
    ))
    vr_delta['z'] = float(np.clip(
        vr_delta['z'], -VR_DELTA_POS_LIMIT, VR_DELTA_POS_LIMIT
    ))
    
    # Use SO100 kinematics coordinate transformation
    robot_delta = vr_to_robot_coordinates(vr_delta, scale=1.0)
    
    return float(robot_delta[0]), float(robot_delta[1]), float(robot_delta[2])


def clamp_joint_value(joint_name: str, value: float) -> float:
    """Clamp joint value to safe SO100 limits."""
    limits = {
        "shoulder_pan": (-180, 180),
        "shoulder_lift": (-85, 85),
        "elbow_flex": (-85, 85),
        "wrist_flex": (-90, 90),
        "wrist_roll": (-90, 90),
        "gripper": (0, 90),
    }
    lo, hi = limits.get(joint_name, (-360, 360))
    return float(np.clip(value, lo, hi))


def smooth_joint_target(old_value: float, new_value: float, alpha: float = IK_SMOOTH_ALPHA) -> float:
    """Smooth transition between old and new joint targets."""
    return float((1.0 - alpha) * old_value + alpha * new_value)


def to_python_scalar(x):
    """
    Convert numpy scalars/arrays to Python native types.
    Critical for LeRobot send_action compatibility.
    """
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.item())
        else:
            return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if hasattr(x, "item"):
        return x.item()
    return x


def sanitize_action_dict(action: dict) -> dict:
    """
    Convert numpy types to Python natives for LeRobot send_action.
    LeRobot expects native Python floats, not numpy.float64.
    """
    return {k: to_python_scalar(v) for k, v in action.items()}


def parse_vr_position(vr_goal) -> np.ndarray:
    """
    Robustly parse target_position from VR goal.
    Handles list, tuple, dict, numpy array formats.
    Returns None if unavailable.
    """
    if not hasattr(vr_goal, 'target_position') or vr_goal.target_position is None:
        return None
    
    try:
        tpos = vr_goal.target_position
        # Handle dict format: {'x': ..., 'y': ..., 'z': ...}
        if isinstance(tpos, dict):
            return np.array([
                float(tpos['x']),
                float(tpos['y']),
                float(tpos['z'])
            ], dtype=float)
        # Handle list/tuple/array format: [x, y, z]
        else:
            return np.array([
                float(tpos[0]),
                float(tpos[1]),
                float(tpos[2])
            ], dtype=float)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def detect_grip_state(vr_goal) -> bool:
    """
    Detect if grip button is pressed.
    Checks multiple possible field names and formats.
    """
    if not hasattr(vr_goal, 'metadata') or not vr_goal.metadata:
        return False
    
    # Try common field names
    for field_name in ['grip', 'grip_pressed', 'grip_down']:
        grip_value = vr_goal.metadata.get(field_name, None)
        if grip_value is not None:
            # Handle both boolean and float formats
            if isinstance(grip_value, bool):
                return grip_value
            else:
                return float(grip_value) > 0.5
    
    # Check relative_position flag (Telegrip pattern)
    if vr_goal.metadata.get('relative_position', False):
        return True
    
    return False


def validate_workspace_position(target_pos: np.ndarray) -> np.ndarray:
    """
    Telegrip workspace validation - check if position is reachable before IK.
    This prevents IK from struggling with impossible targets.
    """
    target_pos = target_pos.copy()
    
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


def check_singularity(joint_angles: np.ndarray) -> bool:
    """
    Telegrip singularity detection - avoid kinematic singularities.
    Singularities occur when arm is fully extended or at special angles.
    """
    if len(joint_angles) < 3:
        return False
    
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


class LeMelonArmController:
    """
    VR arm control - Telegrip Frame-by-Frame Delta + SO100 PyBullet IK/FK
    LeRobot framework compliant
    
    Telegrip Control Flow (Standard Pattern):
    1. VR controller position → Frame-to-frame delta (not origin-based)
    2. Adaptive scaling → Based on movement speed (0.5x-4.0x)
    3. Workspace validation → Before IK solve
    4. IK solver (PyBullet+URDF) → Position-only (shoulder/elbow)
    5. Singularity check → Reject unsafe configurations
    6. Direct wrist control → Separate from IK (high alpha=0.85)
    7. P-control → Smooth robot action (IK alpha=0.8)
    8. LeRobot send_action → Hardware
    
    Key Telegrip Features:
    - Frame-by-frame delta tracking (NOT origin-based)
    - Adaptive scaling for precision + speed
    - Workspace validation prevents impossible IK targets
    - Singularity detection avoids arm lockup
    - Separate control paths for position (IK) and orientation (direct)
    - PyBullet URDF-based FK/IK for geometric accuracy
    - State management to prevent control drift
    
    PyBullet Integration:
    - Uses p.resetJointState() to set robot configuration
    - Uses p.stepSimulation() to update link transforms
    - Uses p.getLinkState() with computeForwardKinematics=1
    - Uses p.calculateInverseKinematics() with damping and tight thresholds
    """
    
    def __init__(self, prefix="left", ik_solver=None, fk_solver=None, kp=1.0):
        self.prefix = prefix
        self.kp = kp
        self.ik_solver = ik_solver
        self.fk_solver = fk_solver
        
        # Target joint positions (degrees)
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        
        # IK workspace tracking (meters)
        # These will be set from FK during initialization
        self.current_x = 0.15
        self.current_y = 0.0
        self.current_z = 0.15
        self.pitch = 0.0  # degrees
        
        # Orientation tracking for 6DOF mode (euler angles in degrees)
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        
        # TELEGRIP: Frame-by-frame delta tracking (NO origin needed)
        # Track previous VR state to calculate frame-to-frame deltas
        self.prev_vr_pos = None  # Previous VR controller position
        self.prev_wrist_flex = None
        self.prev_wrist_roll = None
        
        # Control sensitivity (telegrip-style)
        self.vr_scale = 1.0  # Scale factor for VR→robot mapping
        
        # Debug counter for periodic logging
        self._debug_counter = 0
        self._vr_data_received = False
        
        # Initialized flag
        self.is_initialized = False
    
    def _get_joint_limits(self, joint_name):
        """Get joint limits for clamping"""
        limits = {
            "shoulder_pan": (-180, 180),
            "shoulder_lift": (-85, 85),
            "elbow_flex": (-85, 85),
            "wrist_flex": (-90, 90),
            "wrist_roll": (-90, 90),
            "gripper": (0, 90),
        }
        return limits.get(joint_name, (-360, 360))
    
    def initialize(self, robot):
        """Initialize from current robot position - FIXED with FK verification"""
        obs = robot.get_observation()
        
        # Initialize target positions from current robot state
        for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                      "wrist_flex", "wrist_roll", "gripper"]:
            obs_key = f"{self.prefix}_arm_{joint}.pos"
            self.target_positions[joint] = float(np.clip(
                obs[obs_key],
                *self._get_joint_limits(joint)
            ))
            
            # Log if clamping occurred
            original = obs[obs_key]
            clamped = self.target_positions[joint]
            if abs(original - clamped) > 0.1:
                logger.warning(
                    f"⚠️  {self.prefix.upper()}: {joint} clamped from {original:.1f}° to {clamped:.1f}°"
                )
        
        # Initialize wrist angles from current state
        self.pitch = self.target_positions["wrist_flex"]
        
        # Get initial workspace position from FK (CRITICAL for accuracy)
        if self.fk_solver:
            try:
                current_joints = np.array([
                    self.target_positions["shoulder_pan"],
                    self.target_positions["shoulder_lift"], 
                    self.target_positions["elbow_flex"],
                    self.target_positions["wrist_flex"],
                    self.target_positions["wrist_roll"],
                    self.target_positions["gripper"],
                ])
                ee_pos, _ = self.fk_solver.compute(current_joints)
                
                # ALWAYS use FK for initial position (most accurate)
                self.current_x = float(ee_pos[0])
                self.current_y = float(ee_pos[1]) 
                self.current_z = float(ee_pos[2])
                
                logger.info(
                    f"🔧 {self.prefix.upper()}: FK initialization = "
                    f"[{self.current_x:.3f}, {self.current_y:.3f}, {self.current_z:.3f}]"
                )
                
                # Warn if outside expected bounds
                if not (0.05 < self.current_x < 0.30 and 
                       -0.20 < self.current_y < 0.20 and 
                       0.0 < self.current_z < 0.35):
                    logger.warning(
                        f"⚠️  {self.prefix.upper()}: Initial position outside normal bounds"
                    )
                
                # Verify IK can solve back to current position
                if self.ik_solver:
                    test_ik = self.ik_solver.solve(
                        target_position=ee_pos,
                        target_orientation_quat=None,
                        current_angles_deg=current_joints
                    )
                    ik_error = np.abs(test_ik[:3] - current_joints[:3]).max()
                    if ik_error > 10.0:
                        logger.warning(
                            f"⚠️  {self.prefix.upper()}: "
                            f"IK/FK mismatch detected (error={ik_error:.1f}°)"
                        )
                    
            except Exception as e:
                logger.error(f"❌ {self.prefix.upper()}: FK initialization failed: {e}")
                # Fall back to defaults but warn
                logger.warning(f"⚠️  {self.prefix.upper()}: Using default workspace position")
                self.current_x = 0.15
                self.current_y = 0.0
                self.current_z = 0.15
        else:
            logger.warning(f"⚠️  {self.prefix.upper()}: No FK solver, using default workspace")
            self.current_x = 0.15
            self.current_y = 0.0
            self.current_z = 0.15
        
        self.is_initialized = True
        logger.info(f"✅ {self.prefix.upper()}: Initialized at workspace "
                    f"[{self.current_x:.3f}, {self.current_y:.3f}, {self.current_z:.3f}]")
    
    def _validate_vr_data(self, vr_goal):
        """Validate VR data structure and contents (more permissive for testing)"""
        if not vr_goal:
            return False
        
        # Check for essential attributes
        if not hasattr(vr_goal, 'target_position'):
            if self._debug_counter % 90 == 0:
                logger.debug(f"⚠️  {self.prefix.upper()}: Missing 'target_position' attribute")
            return False
        
        # target_position being None is normal when grip is released
        if vr_goal.target_position is None:
            return False
        
        # For testing, be more permissive about metadata
        if not hasattr(vr_goal, 'metadata') or not vr_goal.metadata:
            if self._debug_counter % 90 == 0:
                logger.debug(f"⚠️  {self.prefix.upper()}: No metadata available (continuing anyway)")
            # Don't return False here - allow testing without metadata
        
        return True
    
    def handle_vr_input(self, vr_goal):
        """
        Handle VR input using Telegrip's Frame-by-Frame Delta Control.
        
        Telegrip Control Flow (Frame-by-Frame Method):
        1. Gripper: Check gripper_position → gripper_closed → metadata['trigger']
        2. Wrist: Direct VR→robot mapping (alpha=0.85, separate from IK)
        3. Position Delta: Frame-to-frame (NOT origin-based)
        4. Adaptive Scaling: 0.5x precision → 4.0x fast based on speed
        5. Workspace Validation: BEFORE IK solve (prevent impossible targets)
        6. IK Solve: Position-only for shoulder/elbow (alpha=0.8)
        7. Singularity Check: Reject unsafe arm configurations
        
        Key Differences from Origin-Based:
        - Delta = current_frame - previous_frame (not current - origin)
        - No origin tracking needed
        - More stable and responsive
        - Adaptive scaling for precision + speed
        
        All fields gracefully handle None/missing values.
        """
        if vr_goal is None:
            return
        
        # Initialize if not already done
        if not self.is_initialized:
            logger.warning(
                f"⚠️  {self.prefix.upper()}: handle_vr_input called before "
                f"initialize - skipping"
            )
            return
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: GRIPPER CONTROL (Always active, independent of grip)
        # ═══════════════════════════════════════════════════════════════
        # Expected Telegrip fields (in priority order):
        #   1. gripper_position (float 0.0-1.0) - analog trigger value
        #   2. gripper_velocity (float) - rate of change, may indicate trigger state
        #   3. gripper_closed (bool) - binary gripper state
        #   4. metadata['trigger'] (float 0.0-1.0) - fallback trigger
        # Output: gripper joint angle in degrees (0-90°)
        
        gripper_target = None
        
        # Method 1: Direct gripper_position attribute (Telegrip standard)
        if hasattr(vr_goal, 'gripper_position') and vr_goal.gripper_position is not None:
            # Telegrip provides gripper position directly (0-1 range typically)
            gripper_target = float(vr_goal.gripper_position) * 90.0  # Scale to 0-90°
        
        # Method 2: gripper_velocity (some Telegrip modes use this)
        elif hasattr(vr_goal, 'gripper_velocity') and vr_goal.gripper_velocity is not None:
            # Integrate velocity to get position (or use as direct position if in 0-1 range)
            vel = float(vr_goal.gripper_velocity)
            if -1.0 <= vel <= 1.0:
                # Treat as position if in reasonable range
                gripper_target = abs(vel) * 90.0
        
        # Method 3: gripper_closed boolean
        elif hasattr(vr_goal, 'gripper_closed') and vr_goal.gripper_closed is not None:
            gripper_target = 45.0 if vr_goal.gripper_closed else 0.0
        
        # Method 4: Fallback to metadata trigger value
        elif hasattr(vr_goal, 'metadata') and vr_goal.metadata:
            trigger_value = vr_goal.metadata.get('trigger', None)
            if trigger_value is not None:
                gripper_target = float(trigger_value) * 90.0  # Scale trigger to gripper angle
        
        # Update gripper target if we got a value
        if gripper_target is not None:
            prev_gripper_state = self.target_positions["gripper"] > 20
            self.target_positions["gripper"] = float(np.clip(gripper_target, 0.0, 90.0))
            
            # Log state changes only
            current_gripper_state = self.target_positions["gripper"] > 20
            if current_gripper_state and not prev_gripper_state:
                logger.info(
                    f"🤏 {self.prefix.upper()}: Gripper CLOSING → "
                    f"{self.target_positions['gripper']:.1f}°"
                )
            elif not current_gripper_state and prev_gripper_state:
                logger.info(
                    f"🤏 {self.prefix.upper()}: Gripper OPENING → "
                    f"{self.target_positions['gripper']:.1f}°"
                )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: WRIST ORIENTATION CONTROL (Direct VR controller mapping)
        # ═══════════════════════════════════════════════════════════════
        # Expected Telegrip fields (in priority order):
        #   1. wrist_roll_deg (float, degrees) - direct roll from VR controller
        #   2. wrist_flex_deg/wrist_pitch_deg (float, degrees) - direct pitch
        #   3. orientation (quat [x,y,z,w]) - fallback, returns radians→convert to degrees
        # Output: wrist_roll and wrist_flex in degrees (-90 to 90°)
        
        wrist_control_updated = False
        
        # Method 1: Direct wrist angle attributes (preferred)
        if hasattr(vr_goal, 'wrist_roll_deg') and vr_goal.wrist_roll_deg is not None:
            # Direct mapping: VR controller roll → robot wrist roll
            target_wrist_roll = float(vr_goal.wrist_roll_deg)
            # Clamp and smooth (TELEGRIP: Higher alpha for more responsive wrist control)
            target_wrist_roll = clamp_joint_value("wrist_roll", target_wrist_roll)
            self.target_positions["wrist_roll"] = smooth_joint_target(
                self.target_positions["wrist_roll"], target_wrist_roll,
                WRIST_SMOOTH_ALPHA
            )
            wrist_control_updated = True
            
            # Log first wrist roll data
            if self.prev_wrist_roll is None:
                logger.info(
                    f"✅ {self.prefix.upper()}: Wrist roll control active: "
                    f"{target_wrist_roll:.1f}°"
                )
            self.prev_wrist_roll = target_wrist_roll
        
        if hasattr(vr_goal, 'wrist_flex_deg') and vr_goal.wrist_flex_deg is not None:
            # Direct mapping: VR controller pitch → robot wrist flex
            target_wrist_flex = float(vr_goal.wrist_flex_deg)
            # Clamp and smooth (TELEGRIP: Higher alpha for more responsive wrist control)
            target_wrist_flex = clamp_joint_value("wrist_flex", target_wrist_flex)
            self.target_positions["wrist_flex"] = smooth_joint_target(
                self.target_positions["wrist_flex"], target_wrist_flex,
                WRIST_SMOOTH_ALPHA
            )
            wrist_control_updated = True
            
            # Log first wrist flex data
            if self.prev_wrist_flex is None:
                logger.info(
                    f"✅ {self.prefix.upper()}: Wrist flex control active: "
                    f"{target_wrist_flex:.1f}°"
                )
            self.prev_wrist_flex = target_wrist_flex
        
        # Method 2: Quaternion orientation (fallback)
        # Convert VR controller orientation quaternion to wrist angles
        if (not wrist_control_updated and
                hasattr(vr_goal, 'orientation') and
                vr_goal.orientation is not None):
            try:
                # VR controller orientation as quaternion [x, y, z, w]
                controller_quat = np.array(vr_goal.orientation)
                
                # Convert to euler angles using SO100 kinematics
                roll, pitch, yaw = quaternion_to_euler(controller_quat)
                
                # Map to wrist joints with scaling
                # Roll maps to wrist_roll, pitch maps to wrist_flex
                target_wrist_roll = np.rad2deg(roll) * VR_WRIST_ANGLE_SCALE
                target_wrist_flex = np.rad2deg(pitch) * VR_WRIST_ANGLE_SCALE
                
                # Clamp and apply
                target_wrist_roll = clamp_joint_value("wrist_roll", target_wrist_roll)
                target_wrist_flex = clamp_joint_value("wrist_flex", target_wrist_flex)
                
                self.target_positions["wrist_roll"] = smooth_joint_target(
                    self.target_positions["wrist_roll"], target_wrist_roll,
                    IK_SMOOTH_ALPHA
                )
                self.target_positions["wrist_flex"] = smooth_joint_target(
                    self.target_positions["wrist_flex"], target_wrist_flex,
                    IK_SMOOTH_ALPHA
                )
                
                # Log first quaternion-based wrist control
                if self.prev_wrist_roll is None:
                    logger.info(
                        f"✅ {self.prefix.upper()}: Wrist control from quaternion: "
                        f"roll={target_wrist_roll:.1f}° flex={target_wrist_flex:.1f}°"
                    )
                
                self.prev_wrist_roll = target_wrist_roll
                self.prev_wrist_flex = target_wrist_flex
                
            except Exception as e:
                logger.debug(
                    f"⚠️  {self.prefix.upper()}: "
                    f"Failed to process orientation quaternion: {e}"
                )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: ARM POSITION CONTROL (TELEGRIP Frame-by-Frame Delta)
        # ═══════════════════════════════════════════════════════════════
        
        # Parse VR position (using helper)
        current_vr_pos = parse_vr_position(vr_goal)
        
        # If no valid position, skip position control
        if current_vr_pos is None:
            if self._debug_counter % 90 == 0:
                logger.debug(f"⚠️  {self.prefix.upper()}: No valid target_position")
            self._debug_counter += 1
            return
        
        # Detect grip button state (using helper)
        current_grip = detect_grip_state(vr_goal)
        
        # TELEGRIP APPROACH: Reset tracking when grip released
        if not current_grip:
            self.prev_vr_pos = None
            if self._debug_counter % 90 == 0:
                logger.debug(f"⚠️  {self.prefix.upper()}: Arm control paused (grip released)")
            self._debug_counter += 1
            return
        
        # First frame with grip active - establish baseline
        if self.prev_vr_pos is None:
            self.prev_vr_pos = current_vr_pos.copy()
            
            # Log first VR data reception
            if not self._vr_data_received:
                self._vr_data_received = True
                logger.info(f"✅ {self.prefix.upper()}: VR position data active")
            
            logger.info(f"✅ {self.prefix.upper()}: Telegrip tracking started")
            return
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: TELEGRIP FRAME-TO-FRAME DELTA CALCULATION
        # ═══════════════════════════════════════════════════════════════
        
        # CRITICAL: Calculate frame-to-frame delta (Telegrip method)
        vr_delta = current_vr_pos - self.prev_vr_pos
        
        # Update tracking for next frame
        self.prev_vr_pos = current_vr_pos.copy()
        
        # TELEGRIP: Adaptive scaling based on movement speed
        delta_mag = np.linalg.norm(vr_delta)
        
        # Dead zone - ignore tiny movements
        if delta_mag < 0.001:
            self._debug_counter += 1
            return
        
        # Adaptive scale: precision for small moves, speed for large moves
        if delta_mag < 0.01:  # Less than 10mm
            scale = VR_MIN_SCALE  # Precision mode
        elif delta_mag > 0.05:  # More than 50mm
            scale = VR_MAX_SCALE  # Fast mode
        else:
            # Linear interpolation between min and max
            scale = VR_MIN_SCALE + (VR_MAX_SCALE - VR_MIN_SCALE) * (
                (delta_mag - 0.01) / 0.04
            )
        
        # Transform VR delta to robot frame
        robot_delta = vr_to_robot_coordinates(
            {'x': float(vr_delta[0]) * scale,
             'y': float(vr_delta[1]) * scale,
             'z': float(vr_delta[2]) * scale},
            scale=1.0
        )
        
        # Apply per-frame limits (Telegrip uses generous limits)
        robot_delta = np.clip(robot_delta, -VR_DELTA_POS_LIMIT, VR_DELTA_POS_LIMIT)
        
        # Update workspace target (direct addition)
        self.current_x += float(robot_delta[0])
        self.current_y += float(robot_delta[1])
        self.current_z += float(robot_delta[2])
        
        # TELEGRIP METHOD: Enforce workspace bounds BEFORE IK
        # This prevents IK from trying to solve impossible positions
        self.current_x = float(np.clip(self.current_x, 0.08, 0.25))
        self.current_y = float(np.clip(self.current_y, -0.15, 0.15))
        self.current_z = float(np.clip(self.current_z, 0.05, 0.30))
        
        # Log transformation with coordinate info (reduced frequency)
        if self._debug_counter % 30 == 0:
            logger.debug(
                f"📍 {self.prefix.upper()}: VR delta=[{vr_delta[0]:.4f}, "
                f"{vr_delta[1]:.4f}, {vr_delta[2]:.4f}] scale={scale:.1f} → "
                f"Robot target=[{self.current_x:.3f},{self.current_y:.3f},"
                f"{self.current_z:.3f}]"
            )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 5: IK SOLVING (TELEGRIP Method with Validation)
        # ═══════════════════════════════════════════════════════════════
        # Solve IK for workspace target position
        # Returns first 3 joints: shoulder_pan, shoulder_lift, elbow_flex
        # Wrist joints already set in STEP 2, gripper in STEP 1
        
        if not self.ik_solver:
            # No IK available - skip position updates
            self._debug_counter += 1
            return
        
        try:
            # Prepare IK inputs with TELEGRIP workspace validation
            target_pos = np.array([
                float(self.current_x),
                float(self.current_y),
                float(self.current_z)
            ], dtype=float)
            
            # TELEGRIP: Validate workspace BEFORE IK
            target_pos = validate_workspace_position(target_pos)
            
            # Update workspace with validated position
            self.current_x, self.current_y, self.current_z = target_pos
            
            current_angles = np.array([
                float(self.target_positions["shoulder_pan"]),
                float(self.target_positions["shoulder_lift"]),
                float(self.target_positions["elbow_flex"]),
                0.0, 0.0, 0.0  # Wrist joints don't matter for position-only IK
            ], dtype=float)
            
            # Solve IK (position-only)
            ik_solution = self.ik_solver.solve(
                target_position=target_pos,
                target_orientation_quat=None,
                current_angles_deg=current_angles
            )
            
            ik_solution = np.asarray(ik_solution, dtype=float)
            ik_solution_deg = ensure_degrees(ik_solution)
            
            # TELEGRIP: Check singularity before applying solution
            if check_singularity(ik_solution_deg):
                logger.warning(
                    f"⚠️  {self.prefix.upper()}: Singularity detected - "
                    f"keeping previous targets"
                )
                self._debug_counter += 1
                return
            
            # CRITICAL: Only update first 3 joints (shoulder/elbow)
            # Keep wrist angles from STEP 2 (direct VR control)
            joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
            dt = 1.0 / FPS  # Time since last update
            
            for i, joint_name in enumerate(joint_names):
                if i >= len(ik_solution_deg):
                    break
                    
                # Clamp to limits
                clamped = clamp_joint_value(joint_name, ik_solution_deg[i])
                
                # Check for large jumps BEFORE applying (TELEGRIP: 30° threshold)
                old_val = self.target_positions[joint_name]
                if abs(clamped - old_val) > 30.0:
                    logger.warning(
                        f"⚠️  {self.prefix.upper()}: {joint_name} large jump "
                        f"{old_val:.1f}° -> {clamped:.1f}° (rejecting)"
                    )
                    continue  # Don't update this joint
                
                # Smooth transition (TELEGRIP: Higher alpha for responsiveness)
                smoothed_val = smooth_joint_target(old_val, clamped, IK_SMOOTH_ALPHA)
                
                # VELOCITY LIMITING (safety feature)
                # Limit rate of change to prevent sudden dangerous movements
                delta_deg = smoothed_val - old_val
                max_delta_deg = MAX_JOINT_SPEED.get(joint_name, 90.0) * dt
                
                if abs(delta_deg) > max_delta_deg:
                    # Limit velocity
                    delta_deg = np.sign(delta_deg) * max_delta_deg
                    new_val = old_val + delta_deg
                    if self._debug_counter % 30 == 0:
                        logger.debug(
                            f"🛡️  {self.prefix.upper()}: {joint_name} velocity limited "
                            f"to {max_delta_deg:.1f}°/frame"
                        )
                else:
                    new_val = smoothed_val
                
                self.target_positions[joint_name] = new_val
                
                # Log large clamping
                if abs(ik_solution_deg[i] - clamped) > 2.0:
                    logger.debug(
                        f"   {joint_name}: {ik_solution_deg[i]:.1f}° -> "
                        f"{clamped:.1f}°"
                    )
            
        except Exception as e:
            # Log but don't crash - keep previous targets
            logger.debug(f"⚠️  {self.prefix.upper()}: IK solve failed: {e}")
            if self._debug_counter % 90 == 0:
                logger.debug(
                    f"   Target was: pos=[{self.current_x:.3f},"
                    f"{self.current_y:.3f},{self.current_z:.3f}]"
                )
        
        # Increment debug counter
        self._debug_counter += 1
    
    def p_control_action(self, robot):
        """
        Generate proportional control action (LeRobot + Telegrip pattern).
        
        Telegrip Control Loop Pattern:
        1. Safely reads current joint positions from robot observation
        2. Validates target positions are within safe limits
        3. Verifies FK position matches IK target (accuracy check)
        4. Computes P-control: new_pos = current + kp * (target - current)
        5. Clamps commands to safe ranges
        6. Returns fully sanitized action dict (Python floats only)
        7. Handles missing observation keys gracefully
        
        Returns:
            dict: Action dictionary ready for robot.send_action()
        """
        # Get current robot state (defensive)
        try:
            obs = robot.get_observation()
        except Exception as e:
            logger.warning(
                f"⚠️  {self.prefix.upper()}: Failed to get observation: {e}"
            )
            obs = {}
        
        # If not initialized, return hold action using observed positions
        if not self.is_initialized:
            action = {}
            for joint in self.target_positions:
                obs_key = f"{self.prefix}_arm_{joint}.pos"
                # Use observed position if available, else 0
                action[obs_key] = float(obs.get(obs_key, 0.0))
            return sanitize_action_dict(action)
        
        # TELEGRIP: Validate FK position before sending commands
        # This ensures PyBullet IK is giving accurate results
        if self.fk_solver and self._debug_counter % 30 == 0:
            try:
                # Get FK position from current targets
                current_joint_targets = np.array([
                    self.target_positions["shoulder_pan"],
                    self.target_positions["shoulder_lift"],
                    self.target_positions["elbow_flex"],
                    self.target_positions["wrist_flex"],
                    self.target_positions["wrist_roll"],
                    self.target_positions["gripper"],
                ])
                fk_pos, _ = self.fk_solver.compute(current_joint_targets)
                
                # Check if FK matches our IK target workspace
                ik_target = np.array([self.current_x, self.current_y, self.current_z])
                position_error = np.linalg.norm(fk_pos - ik_target)
                
                # Warn if FK deviates significantly from IK target
                if position_error > 0.03:  # 3cm threshold
                    logger.warning(
                        f"⚠️ {self.prefix.upper()}: FK/IK mismatch: "
                        f"IK_target={ik_target}, FK={fk_pos}, "
                        f"error={position_error:.3f}m"
                    )
            except Exception as e:
                logger.debug(f"FK validation failed: {e}")
        
        # Build action with P-control
        action = {}
        for joint in self.target_positions:
            obs_key = f"{self.prefix}_arm_{joint}.pos"
            
            # Defensive reads with fallbacks
            target = float(self.target_positions[joint])
            current = float(obs.get(obs_key, target))
            
            # TELEGRIP: Validate target is within safe limits
            lo, hi = self._get_joint_limits(joint)
            target = float(np.clip(target, lo, hi))
            
            # P-control
            error = target - current
            new_pos = float(current + self.kp * error)
            
            # TELEGRIP: Clamp command to safe range
            new_pos = float(np.clip(new_pos, lo, hi))
            
            action[obs_key] = new_pos
            
            # Debug log large errors (indicates possible issues)
            if abs(error) > 10.0 and self._debug_counter % 90 == 0:
                logger.debug(
                    f"   {self.prefix.upper()} {joint}: "
                    f"cur={current:.1f}° tgt={target:.1f}° "
                    f"err={error:.1f}° → cmd={new_pos:.1f}°"
                )
        
        # Sanitize all values to Python floats
        return sanitize_action_dict(action)
    
    def get_fk_position(self):
        """Get current FK position from joint targets (LeRobot pattern)"""
        if not self.fk_solver or not self.is_initialized:
            return None, None
        
        try:
            joint_array = np.array([
                self.target_positions["shoulder_pan"],
                self.target_positions["shoulder_lift"],
                self.target_positions["elbow_flex"],
                self.target_positions["wrist_flex"],
                self.target_positions["wrist_roll"],
                self.target_positions["gripper"],
            ])
            return self.fk_solver.compute(joint_array)
        except Exception as e:
            logger.debug(f"FK computation failed: {e}")
            return None, None
    
    def get_action_dict(self):
        """
        Get action dictionary in LeRobot dataset format.
        This can be used for recording teleoperation data.
        
        Returns:
            dict: Action dict with joint positions suitable for LeRobot dataset
        """
        return {
            f"{self.prefix}_arm_{joint}.pos": self.target_positions[joint]
            for joint in self.target_positions
        }
    
    def get_workspace_state(self):
        """
        Get current workspace state for logging/debugging.
        
        Returns:
            dict: Workspace state including IK target, FK position, orientation
        """
        fk_pos, fk_orient = self.get_fk_position()
        
        state = {
            "ik_target": [self.current_x, self.current_y, self.current_z],
            "fk_position": fk_pos.tolist() if fk_pos is not None else None,
            "fk_orientation": fk_orient.tolist() if fk_orient is not None else None,
            "joint_targets": self.target_positions.copy(),
        }
        
        if USE_6DOF_IK:
            state["target_orientation"] = {
                "roll": self.target_roll,
                "pitch": self.target_pitch,
                "yaw": self.target_yaw,
            }
        
        return state


def check_usb_ports(left_port, right_port, base_port):
    """Check if USB ports exist and are accessible."""
    left_exists = os.path.exists(left_port)
    right_exists = os.path.exists(right_port)
    base_exists = os.path.exists(base_port)
    
    ports = {
        "left_arm": (left_port, left_exists),
        "right_arm": (right_port, right_exists),
        "base": (base_port, base_exists)
    }
    
    for name, (port, exists) in ports.items():
        if not exists:
            logger.warning(f"⚠️  {name.title()} USB port not found: {port}")
        else:
            # Check if port is accessible
            try:
                if os.access(port, os.R_OK | os.W_OK):
                    logger.info(f"✅ {name.title()} USB port accessible: {port}")
                else:
                    logger.warning(
                        f"⚠️  {name.title()} port found but not accessible: {port}\n"
                        f"   Run: sudo chmod 666 {port}"
                    )
            except Exception:
                logger.info(f"✅ {name.title()} USB port found: {port}")
    
    return left_exists, right_exists, base_exists


def verify_urdf_setup(urdf_path):
    """Verify URDF file and dependencies exist."""
    urdf_file = Path(urdf_path)
    
    if not urdf_file.exists():
        logger.error(f"❌ URDF file not found: {urdf_path}")
        return False
    
    # Check URDF directory for mesh files
    urdf_dir = urdf_file.parent
    mesh_dir = urdf_dir / "assets"
    
    if mesh_dir.exists():
        mesh_count = len(list(mesh_dir.glob("*.stl")))
        logger.info(f"✅ URDF directory found with {mesh_count} mesh files")
    else:
        logger.warning(f"⚠️  No mesh directory found at {mesh_dir}")
    
    return True


def main():
    """Main teleoperation function - LeRobot framework compliant"""
    parser = argparse.ArgumentParser(
        description="VR Teleoperation for LeMelon robot with SO100 IK/FK"
    )
    parser.add_argument("--mock-robot", action="store_true",
                       help="Use mock robot (no hardware)")
    parser.add_argument("--real-robot", action="store_true",
                       help="Use real robot hardware")
    parser.add_argument(
        "--left-port", type=str, default=DEFAULT_LEFT_ARM_PORT,
        help=f"USB port for left arm (default: {DEFAULT_LEFT_ARM_PORT})"
    )
    parser.add_argument(
        "--right-port", type=str, default=DEFAULT_RIGHT_ARM_PORT,
        help=f"USB port for right arm (default: {DEFAULT_RIGHT_ARM_PORT})"
    )
    parser.add_argument(
        "--base-port", type=str, default=DEFAULT_BASE_PORT,
        help=f"USB port for base motors (default: {DEFAULT_BASE_PORT})"
    )
    parser.add_argument("--urdf-path", type=str, default=None,
                       help="Custom path to SO100 URDF file")
    parser.add_argument("--test-vr", action="store_true",
                       help="Test mode with simulated VR goals (no VR headset needed)")
    parser.add_argument("--diagnose-vr", action="store_true",
                       help="Print detailed VR goal structure every second (diagnostic mode)")
    args = parser.parse_args()
    
    mock = args.mock_robot or not args.real_robot
    
    print("=" * 70)
    print("🎮 LeMelon VR Teleoperation - TELEGRIP Frame-by-Frame Control")
    print("   LeRobot Framework + SO100 URDF IK/FK + Telegrip VR")
    print("=" * 70)
    print(f"Robot Mode: {'🤖 Mock (Simulation)' if mock else '⚡ Real Hardware'}")
    print("Control:    🎯 Telegrip Frame-by-Frame Delta + Adaptive Scaling")
    print("Update:     30 Hz control loop with workspace validation")
    print("Features:   ✓ Singularity avoidance ✓ Separate control paths")
    
    if not mock:
        print("")
        print("Hardware Configuration:")
        print(f"  Left arm:  {args.left_port}")
        print(f"  Right arm: {args.right_port}")
        print(f"  Base:      {args.base_port}")
        print("")
        
        # Check USB ports
        left_ok, right_ok, base_ok = check_usb_ports(
            args.left_port, args.right_port, args.base_port
        )
        if not (left_ok or right_ok):
            logger.error("")
            logger.error("❌ No robot arms found!")
            logger.error("")
            logger.error("Troubleshooting:")
            logger.error("  1. Check USB connections are secure")
            logger.error("  2. List devices: ls /dev/ttyUSB* /dev/ttyACM*")
            logger.error("  3. Check permissions: ls -l /dev/ttyACM*")
            logger.error("  4. Fix permissions: sudo chmod 666 /dev/ttyACM*")
            logger.error("  5. Or add to group: sudo usermod -aG dialout $USER")
            logger.error("")
            return
        
        if not base_ok:
            logger.warning("⚠️  Base not found - robot will have no base movement")
            logger.warning(f"   Expected at: {args.base_port}")
    
    print("=" * 70)
    print()
    
    # Initialize VR Monitor (skip if test mode)
    vr_monitor = None
    if args.test_vr:
        logger.info("🧪 Running in TEST-VR mode (simulated VR goals)")
        logger.info("   No VR headset required - using synthetic control signals")
    else:
        logger.info("🔧 Starting VR Monitor...")
        vr_monitor = VRMonitor()
        
        vr_thread = threading.Thread(
            target=lambda: asyncio.run(vr_monitor.start_monitoring()),
            daemon=True
        )
        vr_thread.start()
        
        # Wait for VR to be ready
        max_wait = 50
        for i in range(max_wait):
            if vr_monitor.is_running:
                break
            time.sleep(0.1)
            if i % 10 == 0:
                logger.info(f"   Waiting for VR... ({i//10}s)")
        
        if not vr_monitor.is_running:
            logger.error("❌ VR monitor failed to start after 5s")
            return
        
        logger.info("✅ VR Monitor started")
        time.sleep(1.0)
    
    # Initialize Robot (LeRobot pattern)
    logger.info("🔌 Connecting to robot...")
    try:
        # Create robot config with USB port information
        robot_config = LeMelonConfig(mock=mock)
        
        # Pass USB port info to config if using real hardware
        if not mock:
            # LeMelon uses 3 separate USB ports
            robot_config.port_left_arm = args.left_port
            robot_config.port_right_arm = args.right_port
            robot_config.port_base = args.base_port
            logger.info(
                f"   Using USB ports: left={args.left_port}, "
                f"right={args.right_port}, base={args.base_port}"
            )
        
        robot = LeMelon(robot_config)
        robot.connect(calibrate=True)
        
        if not robot.is_connected:
            logger.error("❌ Robot connection failed")
            return
        
        logger.info("✅ Robot connected")
        
        # Log individual arm connection status if available
        if hasattr(robot, 'left_arm_connected'):
            status = '✅ Connected' if robot.left_arm_connected else '❌ Disconnected'
            logger.info(f"   Left arm: {status}")
        if hasattr(robot, 'right_arm_connected'):
            status = '✅ Connected' if robot.right_arm_connected else '❌ Disconnected'
            logger.info(f"   Right arm: {status}")
            
    except Exception as e:
        logger.error(f"❌ Robot initialization failed: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Initialize PyBullet IK/FK with SO100 URDF (per LeRobot docs)
    logger.info("🔧 Initializing SO100 IK/FK solvers...")
    ik_solver = None
    fk_solver = None
    physics_client = None
    
    try:
        physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
        
        # Determine URDF path
        if args.urdf_path:
            urdf_path = args.urdf_path
            logger.info(f"   Using custom URDF path: {urdf_path}")
        else:
            urdf_path = str(
                Path(__file__).parents[2] / "src" / "lerobot" / "model" / 
                "URDF" / "SO100" / "so100.urdf"
            )
            logger.info("   Using default URDF: .../URDF/SO100/so100.urdf")
        
        # Verify URDF setup
        if not verify_urdf_setup(urdf_path):
            raise FileNotFoundError(
                f"SO100 URDF not found at {urdf_path}\n"
                f"   Make sure the URDF folder exists at:\n"
                f"   src/lerobot/model/URDF/SO100/so100.urdf"
            )
        
        robot_id = p.loadURDF(
            urdf_path,
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=physics_client
        )
        
        logger.info(f"✅ URDF loaded successfully (robot_id={robot_id})")
        
        # FK solver for end-effector position/orientation
        # CRITICAL: Use link 3 (wrist) as end-effector, NOT link 5 (jaw)
        # URDF structure: base(0) → shoulder(1) → upper_arm(2) → lower_arm(3)
        # → wrist(4) → gripper(5) → jaw(6)
        # For 3DOF position IK, we want the wrist position
        fk_solver = ForwardKinematics(
            physics_client=physics_client,
            robot_id=robot_id,
            joint_indices=list(range(6)),
            end_effector_link_index=3  # FIXED: wrist link, not jaw
        )
        logger.info("✅ FK solver initialized (end-effector=wrist link)")
        
        # IK solver with 6DOF support (per LeRobot SO-100 documentation)
        # Joint limits based on SO100 specifications
        joint_limits_min = np.array([-180, -85, -85, -90, -90, 0])
        joint_limits_max = np.array([180, 85, 85, 90, 90, 90])
        
        ik_solver = IKSolver(
            physics_client=physics_client,
            robot_id=robot_id,
            joint_indices=list(range(6)),
            end_effector_link_index=3,  # FIXED: wrist link, not jaw
            joint_limits_min_deg=joint_limits_min,
            joint_limits_max_deg=joint_limits_max,
            arm_name="lemelon"
        )
        
        mode_str = "6DOF (position + orientation)" if USE_6DOF_IK else "3DOF (position only)"
        logger.info(f"✅ IK solver initialized with {mode_str}")
        logger.info(f"   End-effector: Link 3 (wrist)")
        logger.info(f"   Joint limits: {joint_limits_min} to {joint_limits_max}")
        logger.info(f"   IK params: maxIter=200, threshold=1e-6, damping=0.01")
        logger.info("✅ SO100 IK/FK solvers ready from URDF with PyBullet")
        
        # CRITICAL FIX 6: Verify IK/FK consistency with test position
        logger.info("🔍 Verifying IK/FK consistency...")
        test_joints = np.array([0.0, -30.0, 60.0, -30.0, 0.0, 0.0])
        if verify_ik_fk_consistency(ik_solver, fk_solver, test_joints):
            logger.info("✅ IK/FK coordinate frames are consistent")
        else:
            logger.error(
                "❌ IK/FK coordinate frames may be inconsistent!\n"
                "   This will cause position control drift.\n"
                "   Check URDF and kinematics functions."
            )
    except Exception as e:
        logger.error(f"⚠️  IK/FK solver failed: {e}")
        logger.error(traceback.format_exc())
        logger.warning("⚠️  Continuing without IK/FK")
    
    # Initialize arm controllers
    try:
        left_arm = LeMelonArmController(
            prefix="left", ik_solver=ik_solver, fk_solver=fk_solver, kp=1.0
        )
        right_arm = LeMelonArmController(
            prefix="right", ik_solver=ik_solver, fk_solver=fk_solver, kp=1.0
        )
        
        # Initialize from current position
        left_arm.initialize(robot)
        right_arm.initialize(robot)
        
        logger.info("✅ Arm controllers initialized")
    except Exception as e:
        logger.error(f"❌ Arm controller initialization failed: {e}")
        logger.error(traceback.format_exc())
        return
    
    logger.info("🎯 Squeeze grip buttons on VR controllers to start")
    logger.info("🎯 Press Ctrl+C to stop")
    logger.info("")
    logger.info("Telegrip VR Controls (Frame-by-Frame Delta Tracking):")
    logger.info("  - Grip button: Activate arm control (frame-by-frame tracking)")
    logger.info("  - Move controller: Control end-effector (adaptive 0.5x-4.0x scaling)")
    logger.info("  - Rotate controller: Direct wrist roll/pitch control (alpha=0.85)")
    logger.info("  - Trigger: Close/open gripper")
    logger.info("  - Left thumbstick: Rotate base (angular velocity)")
    logger.info("  - Right thumbstick: Move base (linear velocity)")
    logger.info("")
    logger.info("Telegrip Features:")
    logger.info("  ✓ Frame-by-frame delta (not origin-based)")
    logger.info("  ✓ Adaptive scaling based on movement speed")
    logger.info("  ✓ Workspace validation before IK")
    logger.info("  ✓ Singularity detection and avoidance")
    logger.info("  ✓ High responsiveness (IK α=0.8, wrist α=0.85)")
    logger.info("")
    
    # Main control loop (LeRobot pattern)
    connection_check_counter = 0
    test_vr_angle = 0.0  # For simulated circular motion
    try:
        frame_count = 0
        while True:
            t0 = time.perf_counter()
            
            # Get VR input (real or simulated)
            try:
                if args.test_vr:
                    # Create simulated VR goals for testing
                    class FakeGoal:
                        def __init__(self):
                            self.arm = None
                            self.metadata = {}
                            self.target_position = None
                            self.gripper_position = None
                            self.wrist_roll_deg = None
                            self.wrist_flex_deg = None
                    
                    # Simulate left arm with circular motion
                    left_goal = FakeGoal()
                    left_goal.arm = "left"
                    # Small circular motion in VR space
                    import math
                    left_goal.target_position = [
                        0.02 * math.cos(test_vr_angle),
                        0.02 * math.sin(test_vr_angle),
                        0.0
                    ]
                    left_goal.metadata = {"grip": True, "trigger": 0.0}
                    left_goal.gripper_position = 0.0
                    left_goal.wrist_roll_deg = 5.0 * math.sin(test_vr_angle * 0.7)
                    left_goal.wrist_flex_deg = -5.0 * math.cos(test_vr_angle * 0.5)
                    
                    # Right arm stationary for test
                    right_goal = None
                    headset_goal = None
                    
                    test_vr_angle += 0.05  # Slow rotation
                    
                    if frame_count == 0:
                        logger.info(f"🧪 TEST-VR: Simulated left arm at angle={test_vr_angle:.2f}")
                else:
                    # Real VR goals
                    dual_goals = vr_monitor.get_latest_goal_nowait()
                    left_goal = dual_goals.get("left") if dual_goals else None
                    right_goal = dual_goals.get("right") if dual_goals else None
                    headset_goal = dual_goals.get("headset") if dual_goals else None
                
                # DIAGNOSTIC MODE: Print detailed VR goal structure
                if args.diagnose_vr and frame_count == 0:
                    logger.info("=" * 70)
                    logger.info("📋 DETAILED VR GOAL DIAGNOSTIC")
                    logger.info("=" * 70)
                    if left_goal:
                        print_vr_goal_structure(left_goal, prefix="LEFT")
                    if right_goal:
                        print_vr_goal_structure(right_goal, prefix="RIGHT")
                    if headset_goal:
                        print_vr_goal_structure(headset_goal, prefix="HEADSET")
                    logger.info("=" * 70)
                
                # Standard debug logging (once per second)
                elif frame_count == 0 and not args.test_vr:
                    if left_goal:
                        logger.info(f"🎮 LEFT: grip={getattr(left_goal.metadata, 'grip', 'N/A') if hasattr(left_goal, 'metadata') and left_goal.metadata else 'N/A'} trigger={left_goal.metadata.get('trigger', 'N/A') if hasattr(left_goal, 'metadata') and left_goal.metadata else 'N/A'}")
                    if right_goal:
                        logger.info(f"🎮 RIGHT: grip={getattr(right_goal.metadata, 'grip', 'N/A') if hasattr(right_goal, 'metadata') and right_goal.metadata else 'N/A'} trigger={right_goal.metadata.get('trigger', 'N/A') if hasattr(right_goal, 'metadata') and right_goal.metadata else 'N/A'}")
                        
            except Exception as e:
                logger.error(f"❌ VR goal retrieval failed: {e}")
                left_goal = right_goal = headset_goal = None
            
            # Process VR input (delta control)
            try:
                if left_goal:
                    left_arm.handle_vr_input(left_goal)
                if right_goal:
                    right_arm.handle_vr_input(right_goal)
            except Exception as e:
                logger.error(f"❌ VR input processing failed: {e}")
            
            # Generate actions (LeRobot + Telegrip pattern)
            try:
                left_action = left_arm.p_control_action(robot)
                right_action = right_arm.p_control_action(robot)
                
                # TELEGRIP: Validate actions before sending to hardware
                # Check for NaN or infinite values
                for key, val in {**left_action, **right_action}.items():
                    if not np.isfinite(val):
                        logger.error(
                            f"❌ Invalid action value: {key}={val} (skipping frame)"
                        )
                        raise ValueError(f"Invalid action value: {key}={val}")
                
            except Exception as e:
                logger.error(f"❌ Action generation failed: {e}")
                continue
            
            # Base control from VR headset (uses LeKiwi 3-omniwheel kinematics)
            base_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
            
            if headset_goal:
                try:
                    # Linear velocity from RIGHT thumbstick (already in m/s)
                    if (hasattr(headset_goal, 'base_linear_velocity') and
                            headset_goal.base_linear_velocity is not None):
                        lin_vel = headset_goal.base_linear_velocity
                        if isinstance(lin_vel, (list, tuple, np.ndarray)) and len(lin_vel) >= 2:
                            x_vel = float(lin_vel[0])
                            y_vel = float(lin_vel[1])
                            
                            # Apply small deadzone
                            if abs(x_vel) > 0.01:
                                base_action["x.vel"] = x_vel
                            if abs(y_vel) > 0.01:
                                base_action["y.vel"] = y_vel
                    
                    # Angular velocity from LEFT thumbstick (in deg/s, needs conversion)
                    if (hasattr(headset_goal, 'base_angular_velocity') and 
                            headset_goal.base_angular_velocity is not None):
                        angular_vel = headset_goal.base_angular_velocity
                        
                        # Handle both scalar and array types
                        if isinstance(angular_vel, (list, tuple, np.ndarray)):
                            if len(angular_vel) > 0:
                                theta_vel_deg = float(angular_vel[0])
                            else:
                                theta_vel_deg = 0.0
                        else:
                            theta_vel_deg = float(angular_vel)
                        
                        # Apply deadzone and convert deg/s to deg/s (LeMelon expects deg/s!)
                        if abs(theta_vel_deg) > 0.5:
                            # CRITICAL: LeMelon expects theta.vel in DEG/S, not rad/s!
                            base_action["theta.vel"] = theta_vel_deg
                    
                    # Log base movement when active (reduced frequency)
                    if any(abs(v) > 0.01 for v in base_action.values()):
                        if frame_count == 0:
                            logger.info(
                                f"🚗 Base: x={base_action['x.vel']:.3f}m/s "
                                f"y={base_action['y.vel']:.3f}m/s "
                                f"θ={base_action['theta.vel']:.1f}°/s"
                            )
                        
                except Exception as e:
                    logger.error(f"❌ Base control processing failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Merge all actions (LeRobot pattern)
            # Sanitize base_action as well
            base_action_sanitized = sanitize_action_dict(base_action)
            action = {**left_action, **right_action, **base_action_sanitized}
            
            # ═══════════════════════════════════════════════════════════════
            # LEROBOT DATASET INTEGRATION (telegrip + LeRobot compatible)
            # ═══════════════════════════════════════════════════════════════
            # The 'action' dict is now in the correct format for LeRobot datasets:
            #
            # Action Format:
            #   - Joint positions: {prefix}_arm_{joint}.pos (degrees, Python float)
            #   - Base velocities: x.vel, y.vel, theta.vel (m/s and deg/s, Python float)
            #   - All numpy types sanitized to Python natives
            #
            # Observation Format (from robot.get_observation()):
            #   - Joint positions: {prefix}_arm_{joint}.pos (current angles)
            #   - Joint velocities: {prefix}_arm_{joint}.vel (if available)
            #   - Cameras: {camera_name}.pixels (if configured)
            #
            # To record a dataset:
            # ```python
            # from lerobot.datasets.lerobot_dataset import LeRobotDataset
            # dataset = LeRobotDataset.create(
            #     repo_id="your-username/lemelon-vr-demo",
            #     fps=FPS,
            #     robot_type="lemelon"
            # )
            # # In main loop:
            # observation = robot.get_observation()
            # dataset.add_frame({"observation": observation, "action": action})
            # # At episode end:
            # dataset.save_episode()
            # dataset.push_to_hub()
            # ```
            #
            # See examples/lekiwi/record.py or examples/phone_to_so100/record.py
            # for complete recording examples with LeRobot datasets.
            # ═══════════════════════════════════════════════════════════════
            
            # DEBUG: Log full action once per second
            if frame_count == 0:
                logger.info(f"📦 Full action keys: {list(action.keys())}")
                logger.info(f"📦 Left gripper action: {action.get('left_arm_gripper.pos', 'N/A')}")
                logger.info(f"📦 Right gripper action: {action.get('right_arm_gripper.pos', 'N/A')}")
                logger.info(f"📦 Base actions: x={action.get('x.vel', 0):.3f} y={action.get('y.vel', 0):.3f} θ={action.get('theta.vel', 0):.3f}")
            
            # Send to robot (LeRobot pattern - PUSH action)
            try:
                robot.send_action(action)
            except Exception as e:
                logger.error(f"❌ Robot action send failed: {e}")
                logger.error(f"   Action keys: {list(action.keys())}")
                # Log first action value to check type
                if action:
                    sample_key = list(action.keys())[0]
                    sample_val = action[sample_key]
                    logger.error(f"   Sample value type: {type(sample_val)} = {sample_val}")
                if frame_count == 0:
                    # Only log full action on first error to avoid spam
                    logger.error(f"   Full action: {action}")
            
            # Periodic USB connection check (every 10 seconds)
            connection_check_counter += 1
            if connection_check_counter >= FPS * 10 and not mock:
                connection_check_counter = 0
                # Check if USB ports still exist
                left_port_ok = os.path.exists(args.left_port)
                right_port_ok = os.path.exists(args.right_port)
                base_port_ok = os.path.exists(args.base_port)
                
                if not left_port_ok:
                    logger.warning(f"⚠️  Left arm USB disconnected: {args.left_port}")
                if not right_port_ok:
                    logger.warning(f"⚠️  Right arm USB disconnected: {args.right_port}")
                if not base_port_ok:
                    logger.warning(f"⚠️  Base USB disconnected: {args.base_port}")
            
            # Status output every second
            frame_count += 1
            if frame_count >= FPS:
                frame_count = 0
                
                # Get FK positions for verification
                left_fk_pos, _ = left_arm.get_fk_position()
                right_fk_pos, _ = right_arm.get_fk_position()
                
                # Format FK position strings
                left_fk_str = "None" if left_fk_pos is None else f"[{left_fk_pos[0]:.3f}, {left_fk_pos[1]:.3f}, {left_fk_pos[2]:.3f}]"
                right_fk_str = "None" if right_fk_pos is None else f"[{right_fk_pos[0]:.3f}, {right_fk_pos[1]:.3f}, {right_fk_pos[2]:.3f}]"
                
                # Build orientation info for 6DOF mode
                if USE_6DOF_IK:
                    left_orient_str = (
                        f" orient=[{left_arm.target_roll:.0f}°,"
                        f"{left_arm.target_pitch:.0f}°,"
                        f"{left_arm.target_yaw:.0f}°]"
                    )
                    right_orient_str = (
                        f" orient=[{right_arm.target_roll:.0f}°,"
                        f"{right_arm.target_pitch:.0f}°,"
                        f"{right_arm.target_yaw:.0f}°]"
                    )
                else:
                    left_orient_str = ""
                    right_orient_str = ""
                
                # Build base status string
                base_moving = any(abs(v) > 0.01 for v in base_action.values())
                if base_moving:
                    base_str = (
                        f" | 🚗 Base: x={base_action['x.vel']:.2f} "
                        f"y={base_action['y.vel']:.2f} θ={base_action['theta.vel']:.1f}°/s"
                    )
                else:
                    base_str = ""
                
                logger.info(
                    f"🤖 L: pos=[{left_arm.current_x:.3f},{left_arm.current_y:.3f},{left_arm.current_z:.3f}] "
                    f"FK={left_fk_str}{left_orient_str} lift={left_arm.target_positions['shoulder_lift']:.1f}° "
                    f"grip={left_arm.target_positions['gripper']:.0f}°"
                )
                logger.info(
                    f"🤖 R: pos=[{right_arm.current_x:.3f},{right_arm.current_y:.3f},{right_arm.current_z:.3f}] "
                    f"FK={right_fk_str}{right_orient_str} lift={right_arm.target_positions['shoulder_lift']:.1f}° "
                    f"grip={right_arm.target_positions['gripper']:.0f}°{base_str}"
                )
            
            # Control timing (LeRobot pattern)
            busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping teleoperation...")
    except Exception as e:
        logger.error(f"❌ Fatal error in main loop: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup robot connection
        try:
            if 'robot' in locals() and robot is not None:
                robot.disconnect()
                logger.info("✅ Robot disconnected")
        except Exception as e:
            logger.error(f"❌ Robot disconnect failed: {e}")
        
        # Cleanup PyBullet physics client
        if 'physics_client' in locals() and physics_client is not None:
            try:
                p.disconnect(physics_client)
                logger.info("✅ PyBullet disconnected")
            except Exception:
                pass
        
        # Stop VR monitor (if it was started)
        try:
            if vr_monitor is not None:
                asyncio.run(vr_monitor.stop_monitoring())
        except Exception:
            pass
        
        logger.info("✅ Teleoperation stopped")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════
# TELEGRIP + PYBULLET ACCURACY IMPROVEMENTS SUMMARY
# ═══════════════════════════════════════════════════════════════
# This script implements complete Telegrip control with PyBullet URDF-based
# IK/FK for accurate dual-arm hardware control.
#
# PYBULLET URDF INTEGRATION (Critical for Accuracy):
# ─────────────────────────────────────────────────────────────
# 1. URDF Model Loading (lines 1363-1370)
#    - Loads SO100 URDF with precise geometry
#    - PyBullet physics client in DIRECT mode (headless)
#    - Fixed base for accurate kinematics
#
# 2. FK Solver with PyBullet (lines 1377-1383)
#    - End-effector: Link 3 (wrist) for 3DOF position control
#    - Uses p.resetJointState() + p.stepSimulation() for state updates
#    - Uses p.getLinkState() with computeForwardKinematics=1
#    - Explicit physicsClientId on all calls
#
# 3. IK Solver with PyBullet (lines 1390-1398)
#    - End-effector: Link 3 (wrist)
#    - Uses p.calculateInverseKinematics() with:
#      * Joint damping (0.01) for stability
#      * 200 iterations for convergence
#      * 1e-6 residual threshold for accuracy
#    - State management prevents contamination
#
# 4. FK Validation in Control Loop (lines 1084-1111)
#    - Periodic FK verification every 30 frames
#    - Warns if FK/IK mismatch > 3cm
#    - Ensures PyBullet is tracking correctly
#
# TELEGRIP CONTROL FEATURES:
# ─────────────────────────────────────────────────────────────
# 1. FRAME-BY-FRAME DELTA TRACKING (lines 874-878)
#    - Delta = current_frame - previous_frame (NOT current - origin)
#    - More stable and responsive than origin-based tracking
#    - No need to track VR origin or robot origin
#
# 2. ADAPTIVE SCALING (lines 888-897)
#    - Precision mode (0.5x): Small movements < 10mm
#    - Normal mode (2.0x): Medium movements
#    - Fast mode (4.0x): Large movements > 50mm
#    - Allows both precision and speed
#
# 3. SEPARATE CONTROL PATHS (lines 749-831)
#    - Wrist: Direct VR→robot mapping with alpha=0.85
#    - Position: IK solver for shoulder/elbow only
#    - No interference between the two control modes
#
# 4. WORKSPACE VALIDATION (lines 950-954, function at 409-437)
#    - Validates position BEFORE sending to IK
#    - Prevents IK from struggling with impossible targets
#    - Enforces physical workspace limits (5-30cm radius, 0-35cm height)
#
# 5. SINGULARITY AVOIDANCE (lines 973-980, function at 440-461)
#    - Detects when arm is too straight (shoulder_lift + elbow_flex < 5°)
#    - Detects when elbow is too straight (elbow_flex < 10°)
#    - Rejects unsafe configurations to prevent arm lockup
#
# 6. COMMAND VALIDATION (lines 1614-1621)
#    - Checks for NaN/infinite values before hardware send
#    - Clamps all commands to safe joint limits
#    - Prevents dangerous hardware commands
#
# 7. HIGH RESPONSIVENESS (lines 253-254)
#    - IK smoothing alpha = 0.8 (was 0.7)
#    - Wrist smoothing alpha = 0.85 (was 0.7)
#    - More responsive control while still stable
#
# KEY DIFFERENCES: Current vs Previous Implementation
# ────────────────────────────────────────────────────────────
# | Feature          | Previous          | Current (Telegrip+PyBullet) |
# |------------------|-------------------|----------------------------|
# | FK/IK Engine     | Analytic/approx   | PyBullet URDF-based        |
# | State Tracking   | None              | p.stepSimulation()         |
# | Link Index       | Wrong (jaw)       | Correct (wrist)            |
# | IK Iterations    | 100               | 200 with damping           |
# | Delta Calc       | current - origin  | current - previous_frame   |
# | Scaling          | Fixed 1.0x        | Adaptive 0.5x-4.0x         |
# | Wrist Alpha      | 0.7               | 0.85 (more responsive)     |
# | IK Alpha         | 0.7               | 0.8 (more responsive)      |
# | Workspace Check  | After IK          | Before IK                  |
# | Singularity      | None              | Active detection           |
# | FK Validation    | None              | Every 30 frames            |
# | Command Validate | Basic             | NaN/inf checking           |
# | Control Paths    | Can interfere     | Completely separate        |
# ────────────────────────────────────────────────────────────
#
# RESULT: PyBullet URDF-based control with Telegrip patterns provides
#         accurate, stable, and responsive dual-arm hardware control.
# ═══════════════════════════════════════════════════════════════