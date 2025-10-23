# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OB15 Robot Implementation
LeRobot-compatible dual-arm robot with omnidirectional base
"""

import logging
import threading
import time
from functools import cached_property
from typing import Any, Dict, Tuple, Optional

import numpy as np

from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig
from lerobot.motors import MotorCalibration
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.utils.errors import DeviceNotConnectedError

from .config_ob15 import OB15Config

logger = logging.getLogger(__name__)


class _MockBus:
    """Mock bus for testing without hardware"""
    
    def __init__(self, name="mockbus"):
        self.name = name
        self.is_connected = False
        self.last_write = None
        self.lock = threading.Lock()

    def write(self, cmd):
        with self.lock:
            self.last_write = cmd

    def read(self):
        with self.lock:
            return self.last_write

    def connect(self):
        self.is_connected = True

    def disconnect(self, disable_torque=True):
        self.is_connected = False

    def sync_read(self, register, motors):
        """Mock sync_read that returns dummy values"""
        return {motor: 0.0 for motor in motors}

    def sync_write(self, register, values):
        """Mock sync_write"""
        pass

    def disable_torque(self, motors=None):
        """Mock disable_torque"""
        pass

    def enable_torque(self, motors=None):
        """Mock enable_torque"""
        pass


class OB15(Robot):
    """
    OB15 dual-arm robot with omnidirectional base.
    
    Features:
    - Dual 6-DOF arms with grippers
    - 3-wheel omnidirectional base
    - Camera integration
    - Mock mode for testing
    """

    config_class = OB15Config
    name = "ob15"

    def __init__(self, config: OB15Config):
        super().__init__(config)
        self.config = config
        
        # Hardware connections
        self.left_arm_bus = None
        self.right_arm_bus = None
        self.base_bus = None
        
        # Cameras
        self.cameras = {}
        
        # State tracking
        self._is_connected = False
        self._is_calibrated = False
        self._action_writer_thread = None
        self._action_queue = []
        self._action_lock = threading.Lock()
        
        # Motor configurations
        self._setup_motor_configs()

    def _setup_motor_configs(self):
        """
        Setup motor configurations for arms and base
        
        OB15 uses 3 separate buses:
        - Left arm bus: motor IDs 1-6
        - Right arm bus: motor IDs 1-6 (same IDs, different bus)
        - Base bus: motor IDs 7, 8, 9
        """
        # Left arm motors (IDs 1-6 on left arm bus)
        self.left_arm_motors = {
            "shoulder_pan": {"id": 1, "min": -180, "max": 180},
            "shoulder_lift": {"id": 2, "min": -90, "max": 90},
            "elbow_flex": {"id": 3, "min": -90, "max": 90},
            "wrist_flex": {"id": 4, "min": -90, "max": 90},
            "wrist_roll": {"id": 5, "min": -180, "max": 180},
            "gripper": {"id": 6, "min": 0, "max": 90},
        }
        
        # Right arm motors (IDs 1-6 on right arm bus - same IDs, different bus!)
        self.right_arm_motors = {
            "shoulder_pan": {"id": 1, "min": -180, "max": 180},
            "shoulder_lift": {"id": 2, "min": -90, "max": 90},
            "elbow_flex": {"id": 3, "min": -90, "max": 90},
            "wrist_flex": {"id": 4, "min": -90, "max": 90},
            "wrist_roll": {"id": 5, "min": -180, "max": 180},
            "gripper": {"id": 6, "min": 0, "max": 90},
        }
        
        # Base motors (IDs 7, 8, 9 on base bus)
        self.base_motors = {
            "left_wheel": {"id": 7, "min": -1023, "max": 1023},
            "back_wheel": {"id": 8, "min": -1023, "max": 1023},
            "right_wheel": {"id": 9, "min": -1023, "max": 1023},
        }

    @property
    def _state_ft(self) -> Dict[str, type]:
        """Define state features for LeRobot compatibility"""
        return {
            # Left arm joints
            "left_arm_shoulder_pan.pos": float,
            "left_arm_shoulder_lift.pos": float,
            "left_arm_elbow_flex.pos": float,
            "left_arm_wrist_flex.pos": float,
            "left_arm_wrist_roll.pos": float,
            "left_arm_gripper.pos": float,
            # Right arm joints
            "right_arm_shoulder_pan.pos": float,
            "right_arm_shoulder_lift.pos": float,
            "right_arm_elbow_flex.pos": float,
            "right_arm_wrist_flex.pos": float,
            "right_arm_wrist_roll.pos": float,
            "right_arm_gripper.pos": float,
            # Base velocities
            "x.vel": float,
            "y.vel": float,
            "theta.vel": float,
        }

    @cached_property
    def _cameras_ft(self) -> Dict[str, Tuple]:
        """Define camera features from config"""
        cameras_ft = {}
        if hasattr(self.config, 'cameras'):
            for cam_name, cam_config in self.config.cameras.items():
                height = getattr(cam_config, 'height', 480)
                width = getattr(cam_config, 'width', 640)
                cameras_ft[cam_name] = (height, width, 3)  # HWC format
        return cameras_ft

    @cached_property
    def observation_features(self) -> Dict[str, type | Tuple]:
        """Observation features combining state and cameras"""
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> Dict[str, type]:
        """Action features"""
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected"""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """
        Check if robot is calibrated
        Following LeKiwi pattern: checks all buses
        """
        if isinstance(self.left_arm_bus, _MockBus):
            return True  # Mock mode is always "calibrated"
        
        # Check if all real buses are calibrated
        left_calibrated = (
            self.left_arm_bus and 
            hasattr(self.left_arm_bus, 'is_calibrated') and 
            self.left_arm_bus.is_calibrated
        )
        right_calibrated = (
            self.right_arm_bus and 
            hasattr(self.right_arm_bus, 'is_calibrated') and 
            self.right_arm_bus.is_calibrated
        )
        # Base doesn't need calibration (velocity mode)
        
        return left_calibrated and right_calibrated

    def configure(self) -> None:
        """
        Configure robot motors (operating modes, PID settings)
        Following LeKiwi pattern from https://huggingface.co/docs/lerobot/lekiwi
        """
        if isinstance(self.left_arm_bus, _MockBus):
            return  # Skip configuration in mock mode
        
        logger.info("🔧 Configuring OB15 robot...")
        
        from lerobot.motors.feetech import OperatingMode
        
        try:
            # Configure left arm
            if self.left_arm_bus and hasattr(self, 'left_arm_motor_names'):
                self.left_arm_bus.disable_torque()
                self.left_arm_bus.configure_motors()
                for name in self.left_arm_motor_names:
                    self.left_arm_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                    # Lower P coefficient to avoid shakiness (default is 32)
                    self.left_arm_bus.write("P_Coefficient", name, 16)
                    self.left_arm_bus.write("I_Coefficient", name, 0)
                    self.left_arm_bus.write("D_Coefficient", name, 32)
                self.left_arm_bus.enable_torque()
            
            # Configure right arm
            if self.right_arm_bus and hasattr(self, 'right_arm_motor_names'):
                self.right_arm_bus.disable_torque()
                self.right_arm_bus.configure_motors()
                for name in self.right_arm_motor_names:
                    self.right_arm_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                    self.right_arm_bus.write("P_Coefficient", name, 16)
                    self.right_arm_bus.write("I_Coefficient", name, 0)
                    self.right_arm_bus.write("D_Coefficient", name, 32)
                self.right_arm_bus.enable_torque()
            
            # Configure base (velocity mode)
            if self.base_bus and hasattr(self, 'base_motor_names'):
                self.base_bus.disable_torque()
                self.base_bus.configure_motors()
                for name in self.base_motor_names:
                    self.base_bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
                self.base_bus.enable_torque()
            
            logger.info("✅ OB15 robot configured")
            
        except Exception as e:
            logger.warning(f"⚠️  Configuration failed: {e}")

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to robot hardware
        Following LeKiwi pattern: connect -> configure -> calibrate
        Reference: https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/lekiwi/lekiwi.py
        """
        logger.info("🔌 Connecting to OB15 robot...")
        
        try:
            if self.config.mock:
                logger.info("🎭 Using mock robot mode")
                self._setup_mock_connections()
            else:
                self._setup_real_connections()
            
            # Initialize cameras
            self._setup_cameras()
            
            self._is_connected = True
            
            # Configure motors BEFORE calibration (like LeKiwi)
            if not self.config.mock:
                self.configure()
            
            # Calibrate if needed (checks is_calibrated internally)
            if calibrate and not self.is_calibrated:
                logger.info(
                    "Mismatch between calibration values in the motor and the calibration file "
                    "or no calibration file found"
                )
                self.calibrate()
            
            logger.info("✅ OB15 robot connected")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to OB15: {e}")
            self._is_connected = False
            raise

    def _setup_mock_connections(self):
        """Setup mock connections for testing"""
        self.left_arm_bus = _MockBus("left_arm")
        self.right_arm_bus = _MockBus("right_arm")
        self.base_bus = _MockBus("base")
        
        self.left_arm_bus.connect()
        self.right_arm_bus.connect()
        self.base_bus.connect()

    def _setup_real_connections(self):
        """
        Setup real hardware connections using Feetech motors
        Following LeKiwi pattern from https://huggingface.co/docs/lerobot/lekiwi
        """
        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
            
            logger.info("🔌 Setting up real hardware connections...")
            
            # Create motor name lists for each bus
            self.left_arm_motor_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper"
            ]
            self.right_arm_motor_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper"
            ]
            self.base_motor_names = ["left_wheel", "back_wheel", "right_wheel"]
            
            # Load calibration if available
            left_calibration = self._get_calibration_for_bus("left_arm")
            right_calibration = self._get_calibration_for_bus("right_arm")
            base_calibration = self._get_calibration_for_bus("base")
            
            # Left arm bus
            logger.info(f"   Connecting to left arm: {self.config.port_left_arm}")
            norm_mode = MotorNormMode.DEGREES if self.config.use_degrees else MotorNormMode.RANGE_M100_100
            self.left_arm_bus = FeetechMotorsBus(
                port=self.config.port_left_arm,
                motors={
                    "shoulder_pan": Motor(1, "sts3215", norm_mode),
                    "shoulder_lift": Motor(2, "sts3215", norm_mode),
                    "elbow_flex": Motor(3, "sts3215", norm_mode),
                    "wrist_flex": Motor(4, "sts3215", norm_mode),
                    "wrist_roll": Motor(5, "sts3215", norm_mode),
                    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                },
                calibration=left_calibration,
            )
            self.left_arm_bus.connect()
            logger.info("   ✅ Left arm connected")
            
            # Right arm bus (IDs 1-6, same as left arm - different bus!)
            logger.info(f"   Connecting to right arm: {self.config.port_right_arm}")
            self.right_arm_bus = FeetechMotorsBus(
                port=self.config.port_right_arm,
                motors={
                    "shoulder_pan": Motor(1, "sts3215", norm_mode),
                    "shoulder_lift": Motor(2, "sts3215", norm_mode),
                    "elbow_flex": Motor(3, "sts3215", norm_mode),
                    "wrist_flex": Motor(4, "sts3215", norm_mode),
                    "wrist_roll": Motor(5, "sts3215", norm_mode),
                    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                },
                calibration=right_calibration,
            )
            self.right_arm_bus.connect()
            logger.info("   ✅ Right arm connected")
            
            # Base bus (IDs 7, 8, 9)
            logger.info(f"   Connecting to base: {self.config.port_base}")
            self.base_bus = FeetechMotorsBus(
                port=self.config.port_base,
                motors={
                    "left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
                    "back_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
                    "right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
                },
                calibration=base_calibration,
            )
            self.base_bus.connect()
            logger.info("   ✅ Base connected")
            
            logger.info("✅ All hardware connections established")
            
        except Exception as e:
            logger.error(f"❌ Real hardware connection failed: {e}")
            logger.warning("⚠️  Falling back to mock mode")
            self._setup_mock_connections()
    
    def _get_calibration_for_bus(self, bus_name: str) -> Optional[Dict[str, MotorCalibration]]:
        """Get calibration data for a specific bus from the calibration file"""
        if not self.calibration:
            return None
        
        # Filter calibration data for motors on this bus
        bus_calibration = {}
        if bus_name == "left_arm":
            for motor_name in self.left_arm_motor_names:
                full_name = f"left_{motor_name}"
                if full_name in self.calibration:
                    bus_calibration[motor_name] = self.calibration[full_name]
        elif bus_name == "right_arm":
            for motor_name in self.right_arm_motor_names:
                full_name = f"right_{motor_name}"
                if full_name in self.calibration:
                    bus_calibration[motor_name] = self.calibration[full_name]
        elif bus_name == "base":
            for motor_name in self.base_motor_names:
                if motor_name in self.calibration:
                    bus_calibration[motor_name] = self.calibration[motor_name]
        
        return bus_calibration if bus_calibration else None
    
    def _setup_cameras(self) -> None:
        """
        Initialize cameras from config.
        Following LeKiwi pattern from https://huggingface.co/docs/lerobot/cameras
        """
        if not hasattr(self.config, 'cameras') or not self.config.cameras:
            logger.warning("⚠️  No cameras configured")
            return
        
        try:
            from lerobot.cameras.utils import make_cameras_from_configs
            
            logger.info(f"📷 Initializing {len(self.config.cameras)} camera(s)...")
            all_cameras = make_cameras_from_configs(self.config.cameras)
            
            # Connect cameras and only keep successfully connected ones
            self.cameras = {}
            for cam_name, cam in all_cameras.items():
                try:
                    # Connect camera (may raise FPS mismatch warning)
                    cam.connect()
                except Exception as e:
                    # Log but continue - FPS warnings are common but camera works
                    logger.debug(f"   Camera {cam_name} warning during connect: {e}")
                
                # Check if actually connected (more important than exceptions)
                try:
                    if cam.is_connected:
                        # Verify camera can read frames
                        test_frame = cam.async_read()
                        if test_frame is not None:
                            self.cameras[cam_name] = cam
                            logger.info(
                                f"   ✅ {cam_name} camera connected "
                                f"({test_frame.shape[1]}x{test_frame.shape[0]})"
                            )
                        else:
                            logger.warning(
                                f"   ⚠️  {cam_name} connected but no frames"
                            )
                    else:
                        logger.warning(
                            f"   ⚠️  {cam_name} not connected"
                        )
                except Exception as e:
                    logger.warning(f"   ⚠️  {cam_name} verification failed: {e}")
            
            logger.info(
                f"✅ Camera initialization complete "
                f"({len(self.cameras)}/{len(all_cameras)} cameras active)"
            )
            
        except Exception as e:
            logger.warning(f"⚠️  Camera setup failed: {e}")
            self.cameras = {}

    def setup_motors(self) -> None:
        """
        Setup motor IDs for OB15 robot
        Following LeKiwi pattern from https://huggingface.co/docs/lerobot/lekiwi
        """
        logger.info("🔧 Setting up motor IDs...")
        logger.info("This will configure motor IDs for:")
        logger.info("  - Left arm: IDs 1-6")
        logger.info("  - Right arm: IDs 1-6")
        logger.info("  - Base wheels: IDs 7-9")
        
        from itertools import chain
        
        # Setup left arm motors (reverse order for safety)
        if self.left_arm_bus and not isinstance(self.left_arm_bus, _MockBus):
            logger.info("\nSetting up LEFT ARM motors...")
            left_motors = ["gripper", "wrist_roll", "wrist_flex", "elbow_flex", 
                          "shoulder_lift", "shoulder_pan"]
            for motor in left_motors:
                input(f"Connect the controller board to the LEFT ARM '{motor}' motor only and press Enter.")
                self.left_arm_bus.setup_motor(motor)
                logger.info(f"✅ '{motor}' motor ID set to {self.left_arm_bus.motors[motor].id}")
        
        # Setup right arm motors
        if self.right_arm_bus and not isinstance(self.right_arm_bus, _MockBus):
            logger.info("\nSetting up RIGHT ARM motors...")
            right_motors = ["gripper", "wrist_roll", "wrist_flex", "elbow_flex",
                           "shoulder_lift", "shoulder_pan"]
            for motor in right_motors:
                input(f"Connect the controller board to the RIGHT ARM '{motor}' motor only and press Enter.")
                self.right_arm_bus.setup_motor(motor)
                logger.info(f"✅ '{motor}' motor ID set to {self.right_arm_bus.motors[motor].id}")
        
        # Setup base motors
        if self.base_bus and not isinstance(self.base_bus, _MockBus):
            logger.info("\nSetting up BASE motors...")
            base_motors = ["right_wheel", "back_wheel", "left_wheel"]
            for motor in base_motors:
                input(f"Connect the controller board to the BASE '{motor}' motor only and press Enter.")
                self.base_bus.setup_motor(motor)
                logger.info(f"✅ '{motor}' motor ID set to {self.base_bus.motors[motor].id}")
        
        logger.info("✅ Motor setup complete!")
    
    def calibrate(self) -> None:
        """
        Calibrate robot arms and base
        Following LeKiwi pattern from https://huggingface.co/docs/lerobot/lekiwi
        Reference: https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/lekiwi/lekiwi.py
        """
        logger.info("🎯 Calibrating OB15 robot...")
        
        # Skip calibration in mock mode
        if isinstance(self.left_arm_bus, _MockBus):
            logger.info("Mock mode - skipping calibration")
            self._is_calibrated = True
            return
        
        # Import required for calibration
        from lerobot.motors import MotorCalibration
        from lerobot.motors.feetech import OperatingMode
        
        # Check if calibration file exists
        if self.calibration:
            # Calibration file exists, ask user
            user_input = input(
                f"Press ENTER to use calibration file for robot '{self.id}', "
                "or type 'c' and press ENTER to run new calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using existing calibration file for {self.id}")
                # Calibration was already loaded during bus initialization
                self._is_calibrated = True
                return
        
        logger.info(f"\nRunning calibration of OB15 robot '{self.id}'")
        
        # Initialize calibration data storage
        self._calibration_data = {}
        
        # Calibrate each bus separately (LeKiwi does it all at once, but we have 3 buses)
        try:
            # Calibrate left arm
            logger.info("\n📝 Calibrating LEFT ARM...")
            self._calibrate_arm_bus(self.left_arm_bus, self.left_arm_motor_names, "left_arm")
            
            # Calibrate right arm
            logger.info("\n📝 Calibrating RIGHT ARM...")
            self._calibrate_arm_bus(self.right_arm_bus, self.right_arm_motor_names, "right_arm")
            
            # Base motors don't need calibration (velocity mode, no homing)
            logger.info("\n✅ Base motors don't require calibration (velocity mode)")
            
            # Save combined calibration to file
            self.calibration = self._calibration_data
            self._save_calibration()
            logger.info(f"✅ Calibration saved to {self.calibration_fpath}")
            
            self._is_calibrated = True
            logger.info("✅ OB15 robot calibration complete")
            
        except Exception as e:
            logger.error(f"❌ Calibration failed: {e}")
            raise
    
    def _calibrate_arm_bus(self, bus, motor_names, bus_label):
        """
        Calibrate a single arm bus following LeKiwi pattern
        Reference: https://huggingface.co/docs/lerobot/lekiwi
        """
        from lerobot.motors import MotorCalibration
        from lerobot.motors.feetech import OperatingMode
        
        # Check if calibration already exists
        if bus.is_calibrated:
            logger.info(f"  {bus_label} already calibrated")
            return
        
        # Disable torque for manual positioning
        bus.disable_torque()
        
        # Set all motors to position mode
        for name in motor_names:
            bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
        
        # Step 1: Set homing offsets
        input(f"  Move {bus_label} to the middle of its range of motion and press ENTER...")
        homing_offsets = bus.set_half_turn_homings(motor_names)
        
        # Step 2: Record range of motion
        full_turn_motors = [m for m in motor_names if any(kw in m for kw in ["wheel", "wrist"])]
        unknown_range_motors = [m for m in motor_names if m not in full_turn_motors]
        
        if unknown_range_motors:
            print(
                f"  Move {bus_label} joints (except {full_turn_motors}) sequentially through their "
                "entire ranges of motion.\n  Recording positions. Press ENTER to stop..."
            )
            range_mins, range_maxes = bus.record_ranges_of_motion(unknown_range_motors)
        else:
            range_mins, range_maxes = {}, {}
        
        # Full turn motors get full range
        for name in full_turn_motors:
            range_mins[name] = 0
            range_maxes[name] = 4095
        
        # Step 3: Create and save calibration
        bus_calibration = {}
        for name, motor in bus.motors.items():
            bus_calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets.get(name, 0),
                range_min=range_mins.get(name, 0),
                range_max=range_maxes.get(name, 4095),
            )
        
        # Write calibration to motors
        bus.write_calibration(bus_calibration)
        
        # Save to global calibration dict (will be saved to file)
        if not hasattr(self, '_calibration_data'):
            self._calibration_data = {}
        
        # Prefix motor names for global storage
        prefix = bus_label.replace("_arm", "").replace("_", "")  # "left_arm" -> "left"
        for name, calib in bus_calibration.items():
            if bus_label == "base":
                full_name = name
            else:
                full_name = f"{prefix}_{name}"
            self._calibration_data[full_name] = calib
        
        logger.info(f"  ✅ {bus_label} calibrated")

    def _move_to_zero_positions(self):
        """Move all joints to zero positions"""
        zero_action = {
            # Left arm
            "left_arm_shoulder_pan.pos": 0.0,
            "left_arm_shoulder_lift.pos": 0.0,
            "left_arm_elbow_flex.pos": 0.0,
            "left_arm_wrist_flex.pos": 0.0,
            "left_arm_wrist_roll.pos": 0.0,
            "left_arm_gripper.pos": 0.0,
            # Right arm
            "right_arm_shoulder_pan.pos": 0.0,
            "right_arm_shoulder_lift.pos": 0.0,
            "right_arm_elbow_flex.pos": 0.0,
            "right_arm_wrist_flex.pos": 0.0,
            "right_arm_wrist_roll.pos": 0.0,
            "right_arm_gripper.pos": 0.0,
            # Base
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        }
        
        self.send_action(zero_action)
        time.sleep(2.0)  # Wait for movement

    def disconnect(self) -> None:
        """Disconnect from robot"""
        logger.info("🔌 Disconnecting OB15 robot...")
        
        # Stop action writer
        if self._action_writer_thread:
            self._action_writer_thread.join(timeout=1.0)
        
        # Disconnect cameras
        for cam_name, cam in self.cameras.items():
            try:
                if cam.is_connected:
                    cam.disconnect()
                    logger.info(f"   📷 {cam_name} camera disconnected")
            except Exception as e:
                logger.warning(f"   ⚠️  Error disconnecting {cam_name}: {e}")
        
        # Disconnect buses
        if self.left_arm_bus:
            self.left_arm_bus.disconnect()
        if self.right_arm_bus:
            self.right_arm_bus.disconnect()
        if self.base_bus:
            self.base_bus.disconnect()
        
        self._is_connected = False
        logger.info("✅ OB15 robot disconnected")

    def send_action(self, action: Dict[str, Any], blocking: bool = False) -> Dict[str, Any]:
        """Send action to robot"""
        if not self.is_connected:
            logger.warning("⚠️  Robot not connected, ignoring action")
            return {}
        
        try:
            # Apply action to hardware
            self._apply_action_to_hardware(action)
            
            if blocking:
                time.sleep(0.1)  # Small delay for hardware response
            
            return action
            
        except Exception as e:
            logger.error(f"❌ Error sending action: {e}")
            return {}

    def _apply_action_to_hardware(self, action: Dict[str, Any]):
        """Apply action to hardware components"""
        # Apply arm actions
        self._apply_arm_action(action, "left", self.left_arm_bus, self.left_arm_motors)
        self._apply_arm_action(action, "right", self.right_arm_bus, self.right_arm_motors)
        
        # Apply base action
        self._apply_base_action(
            action.get("x.vel", 0.0),
            action.get("y.vel", 0.0),
            action.get("theta.vel", 0.0)
        )

    def _apply_arm_action(self, action: Dict[str, Any], arm: str, bus, motors):
        """Apply action to arm"""
        if not bus:
            return
        
        # Check if it's a mock bus or real bus
        if isinstance(bus, _MockBus):
            # Mock implementation
            if not bus.is_connected:
                return
            for joint, motor_config in motors.items():
                key = f"{arm}_arm_{joint}.pos"
                if key in action:
                    position = action[key]
                    position = max(motor_config["min"], min(motor_config["max"], position))
                    bus.write({"id": motor_config["id"], "position": position})
        else:
            # Real Feetech bus implementation
            if not bus.is_connected:
                return
            
            # Build goal positions for sync_write
            goal_positions = {}
            for joint, motor_config in motors.items():
                key = f"{arm}_arm_{joint}.pos"
                if key in action:
                    position = action[key]
                    # Clamp to motor limits
                    position = max(motor_config["min"], min(motor_config["max"], position))
                    # Use joint name as motor name
                    goal_positions[joint] = position
            
            # Send to all motors at once
            if goal_positions:
                try:
                    bus.sync_write("Goal_Position", goal_positions)
                except Exception as e:
                    logger.error(f"Error writing to {arm} arm: {e}")

    def _apply_base_action(self, x_vel: float, y_vel: float, theta_vel: float):
        """Apply base movement action"""
        if not self.base_bus or not self.base_bus.is_connected:
            return
        
        # Convert body velocities to wheel speeds (returns raw integer values)
        wheel_speeds_raw = self._body_to_wheel_raw(x_vel, y_vel, theta_vel)
        
        # Check if mock or real bus
        if isinstance(self.base_bus, _MockBus):
            # Mock implementation
            for wheel, speed in wheel_speeds_raw.items():
                motor_id = self.base_motors[wheel]["id"]
                speed = int(max(-1023, min(1023, speed)))
                self.base_bus.write({"id": motor_id, "velocity": speed})
        else:
            # Real Feetech bus implementation - send raw integer velocities
            if wheel_speeds_raw:
                try:
                    self.base_bus.sync_write("Goal_Velocity", wheel_speeds_raw)
                except Exception as e:
                    logger.error(f"Error writing to base: {e}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        """
        Convert angular velocity in deg/s to raw motor command
        Following LeKiwi implementation
        """
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap to signed 16-bit range
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF
        elif speed_int < -0x8000:
            speed_int = -0x8000
        return speed_int
    
    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        """
        Convert raw motor speed to angular velocity in deg/s
        Following LeKiwi implementation
        """
        steps_per_deg = 4096.0 / 360.0
        degps = raw_speed / steps_per_deg
        return degps
    
    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> Dict[str, int]:
        """
        Convert body-frame velocities to wheel raw commands (integers)
        Following LeKiwi implementation exactly
        Reference: https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/lekiwi/lekiwi.py
        
        Parameters:
          x: Linear velocity in x (m/s)
          y: Linear velocity in y (m/s)
          theta: Rotational velocity (deg/s)
          wheel_radius: Radius of each wheel (meters)
          base_radius: Distance from center to each wheel (meters)
          max_raw: Maximum allowed raw command per wheel
        
        Returns:
          Dictionary with integer raw commands for each wheel
        """
        # Convert theta from deg/s to rad/s
        theta_rad = theta * (np.pi / 180.0)
        velocity_vector = np.array([x, y, theta_rad])
        
        # Define wheel mounting angles with -90° offset (LeKiwi pattern)
        angles = np.radians(np.array([240, 0, 120]) - 90)
        
        # Build kinematic matrix
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        
        # Compute wheel linear speeds and angular speeds
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius
        
        # Convert to deg/s
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)
        
        # Scale if exceeding max
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats) if raw_floats else 0
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale
        
        # Convert to raw integers
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]
        
        return {
            "left_wheel": wheel_raw[0],
            "back_wheel": wheel_raw[1],
            "right_wheel": wheel_raw[2],
        }

    def get_observation(self, include_cameras: bool = True, timeout_ms: int = 1000) -> Dict[str, Any]:
        """
        Get current robot observation.
        Following LeKiwi pattern.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        observation = {}
        
        # Get joint positions from hardware or mock
        if isinstance(self.left_arm_bus, _MockBus):
            # Mock implementation
            observation.update(self._get_mock_joint_positions())
        else:
            # Real hardware implementation
            observation.update(self._get_real_joint_positions())
        
        # Get base velocities
        observation.update(self._get_base_velocities())
        
        # Capture images from cameras (LeKiwi pattern)
        # Track camera failures to avoid spamming warnings
        if not hasattr(self, '_camera_fail_counts'):
            self._camera_fail_counts = {}

        for cam_key, cam in self.cameras.items():
            try:
                start = time.perf_counter()
                frame = cam.async_read()
                if frame is not None:
                    observation[cam_key] = frame
                    # Reset fail count on success
                    self._camera_fail_counts[cam_key] = 0
                    dt_ms = (time.perf_counter() - start) * 1e3
                    logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
                else:
                    # Increment fail count
                    self._camera_fail_counts[cam_key] = \
                        self._camera_fail_counts.get(cam_key, 0) + 1
                    # Only log first few failures
                    if self._camera_fail_counts[cam_key] <= 3:
                        logger.warning(f"Camera {cam_key} returned None")
            except Exception as e:
                # Increment fail count
                self._camera_fail_counts[cam_key] = \
                    self._camera_fail_counts.get(cam_key, 0) + 1
                # Only log first few failures to avoid spam
                if self._camera_fail_counts[cam_key] <= 3:
                    logger.warning(f"Camera {cam_key}: {e}")
                elif self._camera_fail_counts[cam_key] == 4:
                    logger.warning(
                        f"Camera {cam_key}: Suppressing further warnings"
                    )
        
        return observation
    
    def _get_real_joint_positions(self) -> Dict[str, float]:
        """Get real joint positions from hardware"""
        positions = {}
        
        try:
            # Read left arm positions
            if self.left_arm_bus and self.left_arm_bus.is_connected:
                left_pos = self.left_arm_bus.sync_read("Present_Position")
                for joint, value in left_pos.items():
                    positions[f"left_arm_{joint}.pos"] = float(value)
        except Exception as e:
            logger.error(f"Error reading left arm positions: {e}")
        
        try:
            # Read right arm positions
            if self.right_arm_bus and self.right_arm_bus.is_connected:
                right_pos = self.right_arm_bus.sync_read("Present_Position")
                for joint, value in right_pos.items():
                    positions[f"right_arm_{joint}.pos"] = float(value)
        except Exception as e:
            logger.error(f"Error reading right arm positions: {e}")
        
        # Fill in any missing values with zeros
        for key in self._state_ft.keys():
            if key.endswith(".pos") and key not in positions:
                positions[key] = 0.0
        
        return positions
    
    def _get_base_velocities(self) -> Dict[str, float]:
        """Get base velocities from hardware or return zeros"""
        if isinstance(self.base_bus, _MockBus) or not self.base_bus or not self.base_bus.is_connected:
            return {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
            }
        
        try:
            # Read wheel velocities
            wheel_vel = self.base_bus.sync_read("Present_Velocity")
            
            # Convert wheel velocities to body frame
            # This is a simplified version - you may need to implement proper inverse kinematics
            return {
                "x.vel": 0.0,  # Would calculate from wheel_vel
                "y.vel": 0.0,
                "theta.vel": 0.0,
            }
        except Exception as e:
            logger.error(f"Error reading base velocities: {e}")
            return {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
            }

    def _get_mock_joint_positions(self) -> Dict[str, float]:
        """Get mock joint positions"""
        return {
            # Left arm (mock positions)
            "left_arm_shoulder_pan.pos": 0.0,
            "left_arm_shoulder_lift.pos": 0.0,
            "left_arm_elbow_flex.pos": 0.0,
            "left_arm_wrist_flex.pos": 0.0,
            "left_arm_wrist_roll.pos": 0.0,
            "left_arm_gripper.pos": 0.0,
            # Right arm (mock positions)
            "right_arm_shoulder_pan.pos": 0.0,
            "right_arm_shoulder_lift.pos": 0.0,
            "right_arm_elbow_flex.pos": 0.0,
            "right_arm_wrist_flex.pos": 0.0,
            "right_arm_wrist_roll.pos": 0.0,
            "right_arm_gripper.pos": 0.0,
        }

    def stop_base(self):
        """
        Stop base movement (emergency stop)
        Following LeKiwi pattern
        """
        logger.info("🛑 Stopping base movement")
        try:
            if isinstance(self.base_bus, _MockBus):
                # Mock implementation
                stop_action = {
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                }
                self.send_action(stop_action)
            else:
                # Real hardware - send zero velocity directly (integers!)
                if self.base_bus and self.base_bus.is_connected:
                    self.base_bus.sync_write(
                        "Goal_Velocity",
                        dict.fromkeys(self.base_motor_names, 0),
                        num_retry=5
                    )
        except Exception as e:
            logger.error(f"Error stopping base: {e}")
