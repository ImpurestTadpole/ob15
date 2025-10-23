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

from dataclasses import dataclass, field
from typing import Dict

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig


def ob15_cameras_config_single() -> Dict[str, CameraConfig]:
    """
    Single camera config (mast only) - Best for high-speed control.
    Use this for real-time teleoperation at 30+ Hz.
    """
    return {
        "mast": OpenCVCameraConfig(
            index_or_path="/dev/video0",
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
    }


def ob15_cameras_config_dual() -> Dict[str, CameraConfig]:
    """
    Dual camera config (mast + left_wrist) - Good balance.
    Use this for teleoperation with binocular vision.
    """
    return {
        "mast": OpenCVCameraConfig(
            index_or_path="/dev/video0",
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
        "left_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2",
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
    }


def ob15_cameras_config_all() -> Dict[str, CameraConfig]:
    """
    All 3 cameras config - Optimized for USB 2.0.
    
    Note: Innomaker cameras output 640x480 regardless of requested resolution.
    All 3 cameras streaming simultaneously @ 30fps works on USB 2.0.
    """
    return {
        "mast": OpenCVCameraConfig(
            index_or_path="/dev/video0",  # GENERAL WEBCAM
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
        "left_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2",  # Innomaker #1
            fps=30,
            width=640,  # Cameras output 640x480 natively
            height=480,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video4",  # Innomaker #2
            fps=30,
            width=640,  # Cameras output 640x480 natively
            height=480,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
    }


def ob15_cameras_config() -> Dict[str, CameraConfig]:
    """
    Default camera configuration for OB15 (dual cameras).
    
    Uses 2 cameras by default (most reliable on USB 2.0):
    - mast: GENERAL WEBCAM at /dev/video0 (top-down view)
    - left_wrist: Innomaker-U20CAM at /dev/video2 (left arm wrist)
    
    Note: right_wrist (/dev/video4) often times out due to USB bandwidth.
    
    For other configurations:
    - ob15_cameras_config_single(): Mast only (best performance)
    - ob15_cameras_config_all(): All 3 cameras (use --cameras all)
    """
    return ob15_cameras_config_dual()


def ob15_cameras_config_triple() -> Dict[str, CameraConfig]:
    """
    Triple camera config (all cameras) - Requires USB 3.0.
    
    WARNING: This may timeout on USB 2.0 due to bandwidth limits.
    Only use if you have:
    1. USB 3.0 hub with sufficient bandwidth
    2. Cameras on separate USB controllers
    3. Tolerance for reduced FPS
    """
    return {
        "mast": OpenCVCameraConfig(
            index_or_path="/dev/video0",  # GENERAL WEBCAM (top-down)
            fps=10,  # Heavily reduced for 3-camera USB bandwidth
            width=320,  # Low res for bandwidth
            height=240,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
        "left_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2",  # Innomaker-U20CAM #1
            fps=10,
            width=320,
            height=240,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video4",  # Innomaker-U20CAM #2
            fps=10,
            width=320,
            height=240,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
    }


@RobotConfig.register_subclass("ob15")
@dataclass
class OB15Config(RobotConfig):
    """
    Configuration for OB15 dual-arm robot system.
    
    OB15 is a dual-arm robot with a 3-wheel omnidirectional base.
    Each arm has 6 DOF with grippers, and the base can move in any direction.
    """
    
    # Port configuration for 3-bus setup
    # ACTUAL DETECTED CONFIGURATION (from scan_motors.py):
    # /dev/ttyACM0: IDs 1-6 (LEFT ARM) ✅
    # /dev/ttyACM1: IDs 7-9 (BASE - was swapped!)
    # /dev/ttyACM2: IDs 1-6 (RIGHT ARM - was swapped!)
    port_left_arm: str = "/dev/ttyACM0"   # Left arm servo bus (IDs 1-6)
    port_right_arm: str = "/dev/ttyACM2"  # Right arm servo bus (IDs 1-6) - SWAPPED!
    port_base: str = "/dev/ttyACM1"       # Base wheel servo bus (IDs 7-9) - SWAPPED!
    
    # Safety and control settings
    disable_torque_on_disconnect: bool = True
    max_relative_target: int | None = None  # Safety limit for position changes
    
    # Camera configuration
    cameras: Dict[str, CameraConfig] = field(default_factory=ob15_cameras_config)
    
    # Backward compatibility
    use_degrees: bool = False  # Set to True for legacy policies/datasets
    
    # Mock mode for testing without hardware
    mock: bool = False
    
    # Base kinematics configuration
    base_max_forward_vel: float = 0.4   # m/s
    base_max_lateral_vel: float = 0.4   # m/s
    base_max_angular_vel: float = 1.0   # rad/s
    base_wheel_radius: float = 0.05     # m
    base_radius: float = 0.125          # m
    base_deadzone: float = 0.05         # Deadzone for joystick input
    
    # Keyboard teleoperation keys
    teleop_keys: Dict[str, str] = field(
        default_factory=lambda: {
            # Base movement (arrow keys and numpad)
            "forward": "i",
            "backward": "k", 
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            # Speed control
            "speed_up": "p",
            "speed_down": ";",
            # Quit teleop
            "quit": "q",
        }
    )


@RobotConfig.register_subclass("ob15_client")
@dataclass
class OB15ClientConfig(RobotConfig):
    """
    Configuration for OB15 Client robot.
    
    This is a client for remote control of OB15 dual-arm robot system.
    """
    
    # Network configuration
    remote_ip: str = "localhost"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    
    # Connection settings
    polling_timeout_ms: int = 1000
    connect_timeout_s: int = 5
    
    # Camera configuration
    cameras: Dict[str, CameraConfig] = field(default_factory=ob15_cameras_config)
    
    # Backward compatibility
    use_degrees: bool = False  # Set to True for legacy policies/datasets
    
    # Mock mode for testing without hardware
    mock: bool = False
    
    # Keyboard teleoperation keys
    teleop_keys: Dict[str, str] = field(
        default_factory=lambda: {
            # Base movement (arrow keys and numpad)
            "forward": "i",
            "backward": "k", 
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            # Speed control
            "speed_up": "p",
            "speed_down": ";",
            # Quit teleop
            "quit": "q",
        }
    )


@RobotConfig.register_subclass("ob15_host")
@dataclass
class OB15HostConfig(RobotConfig):
    """
    Configuration for OB15 Host robot.
    
    This is a host configuration for remote control of OB15 dual-arm robot system.
    """
    
    # Network configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    ws_host: str = "0.0.0.0"
    ws_port: int = 8442
    
    # Connection settings
    connection_time_s: int = 3600  # 1 hour default
    watchdog_timeout_ms: int = 1000
    max_loop_freq_hz: int = 30
    
    # Camera configuration
    cameras: Dict[str, CameraConfig] = field(default_factory=ob15_cameras_config)
    
    # Backward compatibility
    use_degrees: bool = False  # Set to True for legacy policies/datasets
    
    # Mock mode for testing without hardware
    mock: bool = False