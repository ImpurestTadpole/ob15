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

from lerobot.cameras.configs import CameraConfig, Cv2Rotation, ColorMode
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from ..config import RobotConfig
from .lift_axis import LiftAxisConfig


def xlerobot_cameras_config() -> dict[str, CameraConfig]:
    """
    Camera configuration using SmolVLA's standardized naming convention.
    
    Camera naming aligns with SmolVLA's expected format:
    - camera1 = top/overhead view (was "head") - matches SmolVLA's OBS_IMAGE_1
    - camera2 = wrist view (was "left_wrist") - matches SmolVLA's OBS_IMAGE_2
    - camera3 = additional view (was "right_wrist") - matches SmolVLA's OBS_IMAGE_3
    
    This naming makes the robot natively compatible with SmolVLA policies without
    needing rename_map during training or inference.
    
    Note: camera1 MUST be opened FIRST to avoid resource conflicts.
    Head camera MUST be opened FIRST to avoid resource conflicts.
    Opening it after wrist cameras causes it to fail.
    """
    return {
        # camera1: Top/overhead view (head) - Intel RealSense D435i
        # MUST be opened FIRST to avoid resource conflicts
        # Using RealSense SDK for efficient native compression (better than OpenCV/V4L2)
        "head": RealSenseCameraConfig(
            serial_number_or_name="342222071125",
            fps=30,
            width=640,
            height=360,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=False,
        ),
    
        # IMAGE SIZE RECOMMENDATION:
        # Current 640x480 is 6x more pixels than needed for most manipulation policies.
        # SmolVLA expects 224x224; ACT/Diffusion Policy typically use 224x224 or 256x256.
        # Recording at 640x480 wastes ~350 GB of storage for a 130k-frame dataset and slows
        # every training epoch significantly on the Jetson.
        #
        # Recommended sizes:
        #   head (overview)  : 320x240 — scene context, fine detail not needed
        #   wrist cameras    : 224x224 — fine manipulation, matches policy input directly
        #
        # To apply: change width/height below and re-collect data (or add a resize step in
        # the training pipeline if you want to keep existing datasets at 640x480).
        #
        # OpenCV fallback (if RealSense SDK issues occur, uncomment below and comment above):
        #"head": OpenCVCameraConfig(
        #     index_or_path="/dev/video4",
        #     fps=30,
        #     width=320,   # Reduced from 640 to handle YUYV bandwidth
        #     height=240,  # Reduced from 480 to handle YUYV bandwidth
        #     fourcc="YUYV",  # Changed from MJPG (not supported by RealSense via V4L2)
        #     rotation=Cv2Rotation.NO_ROTATION,
        # ),
        
        # camera2: Wrist view (was "left_wrist")
        # PERFORMANCE: MJPG format is critical for 30 Hz control rate
        # If MJPG fails (camera defaults to YUYV), run: ./setup_camera_formats.sh
        "left_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video8",  # Innomaker camera 2 (swapped)
            fps=30,
            width=640,   # For 30 Hz with MJPG. If using YUYV, reduce to 320x240
            height=360,  # For 30 Hz with MJPG. If using YUYV, reduce to 320x240
            fourcc="MJPG",
            rotation=Cv2Rotation.NO_ROTATION,
            warmup_s=3,  # Increased warmup time for Innomaker cameras
        ),     
        
        # camera3: Additional view (was "right_wrist")
        # PERFORMANCE: MJPG format is critical for 30 Hz control rate
        # If MJPG fails (camera defaults to YUYV), run: ./setup_camera_formats.sh
        "right_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video6",  # Innomaker camera 1 (swapped)
            fps=30,
            width=640,   # For 30 Hz with MJPG. If using YUYV, reduce to 320x240
            height=360,  # For 30 Hz with MJPG. If using YUYV, reduce to 320x240
            fourcc="MJPG",
            rotation=Cv2Rotation.NO_ROTATION,
            warmup_s=5,  # Increased warmup time for right wrist camera (needs more time after other cameras)
        ),
    }


@RobotConfig.register_subclass("xlerobot")
@dataclass
class XLerobotConfig(RobotConfig):
    
    # Port 0 = left arm + base. Port 1 = right arm + head + (optional) lift axis.
    port1: str = "/dev/ttyACM1"  # left arm motors 1-6 + base motors 7-9
    port2: str = "/dev/ttyACM0"  # right arm motors 1-6 + head motors 7-8 + lift (motor 9)
    camera_start_order: tuple[str, ...] | None = ("head", "left_wrist", "right_wrist")
    camera_start_delay_s: float = 2.0  # Increased delay to allow cameras to initialize properly (especially right_wrist)
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    # Optional gantry / Z lift axis (motor_id 9 on bus2). Activate with: lift_axis.enabled=True
    # or --robot.lift_axis.enabled=true. Calibration = homing during calibrate(); control =
    # gantry.height_mm (target mm) or gantry.vel; recorded as observation + action.
    lift_axis: LiftAxisConfig = field(default_factory=LiftAxisConfig)

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "i",
            "backward": "k",
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            # Speed control
            "speed_up": "n",
            "speed_down": "m",
            # quit teleop
            "quit": "b",
        }
    )


# ZMQ bridge: host on robot (Jetson), client on GPU PC.


@dataclass
class XLerobotHostConfig:
    """ZMQ ports bound on the robot machine. Client connects to these."""

    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    connection_time_s: int = 3600
    watchdog_timeout_ms: int = 500
    max_loop_freq_hz: int = 30
    jpeg_quality: int = 90


@RobotConfig.register_subclass("xlerobot_client")
@dataclass
class XLerobotClientConfig(RobotConfig):
    """Remote XLerobot over ZMQ (`type: xlerobot_client` in robot JSON)."""

    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            "forward": "i",
            "backward": "k",
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            "speed_up": "n",
            "speed_down": "m",
            "quit": "b",
        }
    )
    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)
    lift_axis: LiftAxisConfig = field(default_factory=LiftAxisConfig)
    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
