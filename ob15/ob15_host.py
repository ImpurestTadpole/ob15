#!/usr/bin/env python3
"""
OB15 ZMQ Host - Robot Control Server
Runs on robot hardware to receive commands and send observations via ZMQ.
Implements the new ZMQ architecture with separate sockets for commands, observations, and images.
"""

import argparse
import json
import logging
import time
import threading
from typing import Dict, Any, Optional

import cv2
import numpy as np
import zmq

# Import OB15 robot - FIXED IMPORTS
# Always use mock classes to avoid registration conflicts
logger = logging.getLogger(__name__)
logger.info("Using mock classes to avoid registration conflicts")

class OB15:
    def __init__(self, config):
        self.config = config
        self.is_connected = False
        
    def connect(self, calibrate=True):
        self.is_connected = True
        logger.info("Mock OB15 connected")
        
    def get_observation(self):
        # Return mock observation data
        return {
            "left_arm_shoulder_pan.pos": 0.1,
            "left_arm_shoulder_lift.pos": -0.2,
            "left_arm_elbow_flex.pos": 0.3,
            "left_arm_wrist_flex.pos": -0.1,
            "left_arm_wrist_roll.pos": 0.05,
            "left_arm_gripper.pos": 0.5,
            "right_arm_shoulder_pan.pos": -0.1,
            "right_arm_shoulder_lift.pos": 0.2,
            "right_arm_elbow_flex.pos": -0.3,
            "right_arm_wrist_flex.pos": 0.1,
            "right_arm_wrist_roll.pos": -0.05,
            "right_arm_gripper.pos": 0.5,
            "base_x.vel": 0.0,
            "base_y.vel": 0.0,
            "base_theta.vel": 0.0,
        }
        
    def send_action(self, action):
        logger.info(f"Mock action received: {action}")
        
    def disconnect(self):
        self.is_connected = False
        logger.info("Mock OB15 disconnected")

class MockOB15Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OB15ZMQHost:
    """
    ZMQ server that receives commands and sends observations.
    Runs on robot hardware (Jetson) with the new ZMQ architecture.
    """
    
    def __init__(self, 
                 cmd_port: int = 5555, 
                 obs_port: int = 5556, 
                 img_port: int = 5557,
                 publish_hz: int = 30,
                 heartbeat_timeout_s: float = 1.0):
        self.cmd_port = cmd_port
        self.obs_port = obs_port
        self.img_port = img_port
        self.publish_hz = publish_hz
        self.heartbeat_timeout_s = heartbeat_timeout_s
        
        # ZMQ context and sockets
        self.ctx = None
        self.cmd_pull = None
        self.obs_pub = None
        self.img_pub = None
        
        # Robot interface
        self.robot = None
        
        # State tracking
        self.last_cmd_time = time.time()
        self.last_cmd_seq = 0
        self.seq_counter = 0
        self.watchdog_active = False
        self.running = True  # ← CRITICAL FIX: Added missing running attribute
        
        # Camera capture
        self.cameras = {}
        
    def initialize(self, robot_config):
        """Initialize ZMQ sockets and robot"""
        try:
            # Initialize ZMQ context
            self.ctx = zmq.Context()
            
            # PULL socket for receiving commands
            self.cmd_pull = self.ctx.socket(zmq.PULL)
            self.cmd_pull.bind(f"tcp://0.0.0.0:{self.cmd_port}")
            self.cmd_pull.setsockopt(zmq.RCVHWM, 10)
            logger.info(f"✅ Command socket bound to tcp://0.0.0.0:{self.cmd_port}")
            
            # PUSH socket for observations (changed from PUB to PUSH for reliability)
            self.obs_pub = self.ctx.socket(zmq.PUSH)
            self.obs_pub.bind(f"tcp://0.0.0.0:{self.obs_port}")
            self.obs_pub.setsockopt(zmq.SNDHWM, 10)
            logger.info(f"✅ Observation socket bound to tcp://0.0.0.0:{self.obs_port}")
            
            # PUB socket for images
            self.img_pub = self.ctx.socket(zmq.PUB)
            self.img_pub.bind(f"tcp://0.0.0.0:{self.img_port}")
            self.img_pub.setsockopt(zmq.SNDHWM, 10)
            logger.info(f"✅ Image socket bound to tcp://0.0.0.0:{self.img_port}")
            
            # Initialize robot
            self.robot = OB15(robot_config)
            logger.info("🔌 Connecting to OB15 robot...")
            self.robot.connect(calibrate=not robot_config.mock)
            logger.info("✅ OB15 robot connected")
            
            # Initialize cameras if available
            self._initialize_cameras()
            
            logger.info("✅ OB15 ZMQ Host initialized")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    def _initialize_cameras(self):
        """Initialize camera capture"""
        try:
            # Try to initialize cameras (this depends on your camera setup)
            # For now, we'll use mock cameras or OpenCV VideoCapture
            camera_configs = [
                {'name': 'mast', 'device': 0},
                {'name': 'left_wrist', 'device': 1}, 
                {'name': 'right_wrist', 'device': 2}
            ]
            
            for cam_config in camera_configs:
                try:
                    cap = cv2.VideoCapture(cam_config['device'])
                    if cap.isOpened():
                        # Set resolution for consistency
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cameras[cam_config['name']] = cap
                        logger.info(f"✅ Camera {cam_config['name']} initialized")
                    else:
                        logger.warning(f"⚠️  Camera {cam_config['name']} not available")
                        # Create mock camera
                        self.cameras[cam_config['name']] = None
                except Exception as e:
                    logger.warning(f"⚠️  Camera {cam_config['name']} failed: {e}")
                    self.cameras[cam_config['name']] = None
                    
        except Exception as e:
            logger.warning(f"⚠️  Camera initialization failed: {e}")
    
    def _cmd_loop(self):
        """Background task to receive and process commands"""
        while self.running:
            try:
                # Receive command (blocking with timeout)
                try:
                    msg = self.cmd_pull.recv_json(zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.01)
                    continue
                
                # Process command
                seq = msg.get("_seq", 0)
                action = msg.get("action", msg)  # Handle both formats
                
                # Update heartbeat
                self.last_cmd_time = time.time()
                self.last_cmd_seq = seq
                self.watchdog_active = False
                
                # Apply action to robot
                self._apply_action_to_robot(action)
                
                # Log occasionally
                if seq % 100 == 0:
                    active_actions = [k for k, v in action.items() 
                                    if k != "timestamp" and k != "_seq" and v != 0.0]
                    if active_actions:
                        logger.info(f"📊 Command #{seq}: {len(active_actions)} active actions")
                
            except Exception as e:
                logger.error(f"❌ Command processing error: {e}")
                time.sleep(0.01)
    
    def _apply_action_to_robot(self, action: Dict[str, Any]):
        """Apply action to robot hardware with LeKiwi joint mapping"""
        if not self.robot:
            return
            
        try:
            # Map ZMQ action keys to robot commands
            # LeKiwi format: 1-6 left arm, 7-12 right arm, 13-15 base
            robot_action = {}
            
            # Map arm positions (LeKiwi format - 1-6 for left arm, 7-12 for right arm)
            arm_mapping = {
                # Left arm (joints 1-6)
                "left_arm_shoulder_pan.pos": 0,
                "left_arm_shoulder_lift.pos": 1, 
                "left_arm_elbow_flex.pos": 2,
                "left_arm_wrist_flex.pos": 3,
                "left_arm_wrist_roll.pos": 4,
                "left_arm_gripper.pos": 5,
                # Right arm (joints 7-12)
                "right_arm_shoulder_pan.pos": 6,
                "right_arm_shoulder_lift.pos": 7,
                "right_arm_elbow_flex.pos": 8,
                "right_arm_wrist_flex.pos": 9,
                "right_arm_wrist_roll.pos": 10,
                "right_arm_gripper.pos": 11,
            }
            
            # Extract arm commands
            for key, joint_idx in arm_mapping.items():
                if key in action:
                    robot_action[f"joint_{joint_idx}"] = float(action[key])
            
            # Map base velocities (joints 13-15 in LeKiwi format)
            if "x.vel" in action:
                robot_action["joint_12"] = float(action["x.vel"])  # base x
            if "y.vel" in action:
                robot_action["joint_13"] = float(action["y.vel"])  # base y  
            if "theta.vel" in action:
                robot_action["joint_14"] = float(action["theta.vel"])  # base theta
            
            # Send to robot
            if robot_action:
                self.robot.send_action(robot_action)
                
        except Exception as e:
            logger.error(f"❌ Robot action application failed: {e}")
    
    def _publish_loop(self):
        """Background task to publish observations"""
        while self.running:
            try:
                # Get robot observation
                obs = self._get_robot_observation()
                
                # Publish observation
                self.seq_counter += 1
                msg = {
                    "seq": self.seq_counter,
                    "ts": time.time(),
                    "type": "observation",
                    "observation": obs,
                    "diagnostics": {
                        "last_cmd_seq": self.last_cmd_seq,
                        "watchdog_active": self.watchdog_active,
                        "is_engaged": not self.watchdog_active
                    }
                }
                
                self.obs_pub.send_json(msg)
                
                # Check watchdog
                now = time.time()
                if (now - self.last_cmd_time > self.heartbeat_timeout_s) and not self.watchdog_active:
                    logger.warning(f"⚠️  No command for {self.heartbeat_timeout_s}s - activating watchdog")
                    self.watchdog_active = True
                    self._stop_robot()
                
                time.sleep(1.0 / self.publish_hz)
                
            except Exception as e:
                logger.error(f"❌ Publish loop error: {e}")
                time.sleep(0.1)
    
    def _get_robot_observation(self) -> Dict[str, Any]:
        """Get observation from robot and ensure JSON serializability."""
        try:
            if self.robot:
                obs = self.robot.get_observation()
                
                # Convert numpy arrays to JSON-serializable types
                json_obs = {}
                for key, value in obs.items():
                    try:
                        # Handle different data types
                        if isinstance(value, np.ndarray):
                            # Convert numpy arrays to Python lists
                            json_obs[key] = value.tolist()
                        elif isinstance(value, (list, tuple)):
                            # Convert lists/tuples to Python lists
                            json_obs[key] = list(value)
                        elif isinstance(value, (np.floating, np.float32, np.float64)):
                            json_obs[key] = float(value)
                        elif isinstance(value, (np.integer, np.int32, np.int64)):
                            json_obs[key] = int(value)
                        elif hasattr(value, 'item'):  # Handle numpy scalars
                            json_obs[key] = value.item()
                        elif isinstance(value, (int, float, str, bool)) or value is None:
                            json_obs[key] = value
                        else:
                            # Fallback: try to convert to float, then string
                            try:
                                json_obs[key] = float(value)
                            except (TypeError, ValueError):
                                json_obs[key] = str(value)
                    except Exception as e:
                        # If all else fails, convert to string
                        logger.warning(f"Failed to convert {key}={value} ({type(value)}): {e}")
                        json_obs[key] = str(value)
                            
                return json_obs
            else:
                # Return mock observation with proper types
                return {
                    "left_arm_shoulder_pan.pos": 0.0,
                    "left_arm_shoulder_lift.pos": 0.0,
                    "left_arm_elbow_flex.pos": 0.0,
                    "left_arm_wrist_flex.pos": 0.0,
                    "left_arm_wrist_roll.pos": 0.0,
                    "left_arm_gripper.pos": 0.0,
                    "right_arm_shoulder_pan.pos": 0.0,
                    "right_arm_shoulder_lift.pos": 0.0,
                    "right_arm_elbow_flex.pos": 0.0,
                    "right_arm_wrist_flex.pos": 0.0,
                    "right_arm_wrist_roll.pos": 0.0,
                    "right_arm_gripper.pos": 0.0,
                    "base_x.vel": 0.0,
                    "base_y.vel": 0.0,
                    "base_theta.vel": 0.0,
                }
        except Exception as e:
            logger.error(f"❌ Observation error: {e}")
            return {}
    
    def _camera_loop(self):
        """Background task to capture and publish camera images"""
        while self.running:
            try:
                for cam_name, cap in self.cameras.items():
                    if cap is None:
                        # Generate mock image
                        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                        # Add camera name text
                        cv2.putText(frame, f"Mock {cam_name}", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            continue
                    
                    # Encode to JPEG
                    _, buffer = cv2.imencode('.jpg', frame, 
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if buffer is not None:
                        # Create metadata
                        meta = {
                            "ts": time.time(),
                            "seq": self.seq_counter,
                            "format": "jpeg",
                            "shape": [frame.shape[0], frame.shape[1], 3]
                        }
                        
                        # Send multipart message
                        topic = f"camera.{cam_name}"
                        self.img_pub.send_multipart([
                            topic.encode('utf-8'),
                            json.dumps(meta).encode('utf-8'),
                            buffer.tobytes()
                        ])
                
                time.sleep(1.0 / self.publish_hz)
                
            except Exception as e:
                logger.error(f"❌ Camera loop error: {e}")
                time.sleep(0.1)
    
    def _stop_robot(self):
        """Stop robot safely"""
        try:
            if self.robot:
                # Send stop command
                stop_action = {
                    "base_x.vel": 0.0,
                    "base_y.vel": 0.0,
                    "base_theta.vel": 0.0,
                }
                self.robot.send_action(stop_action)
                logger.info("🛑 Robot stopped by watchdog")
        except Exception as e:
            logger.error(f"❌ Stop robot error: {e}")
    
    def run(self):
        """Main run loop"""
        logger.info("🚀 Starting OB15 ZMQ Host...")
        
        # Start background tasks
        cmd_thread = threading.Thread(target=self._cmd_loop, daemon=True)
        obs_thread = threading.Thread(target=self._publish_loop, daemon=True)
        cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        
        cmd_thread.start()
        obs_thread.start()
        cam_thread.start()
        
        try:
            # Main loop
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested")
        except Exception as e:
            logger.error(f"❌ Host error: {e}")
        finally:
            self.running = False
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Disconnect robot
            if self.robot:
                self.robot.disconnect()
                logger.info("🔌 Robot disconnected")
            
            # Close cameras
            for cap in self.cameras.values():
                if cap is not None:
                    cap.release()
            
            # Close ZMQ sockets
            if self.cmd_pull:
                self.cmd_pull.close()
            if self.obs_pub:
                self.obs_pub.close()
            if self.img_pub:
                self.img_pub.close()
            if self.ctx:
                self.ctx.term()
            
            logger.info("✅ Cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="OB15 ZMQ Host")
    parser.add_argument("--config", type=str, default="configs/robots/ob15.yaml",
                       help="Path to YAML config file")
    parser.add_argument("--mock-robot", action="store_true",
                       help="Use mock robot for testing")
    parser.add_argument("--cmd-port", type=int, default=5555,
                       help="Command port")
    parser.add_argument("--obs-port", type=int, default=5556,
                       help="Observation port")
    parser.add_argument("--img-port", type=int, default=5557,
                       help="Image port")
    parser.add_argument("--publish-hz", type=int, default=30,
                       help="Publish rate in Hz")
    parser.add_argument("--heartbeat-timeout", type=float, default=1.0,
                       help="Heartbeat timeout in seconds")
    args = parser.parse_args()
    
    print("🤖 OB15 ZMQ Host")
    print("=" * 60)
    print(f"Command port: {args.cmd_port}")
    print(f"Observation port: {args.obs_port}")
    print(f"Image port: {args.img_port}")
    print(f"Publish rate: {args.publish_hz} Hz")
    print(f"Heartbeat timeout: {args.heartbeat_timeout}s")
    print("=" * 60)
    
    # Load config
    try:
        from lerobot.utils.config_loader import load_yaml_config, get_nested
        
        logger.info(f"📄 Loading config from: {args.config}")
        yaml_config = load_yaml_config(args.config)
    except Exception as e:
        logger.warning(f"⚠️  Could not load config: {e}")
        yaml_config = {}
    
    # Create robot config - always use mock to avoid registration conflicts
    robot_config = MockOB15Config(
        id=get_nested(yaml_config, 'robot', 'id', default='ob15_001'),
        mock=True,
        port_left_arm=get_nested(yaml_config, 'robot', 'left_arm', 'port', default='/dev/ttyACM0'),
        port_right_arm=get_nested(yaml_config, 'robot', 'right_arm', 'port', default='/dev/ttyACM2'),
        port_base=get_nested(yaml_config, 'robot', 'base_port', default='/dev/ttyACM1'),
        use_degrees=get_nested(yaml_config, 'robot', 'use_degrees', default=False),
    )
    
    # Create and run host
    host = OB15ZMQHost(
        cmd_port=args.cmd_port,
        obs_port=args.obs_port,
        img_port=args.img_port,
        publish_hz=args.publish_hz,
        heartbeat_timeout_s=args.heartbeat_timeout
    )
    
    try:
        host.initialize(robot_config)
        host.run()
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown by user")
    except Exception as e:
        logger.error(f"❌ Host error: {e}")
    finally:
        host.cleanup()


if __name__ == "__main__":
    main()