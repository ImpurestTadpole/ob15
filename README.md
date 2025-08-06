# LeMelon - Dual SO-ARM100 Mobile Manipulator

A comprehensive mono-repo for building, controlling, and training a dual-arm mobile manipulator using ROS2, VR teleoperation, and machine learning.

## 🚀 Quick Start

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/ImpurestTadpole/lemelon.git
cd lemelon

# Run the setup script
./scripts/setup_env.sh

# Build the ROS2 workspace
./scripts/build_ros.sh

# Start data collection
./scripts/collect_real.sh
```

## 📁 Project Structure

```
LeMelon/
├── external/                            # External repos cloned/submoduled
│   ├── lerobot/                         # HuggingFace's LeRobot
│   ├── leisaac/                         # Lightwheel's Isaac-based envs
│   ├── lekkiwi/                         # LeKiwi platform repo (UIUC)
│   ├── so_arm_100/                      # SO-100 URDF/control
│   ├── lightwheel_kitchen/             # Scene/env reference
│   └── collab-sim/                     # NVLabs dataset / RL tools
│
├── src/                                 # Main source code for LeMelon
│   ├── lemelon_description/            # URDFs, meshes, XACROs for full robot
│   ├── lemelon_bringup/                # ROS2 launch files to start system
│   ├── lemelon_teleop/                 # VR control nodes + Quest 3 integration
│   ├── lemelon_interfaces/             # Custom ROS2 msgs/srvs
│   └── lemelon_utils/                  # Shared utils: transforms, kinematics, logging
│
├── training/                            # Model training and sim scripts
│   ├── configs/                        # YAML configs for RL and BC
│   ├── ppo/                            # PPO-based training envs (IsaacLab + LeRobot)
│   ├── diffusion/                      # Diffusion policy training scripts
│   └── eval/                           # Evaluation pipelines and benchmarks
│
├── data/                                # Raw and processed data
│   ├── real/                           # Collected from teleop + cameras
│   ├── sim/                            # Sim-collected trajectories
│   └── processed/                      # HDF5 or Datasets for training
│
├── deploy/                              # Onboard tools and scripts
│   ├── jetson/                         # Jetson Nano setup for inference
│   └── rpi/                            # Raspberry Pi tools for remote ops
│
├── ros2_ws/                             # ROS2 workspace overlay
│   └── src/ → symlink to ../src/
│
├── scripts/                             # Top-level utilities and run scripts
│   ├── setup_env.sh                    # Setup conda/venv and install deps
│   ├── build_ros.sh                    # ROS2 build helper
│   ├── collect_real.sh                 # Real world data collection launch
│   ├── collect_sim.sh                  # Sim collection + teleop launch
│   └── train_model.sh                  # Unified training entrypoint
│
├── env/                                 # Conda or venv environment files
│   ├── environment.yml
│   └── requirements.txt
│
├── .gitmodules                          # Git submodule definitions
├── README.md
└── pyproject.toml (optional if using poetry)
```

## 🏗️ System Architecture

- **Hardware**: Dual 650mm SO-ARM100 arms on LeKiwi omnidirectional base
- **Sensors**: Three cameras (left/right wrist monocular, chest RGB-D)
- **Teleoperation**: Meta Quest 3 WebXR application for real-time control
- **Data Pipeline**: Real-world and simulated data collection for ML training
- **Compute**: Distributed (training on PC, inference on Jetson Nano/Raspberry Pi)

## 📋 Prerequisites

- Ubuntu 22.04 LTS
- ROS2 Humble
- Python 3.10+
- Meta Quest 3 (for teleoperation)
- NVIDIA GPU (for training)

## 🔧 Installation

See [SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for detailed installation instructions.

## 🎮 Usage

### Data Collection
```bash
# Start real-world data collection with Quest 3
./scripts/collect_real.sh

# Start simulation data collection
./scripts/collect_sim.sh
```

### Training
```bash
# Train a model on collected data
./scripts/train_model.sh --config training/configs/ppo_config.yaml
```

### Deployment
```bash
# Deploy to Jetson Nano
./deploy/jetson/deploy_model.sh --model path/to/trained/model
```

## 📚 Documentation

- [Setup Guide](docs/SETUP_GUIDE.md) - Complete installation and configuration
- [Hardware Integration](docs/HARDWARE.md) - Robot assembly and wiring
- [Teleoperation](docs/TELEOP.md) - VR control setup and usage
- [Training](docs/TRAINING.md) - ML model training and evaluation
- [Deployment](docs/DEPLOYMENT.md) - Edge deployment guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) - Robot learning framework
- [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) - Mobile base platform
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) - Robotic arm
- [IsaacLab](https://github.com/isaac-sim/IsaacLab) - Simulation environment 
