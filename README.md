# LeMelon
Plan 
To create a custom 650mm tall duel so-100 with lekiwi base to be able to complete houseold and workshop tasks,
using a mono camera on each wrist and a chest camera to get workspace and navigation infomation.
Making this all in a mono repo to be able centerilize and transfer more easily. 

LeMelon
A Lekiwi base with 3 500mm 20x20 aluminum extrusions equal distance pointed up at a 10 degree angle with a platform made 
of a sinlge 20x40 300mm aluminum extrustion, with a duel so-100/101 arm configuration, these arms bases will be at 650mm 
from ground level, will be parrellel with one another. 
The system will be powered with a 10k mAh battery, and compute can ether be a raspberry pi5 or a jetson nano depending on compute needs. 

data will be coming;
6 joint data from left so100 arm, 
6 joint data from right so100 arm,
3 joint data from lekiwi base,
image data from left wrist cam,
image data from right wrist cam,
image /depth from chest/workspace camera,

Optional future
tactile feedback, using a anyskin on each so100 gripper

Tasks;
Houseold:
Laundy out of machine into basket;	~1:30min	~250eps
Laundy into machine out of basket;	~1:30min	~250eps
Spill clean up;		~0:30min	~100eps
Laundy clean up;	~2:00min	~300eps


Workshop:
Organize wrenches;	~1:00	~150eps




Data collection;
REAL WORLD
real world data will be collected from teleoperation using quest 3s to be able to control the robot
using the controllers; 
RIGHT Contoller
right controller while trigger is pressed will control the end effector position of right so100
right controller joystick will control lekiwi base directionand movement; 
left for turn left, right for turn right forward for forward and back for back. the A button will be to speed up and B button to slow down 
(potential use of the right controller side button to refresh right wrist camera stream to quest) 

LEFT Controller
left controller while trigger is pressed will control the end effector position of left so100
the X button will be to start recording an episode and Y to end the epsiode. 
(potential use of the left controller side button to refresh left wrist camera stream to quest) 

Quest streaming;
Image data while collecting episodes will be streamed to the quest 3s with the left and right wrist camera streams being on the top of the 
chest camera stream. This stream will be shown in the vision pro with passthrough vision for saftey so that the operator gets the idea of 
their enviroment.  

SIM data collection;
Sim teleoperation
Using isaac lab and sim and leisaac to teleoperate and collect data in sim that way, for potential use with subset training for ppo 

Sim training
also using isaac lab and sim but with ppo to set up enviroments from leisaac or other sources to create enviroments and to place the 
robot into then running ppo model to train for tasks,

Model training from data collection;
models such as smolvla, pi0, Grootn1, diffusion, ACT.
will be used to get the best results from the training data.

depolyment and compute;
inital training and testing should be done with desktop PC, current option for semi remote training is to use raspberry pi mounted on lekiwi 
to ssh into an collect and control remotley as lekiwi was initlly designed for. 

Later or prior once more data is collected, infrence could be run on a jetson nano, 
or do the same as pi for data collection and mount the jetson onto the lekiwi and ssh and teleoperate that way 


Repo's to pull from for resources 
https://github.com/huggingface/lerobot
https://github.com/SIGRobotics-UIUC/LeKiwi
https://github.com/LightwheelAI/leisaac/tree/main
https://github.com/NVlabs/collab-sim
https://github.com/isaac-sim/IsaacLab
https://github.com/TheRobotStudio/SO-ARM100
https://github.com/LightwheelAI/Lightwheel_Kitchen

https://wiki.ros.org/sw_urdf_exporter/Tutorials/Export%20an%20Assembly



Potential outline for directory

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
├── ros2_ws/                             # Optional: ROS2 workspace overlay
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
