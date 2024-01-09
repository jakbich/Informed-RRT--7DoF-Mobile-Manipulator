# Autonomous Warehouse Robot
The robotic system is designed to operate within a warehouse environment. The robot, named Albert, is capable of autonomous navigation. The repository is structured to separate the functionalities of the arm control, base movement, environment setup, and the full robot model.

## Repository Structure

```plaintext
PROJECT/
├── arm_model/
│   ├── control_arm.py
│   ├── kinematics_arm.py
│   └── operate_arm.py
├── environments/
│   └── create_environments.py
├── full_robot_model/
│   ├── ab_control.py
│   ├── ab_kinematics.py
│   ├── ab_operate.py
│   └── main_ab.py
├── global_path/
│   └── RRT_global.py
├── mobile_base/
│   └── pid_control.py
├── path_planning/
│   └── RRT_global.py
├── main_arm.py
├── main_base.py
├── main.py
└── README.md
```

## Installation
Before running the scripts, ensure the following dependencies are installed:

- Python 3.x
- gymnasium
- numpy
- pybullet

## Usage
To run the autonomous robot simulation, execute the main scripts provided for different sections of the robot:

- `main_arm.py`: For testing the arm's path planning and object manipulation. It initializes the arm control, computes kinematics, and sets up the simulation environment with obstacles. Then it utilizes RRT* for path planning and PID control for arm movement.
- `main_base.py`: For testing the base's navigation and obstacle avoidance. It sets up a mobile base in the gymnasium environment, navigates through obstacles using Informed RRT*, and visualizes the path and actions over time.
- `main.py`: For full robot simulation including both arm and base operations. It combines functionalities of the arm and base to perform complete tasks, including reaching target positions and handling objects

For example, to run the full robot simulation:

```bash
python main.py
```

## Visualization

When running the scripts, they will render a simulation environment that allows you to track the robot's performance visually. Below are some snapshots of what you should expect to see:

### Arm Path Planning Visualization

<p float="left">
  <img src="images/arm_first_target.png" alt="Arm Path Planning (First Target)" width="45%" />
  <img src="images/arm_second_target.png" alt="Arm Path Planning (Second Target)" width="45%" /> 
</p>

### Base Navigation Visualization

<p float="left">
  <img src="images/base_top.png" alt="Base Navigation (Topview)" width="45%" />
  <img src="images/base_side.png" alt="Base Navigation (Sideview)" width="45%" />
</p>

## Authors
- Jakob Bichler
- Nicolas Wim Landuyt
- Christian Reinprecht 
- Korneel Somers
