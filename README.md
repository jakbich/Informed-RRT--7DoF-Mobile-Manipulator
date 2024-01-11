# Informed RRT* algorithm on a mobile manipulator in simulation
Mobile manipulators find extensive applications, such as serving as picking agents in logistics centers for delivery services. This paper introduces the implementation of two random sampling algorithms—RRT* and informed RRT*—enabling the robot to manipulate its arm to retrieve items from a box and navigate a 2D environment to reach a predefined goal point, all while avoiding static obstacles. We conduct a comparative analysis, highlighting the superior performance of informed RRT* over RRT* in various environments and assessing the impact of specific hyperparameters.


[Link to video](https://www.youtube.com/watch?v=VViC1GyVnd4&feature=youtu.be)
## Repository Structure

```plaintext
PROJECT/
├── arm_model/1
│   ├── control_arm.py
│   ├── kinematics_arm.py
│   └── operate_arm.py
├── environments/
│   └── create_environments.py
├── full_robot_model/
│   ├── control_arm_on_albert.py
│   ├── kinematics_arm_on_albert.py
│   └── operate_arm_on_albert.py
├── global_path/
│   └── RRT_global.py
├── images/
├── mobile_base/
│   └── pid_control.py
├── main_arm.py
├── main_base.py
├── main.py
└── README.md

```

## Installation
Before proceeding, ensure that you have Conda installed on your system. If you do not have Conda installed, you can download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual).


1. Clone the repository:

   ```bash
   git clone git@github.com:Hobsyllvin/PDM_project.git
    ```

2. **Create the Environment**: Run the following command to create the python virtual environment:

   ```bash
   # Create a new virtual environment named `pdm_project_20`.
   python3 -m venv pdm_project_20

   # Activate the newly created virtual environment.
   source pdm_project_20/bin/activate
   cd PDM_project

   # Upgrade pip in the virtual environment. Essential for the urdfenvs package
   pip install --upgrade pip
   pip install gymnasium
   pip install pybullet
   pip install urdfenvs 
   pip install tqdm
    ```


## Usage
To run the autonomous robot simulation, execute the main scripts provided for different sections of the robot:

- `main_arm.py`: For testing the arm's path planning and object manipulation. It initializes the arm control, computes kinematics, and sets up the simulation environment with obstacles. Then it utilizes RRT* for path planning and PID control for arm movement.
- `main_base.py`: For testing the base's navigation and obstacle avoidance. It sets up a mobile base in the gymnasium environment, navigates through obstacles using Informed RRT*, and visualizes the path and actions over time.
- `main.py`: For full robot simulation including both arm and base operations. It combines functionalities of the arm and base to perform complete tasks, including reaching target positions and handling objects

For example, to run the full robot simulation:

```bash
python3 main.py
```

## Visualization

When running the scripts, they will render a simulation environment that allows you to track the robot's performance visually. Below are some snapshots of what you should expect to see:

### Arm Path Planning Visualization

<div style="text-align: center;">
  <img src="images/arm_first_target.png" alt="Arm Path Planning (First Target)" style="width: 100%; object-fit: contain;" />
  <p>Arm Path Planning (First Target)</p>
</div>
<div style="text-align: center;">
  <img src="images/arm_second_target.png" alt="Arm Path Planning (Second Target)" style="width: 100%; object-fit: contain;" />
  <p>Arm Path Planning (Second Target)</p>
</div>

### Base Navigation Visualization

<div style="text-align: center;">
  <img src="images/base_top.png" alt="Base Navigation (Topview)" style="width: 100%; object-fit: contain;" />
  <p>Base Navigation (Topview)</p>
</div>
<div style="text-align: center;">
  <img src="images/base_side.png" alt="Base Navigation (Sideview)" style="width: 100%; object-fit: contain;" />
  <p>Base Navigation (Sideview)</p>
</div>

## Authors
- Jakob Bichler
- Nicolas Wim Landuyt
- Christian Reinprecht 
- Korneel Somers
