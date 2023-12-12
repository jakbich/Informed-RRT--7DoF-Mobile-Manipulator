# import matplotlib.pyplot as plt
# import warnings
# import gymnasium as gym
# import numpy as np
# from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
# from urdfenvs.urdf_common.urdf_env import UrdfEnv
# from armcontrol import ArmControl
# import pybullet as p
# from kinematics import Kinematics
# from kinematics_3_joints import Kinematics3joints
# from mpl_toolkits.mplot3d import Axes3D


# def run_albert(n_steps=1000, render=False, goal=True, obstacles=True, albert_radius=0.3):
#     robots = [
#         GenericDiffDriveRobot(
#             urdf="albert.urdf",
#             mode="vel",
#             actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
#             castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
#             wheel_radius = 0.08,
#             wheel_distance = 0.494,
#             spawn_rotation = 0,
#             facing_direction = '-y',
#         ),
#     ]
#     env: UrdfEnv = gym.make(
#         "urdf-env-v0",
#         dt=0.01, robots=robots, render=render
#     )
#     action = np.zeros(env.n())
#     action[0] = 0.0
#     ob = env.reset(
#         pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     )
#     ob, *_ = env.step(action)
#     # robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
#     # current_joint_angles = robot_config[0][3:10]
#     current_joint_angles = ob['robot_0']['joint_state']['position'][3:10]

#     # Add axes at the origin (you can change the position as needed)
#     origin = [0, 0, 0]
#     axis_length = 10.0 # Length of each axis
#     p.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0], 2.0)  # X-axis in red
#     p.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0], 2.0)  # Y-axis in green
#     p.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1], 2.0)  # Z-axis in blue
#     # Add a visual marker at the target position
#     visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])

#     # Joint limits and plotting setup
#     joint_limits = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973)]
#     num_steps_per_joint = 10

#     x_positions = [[], [], []]  # x positions for each joint
#     y_positions = [[], [], []]  # y positions for each joint
#     z_positions = [[], [], []]  # z positions for each joint

#     # Define a small radius for the dots
#     dot_radius = 0.02

#     # Colors for each joint's dots
#     dot_colors = [
#         [1, 0, 0, 1],  # Red for the first joint
#         [0, 1, 0, 1],  # Green for the second joint
#         [0, 0, 1, 1]   # Blue for the third joint
#     ]

#     history = []
#     for _ in range(n_steps):
#         for joint_index, (lower_limit, upper_limit) in enumerate(joint_limits):
#             for step in np.linspace(lower_limit, upper_limit, num_steps_per_joint):
#                 # Set the joint angles
#                 joint_angles = [0, 0, 0]  # Reset other joints
#                 joint_angles[joint_index] = step

#                 # Update the robot's joint states in the simulation
#                 action = np.zeros(env.n())
#                 action[3:6] = joint_angles
#                 ob, *_ = env.step(action)

#                 # Calculate the end-effector position using forward kinematics
#                 xyz = Kinematics3joints(joint_angles).forward_kinematics()
#                 # Append positions to the respective lists
#                 x_positions[joint_index].append(xyz[0])
#                 y_positions[joint_index].append(xyz[1])
#                 z_positions[joint_index].append(xyz[2])

#                 # Optional: Add visualization in the simulation
#                 # p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=xyz)

#                 # Create a visual marker (dot) at the calculated position
#                 visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=dot_radius, rgbaColor=dot_colors[joint_index])
#                 p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=xyz)

#                 # Add the current position to the history for plotting
#                 history.append(ob)
        
#     env.close()

#     # Plotting
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Colors for each joint
#     colors = ['r', 'g', 'b']

#     for i in range(3):
#         ax.scatter(x_positions[i], y_positions[i], z_positions[i], c=colors[i], label=f'Joint {i+1}')

#     ax.set_xlabel('X Position')
#     ax.set_ylabel('Y Position')
#     ax.set_zlabel('Z Position')
#     plt.title('3D End-Effector Positions for Joint Movements')
#     plt.legend()
#     plt.show()
#     return history


# if __name__ == "__main__":
#     show_warnings = False
#     warning_flag = "default" if show_warnings else "ignore"
#     with warnings.catch_warnings():
#         warnings.filterwarnings(warning_flag)
#         run_albert(render=True)

import matplotlib.pyplot as plt
import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
import pybullet as p
from kinematics_3_joints import Kinematics3joints
from mpl_toolkits.mplot3d import Axes3D

def run_albert(n_steps=1000, render=False, goal=True, obstacles=True, albert_radius=0.3):
    # Environment setup
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_rotation=0,
            facing_direction='-y',
        ),
    ]
    env = gym.make("urdf-env-v0", dt=0.01, robots=robots, render=render)
    env.reset(pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # Joint limits and plotting setup
    joint_limits = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973)]
    num_steps_per_joint = 100
    x_positions = [[], [], []]
    y_positions = [[], [], []]
    z_positions = [[], [], []]

    # Define a small radius for the dots
    dot_radius = 0.02
    dot_colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]  # Red, Green, Blue

    for joint_index, (lower_limit, upper_limit) in enumerate(joint_limits):
        for step in np.linspace(lower_limit, upper_limit, num_steps_per_joint):
            joint_angles = [0, 0, 0]
            joint_angles[joint_index] = step

            action = np.zeros(env.n())
            action[3:6] = joint_angles
            env.step(action)

            xyz = Kinematics3joints(joint_angles).forward_kinematics()
            x_positions[joint_index].append(xyz[0])
            y_positions[joint_index].append(xyz[1])
            z_positions[joint_index].append(xyz[2])

            visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=dot_radius, rgbaColor=dot_colors[joint_index])
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=xyz)

    env.close()

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    for i in range(3):
        ax.scatter(x_positions[i], y_positions[i], z_positions[i], c=colors[i], label=f'Joint {i+1}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.title('3D End-Effector Positions for Joint Movements')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        run_albert(render=True)
