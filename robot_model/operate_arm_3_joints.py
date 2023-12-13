import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
import pybullet as p
from kinematics_3_joints import Kinematics3joints
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run_albert(n_steps=100, render=False, goal=True, obstacles=True, albert_radius=0.3):
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
    ob = env.reset(pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    #MULTIPLE JOINTS
    # joint_limits = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973)]
    # coordinates = [[], [], []]  # Separate list for each joint
    # dot_colors = [
    # [1.0, 0.0, 0.0],  # Red for the first joint
    # [0.0, 1.0, 0.0],  # Green for the second joint
    # [0.0, 0.0, 1.0]   # Blue for the third joint
    # ]

    joint_limits = [(-2.8973, 2.8973)]  # Only the first joint's limits
    coordinates = [[]]  # List for the first joint only
    dot_colors = [[1.0, 0.0, 0.0]]  # Red for the first joint
    dot_radius = 0.02

    dot_radius = 0.02


    for joint_index, (lower_limit, upper_limit) in enumerate(joint_limits):
        for angle in np.linspace(lower_limit, upper_limit, n_steps):
            joint_angles = [0, 0, 0]
            joint_angles[joint_index] = angle

            # Simulate the action in the environment
            action = np.zeros(env.n())
            action[joint_index+2] = joint_angles[joint_index]
            ob, *_ = env.step(action)

            # Calculate the forward kinematics
            transformation_matrix = Kinematics3joints(joint_angles).forward_kinematics()
            x, y, z = transformation_matrix[0][3], transformation_matrix[1][3], transformation_matrix[2][3]
            print(x)
            xyz = [x, y, z]
            coordinates[joint_index].append(xyz)

            # Store the coordinates
            coordinates[joint_index].append([x, y, z])

            # Visualize the point in the simulation
            # rgba_color = dot_colors[joint_index] + [1]  # Adding alpha value for color
            # visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=dot_radius, rgbaColor=rgba_color)
            # p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=xyz)


    env.close()
    
   # Plotting the trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']  # Colors for each joint

    for i in range(len(coordinates)):
        joint_coordinates = np.array(coordinates[i])
        ax.plot(joint_coordinates[:, 0], joint_coordinates[:, 1], joint_coordinates[:, 2], color=colors[i], label=f'Joint {i+1}')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()

    return coordinates

if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        coordinates = run_albert(render=True)