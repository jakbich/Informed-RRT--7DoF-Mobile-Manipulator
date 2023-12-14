import gymnasium as gym
import numpy as np
import pybullet as p
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from korneel_control import ArmControl

# Import or define ArmControl and Kinematics as needed

def run_panda(n_steps=1000, render=False, goal=True, obstacles=False):
    robots = [
        GenericUrdfReacher(urdf="panda_with_gripper.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots,
        render=render,
        observation_checking=False,
    )
    action = np.zeros(env.n())  # No action to keep the arm still
    ob = env.reset()

    ob, *_ = env.step(action)
    current_joint_angles = ob['robot_0']['joint_state']['position'][:7]

    # Add axes at the origin (you can change the position as needed)
    origin = [0, 0, 0]
    axis_length = 10.0  # Length of each axis
    p.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0], 2.0)  # X-axis in red
    p.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0], 2.0)  # Y-axis in green
    p.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1], 2.0)  # Z-axis in blue

    # Add a visual marker at the target position
    # target_position = np.array([0.550655275233520, -1.68862385257615E-17, 0.65])  # Adjust as needed
    target_position = np.array([0.31940916,  -0.21587151,  1.09758974 ])
    # target_position = np.array([0.58193844, 0.5       ,0.654902])
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.07, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=target_position)

    history = []
    xyz_history = []  # To store XYZ values
    error_hisotry = []
    arm_control = ArmControl()

    for i in range(n_steps):
        # Get the current joint angles
        current_joint_angles = np.array(ob['robot_0']['joint_state']['position'][:7])

        # Calculate joint space control action using ArmControl
        joint_space_action, error = arm_control.control_action(current_joint_angles, target_position)

        # Construct the full control action array
        control_action = np.zeros(env.n())
        control_action[:7] = joint_space_action # Assuming the first 7 elements are for joint control
        # print(control_action[:7])

        # Step the environment with the computed control action
        ob, *_ = env.step(control_action)

        # Compute the current XYZ position of the end effector
        # xyz, _ = Kinematics().FK(current_joint_angles)
        # p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=xyz)
        # xyz_history.append(xyz)  # Save XYZ values
        # error_hisotry.append(np.linalg.norm(error))
        history.append(ob)

    env.close()
    
    # Plot the XYZ values in 3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xyz_history = np.array(xyz_history)
    # print('xyz_history', xyz_history[500])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz_history[:, 0], xyz_history[:, 1], xyz_history[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Endpoint Position (XYZ) Over Time')
    plt.show()
    
    # Plot the error over time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(error_hisotry)
    ax.set_xlabel('Time')
    ax.set_ylabel('Error')
    ax.set_title('Error Over Time')
    plt.show()

    return history

if __name__ == "__main__":
    run_panda(render=True)
