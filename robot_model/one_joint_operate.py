import gymnasium as gym
import numpy as np
import pybullet as p
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from one_joint_control import ArmControl
from one_joint_kinematics import Kinematics as one_joint_kinematics

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
    target_position = np.array([-0.61336948,  0.03813614,  0.7767755 ])
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.07, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=target_position)

    history = []
    xyz_history = []  # To store XYZ values
    for i in range(n_steps):
        
        control_action = np.zeros(env.n())
        control_action[3] = ArmControl().control_action(current_joint_angles, target_position)
        print('control_action', control_action)        
        
        debug_action = np.zeros(env.n())
        debug_action[3] = 1

        ob, *_ = env.step(control_action)
        current_joint_angles = ob['robot_0']['joint_state']['position'][:7]
        xyz = one_joint_kinematics(current_joint_angles).forward_kinematics()
        xyz_history.append(xyz)  # Save XYZ values

        # Drawing the xyz pos in the simulation
        # p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=xyz)

        history.append(ob)

    env.close()
    
    # Plot the XYZ values in 3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xyz_history = np.array(xyz_history)
    print('xyz_history', xyz_history[500])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz_history[:, 0], xyz_history[:, 1], xyz_history[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Endpoint Position (XYZ) Over Time')
    plt.show()

    return history

if __name__ == "__main__":
    run_panda(render=True)
