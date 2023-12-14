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
    print('current_joint_angles' , current_joint_angles)


    # Add axes at the origin (you can change the position as needed)
    origin = [0, 0, 0]
    axis_length = 10.0  # Length of each axis
    p.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0], 2.0)  # X-axis in red
    p.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0], 2.0)  # Y-axis in green
    p.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1], 2.0)  # Z-axis in blue

    # Add a visual marker at the target position
    target_position = np.array([0.31940916,  0.01587151,  1.09758974])  # Adjust as needed
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.07, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=target_position)

    history = []
    for i in range(n_steps):
        # Control the endpoint here using inverse kinematics or similar approach
        # Example: joint_action = ArmControl().task_space_to_joint_space(...)
        joint_action = ArmControl().task_space_to_joint_space(current_joint_angles, target_position)
        padded_joint_action = np.pad(joint_action, (2, 12 - len(joint_action) - 2), 'constant')

        ob, *_ = env.step(padded_joint_action)
        current_joint_angles = ob['robot_0']['joint_state']['position'][:7]
        history.append(ob)

    env.close()
    return history

if __name__ == "__main__":
    run_panda(render=True)
