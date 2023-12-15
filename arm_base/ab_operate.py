import gymnasium as gym
import numpy as np
import pybullet as p
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from ab_control import ArmControl


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

    # target_position = np.array([-0.31940916,  0.21587151,  1.09758974 ])
    target_position = np.array([0.78193844, 0.3       ,0.454902])
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.07, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=target_position)

    # Parameters for the sphere
    sphere_radius = 0.855
    sphere_center = [0, 0, 0.335]

    # Add a visual representation of the robot's horizontal reach as a semi-transparent sphere
    sphere_visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, 
        radius=sphere_radius, 
        rgbaColor=[0.5, 0.5, 1.0, 0.3]
    )
    p.createMultiBody(
        baseMass=0, 
        baseVisualShapeIndex=sphere_visual_shape_id, 
        basePosition=sphere_center
    )

    history = []
    arm_control = ArmControl()

    for i in range(n_steps):
        current_joint_angles = np.array(ob['robot_0']['joint_state']['position'][:7])
        joint_space_action = arm_control.control_action(current_joint_angles, target_position).flatten()
        control_action = np.zeros(env.n())
        control_action[:7] = joint_space_action
        ob, *_ = env.step(control_action)
        history.append(ob)
    env.close()
    return history

if __name__ == "__main__":
    run_panda(render=True)
