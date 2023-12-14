import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from arm_control import ArmControl
import pybullet as p
from robot_model.arm_kinematics import Kinematics
# from kinematics_3_joints import Kinematics3joints


def run_albert(n_steps=10000, render=False, goal=True, obstacles=True, albert_radius=0.3):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation = 0,
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    action = np.zeros(env.n())
    ob = env.reset

    ob, *_ = env.step(action)
    # robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
    # current_joint_angles = robot_config[0][3:10]
    current_joint_angles = ob['robot_0']['joint_state']['position'][3:10]


    target_position = np.array([1, -1.6886238525761472e-17, 1.25])

    # Add axes at the origin (you can change the position as needed)
    origin = [0, 0, 0]
    axis_length = 10.0 # Length of each axis
    p.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0], 2.0)  # X-axis in red
    p.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0], 2.0)  # Y-axis in green
    p.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1], 2.0)  # Z-axis in blue
    # Add a visual marker at the target position
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=[0.15, -0.55065528, 1.3872268])



    history = []
    for _ in range(n_steps):
        # joint_action = ArmControl().task_space_to_joint_space(current_joint_angles, target_position)
        # padded_joint_action = np.pad(joint_action, (2, 12 - len(joint_action) - 2), 'constant')
        # ob, *_ = env.step(padded_joint_action)
        # current_joint_angles = ob['robot_0']['joint_state']['position'][3:10]
        # history.append(ob)
        action = np.zeros(env.n())
        action[2] = 0.5
        ob, *_ = env.step(action)
        history.append(ob)
        xyz = Kinematics(current_joint_angles).forward_kinematics()
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=xyz)
        
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)