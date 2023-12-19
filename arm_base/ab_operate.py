import warnings
import gymnasium as gym
import numpy as np
import pybullet as p
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from ab_control import ArmControl
from ab_kinematics import Kinematics


def run_albert(n_steps=10000, render=False, goal=True, obstacles=True):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation = np.pi/2,       # in radians
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )

    action = np.zeros(env.n())  # No action to keep the arm still
    ob = env.reset()

    ob, *_ = env.step(action)
    current_joint_angles = ob['robot_0']['joint_state']['position'][3:10]

    arm_control = ArmControl()
    kinematics = Kinematics()



    # target_position_temp = np.array([5.80310599e-01, 6.08140775e-07, 6.89718851e-01])
    target_position = np.array([[0.5903106, 0.3, 1.02971885]])

    target_position_homogeneous = np.append(target_position, 1) 

    # Add target position as a red sphere
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.07, rgbaColor=[1, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=np.array([0.5903106, 0.3, 1.02971885]))




    # -------------------- SHAPE VISUALIZATION --------------------
    # Add axes at the origin (you can change the position as needed)
    origin = [0, 0, 0]
    axis_length = 10.0  # Length of each axis
    p.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0], 2.0)  # X-axis in red
    p.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0], 2.0)  # Y-axis in green
    p.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1], 2.0)  # Z-axis in blue



    # # Add arm reach
    # sphere_radius = 0.855
    # sphere_center = [0, 0, 0.335]
    # sphere_center = kinematics.base_to_arm(sphere_center)
    # sphere_visual_shape_id = p.createVisualShape(
    #     shapeType=p.GEOM_SPHERE, 
    #     radius=sphere_radius, 
    #     rgbaColor=[0.5, 0.5, 1.0, 0.3])
    # p.createMultiBody(
    #     baseMass=0, 
    #     baseVisualShapeIndex=sphere_visual_shape_id, 
    #     basePosition=sphere_center)


    # -------------------- ALBERT CONTROL --------------------
    history = []
    for _ in range(n_steps):
        current_joint_angles = np.array(ob['robot_0']['joint_state']['position'][3:10])
        current_base_orientation = np.array(ob['robot_0']['joint_state']['position'][2])
        current_base_position = np.array(ob['robot_0']['joint_state']['position'][:2])

        # Transform target position from world to arm coordinates
        print(current_base_position)
        T_world_to_arm = kinematics.transform_world_to_arm(current_base_orientation, current_base_position)
        arm_target_position_homogeneous = np.dot(T_world_to_arm, target_position_homogeneous)
        arm_target_position = arm_target_position_homogeneous[:3]

        joint_space_action = arm_control.control_action(current_joint_angles, arm_target_position).flatten()
        control_action = np.zeros(env.n())
        # control_action[0] = 0.1
        control_action[1] = -0.1
        control_action[2:9] = joint_space_action
        ob, *_ = env.step(control_action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
