import warnings
import gymnasium as gym
import numpy as np
import pybullet as p
import sys
import os
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from ab_control import ArmControl
from ab_kinematics import Kinematics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_path.RRT_global import RRTStar
from mobile_base.pid_control import PIDBase, path_smoother, interpolate_path





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
    target_position = (0.6, 0.3, 1)

    target_position_homogeneous = np.append(target_position, 1) 

    # Add target position as a red sphere
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.07, rgbaColor=[1, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=np.array(target_position))

    # -------------------- SHAPE VISUALIZATION --------------------
    # Add axes at the origin (you can change the position as needed)
    origin = [0, 0, 0]
    axis_length = 10.0  # Length of each axis
    p.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0], 2.0)  # X-axis in red
    p.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0], 2.0)  # Y-axis in green
    p.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1], 2.0)  # Z-axis in blue


    # Initial action to get initial observation
    action = np.zeros(env.n())
    for stp in range(10):
        ob, *_ = env.step(action)

        current_joint_angles = np.array(ob['robot_0']['joint_state']['position'][3:10])
        current_base_orientation = np.array(ob['robot_0']['joint_state']['position'][2])
        current_base_position = np.array(ob['robot_0']['joint_state']['position'][:2])

        arm_end_position,_,_ = kinematics.matrices(current_joint_angles)
        T_world_to_arm = kinematics.transform_world_to_arm(current_base_orientation, current_base_position)
        T_arm_to_world = np.linalg.inv(T_world_to_arm)

        current_end_position = np.dot(T_arm_to_world, np.append(arm_end_position, 1))[:3]  
 


    # ------------------- RRT -------------------
    rrt = RRTStar(config_start=current_end_position,
            obstacles=[], iter_max=500, 
            config_goal=target_position, step_len=0.01,
            sampling_range=1, rewire_radius=0.4, arm=True)
    
    rrt.planning()
    path_to_goal = np.array(rrt.find_path())
    print(path_to_goal)

    # total_cost_path = sum(rrt.cost.values())
    total_cost_path = rrt.calculate_path_cost(path_to_goal)
    
    
    if len(path_to_goal) > 3:
        interpolated_path = interpolate_path(path_to_goal, max_dist=4.0)
        path_to_goal_smooth = path_smoother(interpolated_path, total_cost_path=total_cost_path)
        rrt.visualize_path(path_to_goal[:,0,:])
        rrt.visualize_path(path_to_goal_smooth, spline=True)

        # Make path_to_goal sparse (every 10th point) while keeping the last point
        path_to_goal_sparse = path_to_goal_smooth[::20]
        path_to_goal_sparse[-1] = path_to_goal_smooth[-1]
        final_path = path_to_goal_sparse
        pid_controller = PIDBase(kp=[1, 0.75], ki=[0.0, 0.0], kd=[0.01, 0.01], dt=0.01)


    else:
        final_path = path_to_goal[:,0,:]
        print("Final path: ", path_to_goal)
        print(final_path)
        rrt.visualize_path(final_path)

    
    # -------------------- ALBERT CONTROL --------------------
    history = []
    for _ in range(n_steps):
        current_joint_angles = np.array(ob['robot_0']['joint_state']['position'][3:10])
        current_base_orientation = np.array(ob['robot_0']['joint_state']['position'][2])
        current_base_position = np.array(ob['robot_0']['joint_state']['position'][:2])

        arm_end_position,_,_ = kinematics.matrices(current_joint_angles)
        T_world_to_arm = kinematics.transform_world_to_arm(current_base_orientation, current_base_position)
        T_arm_to_world = np.linalg.inv(T_world_to_arm)

        current_end_position = np.dot(T_arm_to_world, np.append(arm_end_position, 1))[:3]  
 
        # Transform target position from world to arm coordinates
        T_world_to_arm = kinematics.transform_world_to_arm(current_base_orientation, current_base_position)
        arm_target_position_homogeneous = np.dot(T_world_to_arm, target_position_homogeneous)
        arm_target_position = arm_target_position_homogeneous[:3]

        joint_space_action = arm_control.control_action(current_joint_angles, arm_target_position).flatten()
        control_action = np.zeros(env.n())
        control_action[0] = 0.0
        control_action[1] = 0
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
