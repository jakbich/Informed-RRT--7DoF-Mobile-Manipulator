import gymnasium as gym
import warnings
import numpy as np
import argparse
import math
import sys
import os
import pybullet as p
import matplotlib.pyplot as plt
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from ab_control import ArmControl
from ab_kinematics import Kinematics

arm_control = ArmControl()
kinematics = Kinematics()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests_jakob.create_environments import fill_env_with_obstacles
from global_path.RRT_global import RRTStar
from mobile_base.pid_control import PIDBase, path_smoother, interpolate_path


def run_albert(n_steps=100000, render=False, goal=True, obstacles=True, env_type='empty', sphere_density=1.0):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494, 
            spawn_rotation = math.pi/2,       # in radians
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    
    
    print("\nStarting simulation...\n--------------------------------\n\n")



    # Add axes at the origin (you can change the position as needed)
    origin = [0, 0, 0]
    axis_length = 10.0 # Length of each axis
    p.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0], 2.0)  # X-axis in red
    p.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0], 2.0)  # Y-axis in green
    p.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1], 2.0)  # Z-axis in blue


    action = np.zeros(env.n())

    ob = env.reset(
        pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

    # Filling with obstacles and creating the list with al spheres [x,y,z,radius]
    all_obstacles = np.array(fill_env_with_obstacles(env, 'medium',1))

    ####RRT#####

    history = []

    # Goal for medium env (1.5,-4.5,0)
    goal_pos = (2,0,0)

    target_position = np.array([2.2903106, 0.30, 1.02971885])
    target_position_homogeneous = np.append(target_position, 1) 

    draw_target_position = np.array(goal_pos) + target_position
    

    # PLottin the goal
    visual_shape_goal = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_goal, basePosition=goal_pos)
    
    # Add target position as a red sphere
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.07, rgbaColor=[1, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=np.array([2.2903106, 0.30, 1.02971885]))

    # Initial action to get initial observation
    action = np.zeros(env.n())
    for stp in range(10):
        ob, *_ = env.step(action)

    rrt = RRTStar(config_start=ob['robot_0']['joint_state']['position'][0:3],
                  obstacles=all_obstacles, iter_max=500, 
                  config_goal=goal_pos, step_len=1,
                  sampling_range=10, rewire_radius=1)
    rrt.planning()
    path_to_goal = np.array(rrt.find_path())
    

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
        pid_controller = PIDBase(kp=[1, 2], ki=[0.0, 0.0], kd=[0.01, 0.01], dt=0.01)


    ###/RRT####


    print(f"Initial observation : {ob}")
    linear_actions = []  # List to store action[0] values
    linear_errors = []
    final_reach_sent = False

    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    prev_action = np.zeros(env.n())





    for step in range(n_steps):

        ob, *_ = env.step(action)
        current_obs  = ob['robot_0']['joint_state']['position']
        print(f"Current observation : {current_obs}")

        # Track reached positions
        if pid_controller.count_reached != len(final_path)-1:
            action = pid_controller.update(current_pos=current_obs[0:3], goal_pos=final_path[pid_controller.count_reached], action=prev_action)
            prev_action = action    
        
        # right before last step
        elif pid_controller.count_reached == len(final_path)-1:
            action = pid_controller.update(current_pos=current_obs[0:3], goal_pos=final_path[-1], action=prev_action, last_step = True)
            current_position = current_obs[0:3]
            if np.linalg.norm(np.linalg.norm(current_position[:2] - goal_pos[:2])) < 0.3:
                print("Final position reached!")
                break
    
    
    for _ in range(n_steps):
        current_joint_angles = np.array(ob['robot_0']['joint_state']['position'][3:10])
        current_base_orientation = np.array(ob['robot_0']['joint_state']['position'][2])
        current_base_position = np.array(ob['robot_0']['joint_state']['position'][:2])

        # Transform target position from world to arm coordinates
        T_world_to_arm = kinematics.transform_world_to_arm(current_base_orientation, current_base_position)
        arm_target_position_homogeneous = np.dot(T_world_to_arm, target_position_homogeneous)
        arm_target_position = arm_target_position_homogeneous[:3]

        joint_space_action = arm_control.control_action(current_joint_angles, arm_target_position).flatten()
        control_action = np.zeros(env.n())
        control_action[2:9] = joint_space_action
        ob, *_ = env.step(control_action)
        history.append(ob)

    env.close()
    return history, linear_actions, linear_errors


if __name__ == "__main__":
    show_warnings = False

    parser = argparse.ArgumentParser(description='Fill environment with obstacles.')
    parser.add_argument('--env_type', type=str, help='Type of the environment to create', default='empty')
    parser.add_argument('--sphere_density', type=float, help='sphere_density to make spheres overlap', default=1.0)
    args = parser.parse_args()
    env_type = args.env_type
    sphere_density = args.sphere_density


    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        history, linear_actions, linear_errors = run_albert(render=True, env_type=env_type, sphere_density=sphere_density)

    plt.plot(linear_actions)
    plt.plot(linear_errors)
    plt.xlabel('Time Steps')
    plt.ylabel('Linear Action (action[0])')
    plt.title('Linear Action Over Time')
    plt.show()
