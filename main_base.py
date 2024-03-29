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
 
from environments.create_environments import fill_env_with_obstacles

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_path.RRT_global import RRTStar, InformedRRTStar
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
    all_obstacles = np.array(fill_env_with_obstacles(env, 'easy',1))

    # ####RRT#####

    history = []

    #goal_pos = (1.5,-2, 0) # for easy env
    goal_pos = (1.5,-4.5, 0) # for medium and video env
    #goal_pos = (5,1, 0) # for hard env

    # Plottin the goal
    visual_shape_goal = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_goal, basePosition=goal_pos)


    # Initial action to get initial observation
    action = np.zeros(env.n())
    for stp in range(10):
        ob, *_ = env.step(action)
        current_joint_angles = np.array(ob['robot_0']['joint_state']['position'][3:10])

    rrt_informed = InformedRRTStar(config_start=ob['robot_0']['joint_state']['position'][0:3],
                  obstacles=all_obstacles, iter_max=500, 
                  config_goal=goal_pos, step_len=1,
                  sampling_range=10, rewire_radius=2)
    rrt_informed.planning()
    
    path_to_goal = np.array(rrt_informed.find_path())
    

    # total_cost_path = sum(rrt_informed.cost.values())
    total_cost_path = rrt_informed.calculate_path_cost(path_to_goal)
    
    if len(path_to_goal) > 3:

        interpolated_path = interpolate_path(path_to_goal, max_dist=4.0)
        path_to_goal_smooth = path_smoother(interpolated_path, total_cost_path=total_cost_path)
        rrt_informed.visualize_path(path_to_goal[:,0,:], color = [1,0,0])
        rrt_informed.visualize_path(path_to_goal_smooth, spline=True)

        # Make path_to_goal sparse (every 10th point) while keeping the last point
        path_to_goal_sparse = path_to_goal_smooth[::20]
        
        path_to_goal_sparse[-1] = path_to_goal_smooth[-1]
        final_path = path_to_goal_sparse
        pid_controller = PIDBase(kp=[1, 0.75], ki=[0.0, 0.0], kd=[0.01, 0.01], dt=0.01)


    else:
        final_path = path_to_goal[:,0,:]
        rrt_informed.visualize_path(final_path)
        pid_controller = PIDBase(kp=[1, 2], ki=[0.0, 0.0], kd=[0.01, 0.01], dt=0.01)


    ###/RRT####


    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    prev_action = np.zeros(env.n())


    # plt.plot(rrt_informed.all_path_costs)
    # plt.xlabel('Number of Paths Found')
    # plt.ylabel('Path Cost')
    # plt.title('Path Costs Over Iterations')
    # plt.show()

    print ("Path cost is: ", total_cost_path)

    for step in range(n_steps):

        ob, *_ = env.step(action)
        current_obs  = ob['robot_0']['joint_state']['position']

        # Track reached positions
        if pid_controller.count_reached != len(final_path)-1:
            action = pid_controller.update(current_pos=current_obs[0:3], goal_pos=final_path[pid_controller.count_reached], action=prev_action)
            prev_action = action    
        
        # right before last step
        elif pid_controller.count_reached == len(final_path)-1:
            action = pid_controller.update(current_pos=current_obs[0:3], goal_pos=final_path[-1], action=prev_action, last_step = True)


        # action for last step
        else:
            if not final_reach_sent:
                print("All positions reached!")
                final_reach_sent = True
            action = np.zeros(env.n())

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
