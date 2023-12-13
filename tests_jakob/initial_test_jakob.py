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
import icecream as ic   
 
from create_environments import fill_env_with_obstacles

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_path.RRT_global import RRTStar
from mobile_base.pid_control import PIDBase


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
    
    
    # Set the target position
    #target_positions = np.array([[1,4,0], [4,1,0], [10,10, 0], [0,0,0]])

    # Add axes at the origin (you can change the position as needed)
    origin = [0, 0, 0]
    axis_length = 10.0 # Length of each axis
    p.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0], 2.0)  # X-axis in red
    p.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0], 2.0)  # Y-axis in green
    p.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1], 2.0)  # Z-axis in blue
    # Add a visual marker at the target position
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])

    #for target in target_positions:
    #    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=target)



    action = np.zeros(env.n())

    ob = env.reset(
        pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    

    # Filling with obstacles and creating the list with al spheres [x,y,z,radius]
    all_obstacles = np.array(fill_env_with_obstacles(env, 'easy',1))

    ####RRT#####

    history = []

    goal_pos = (1,-3,0)
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=goal_pos)
    
    rrt = RRTStar(obstacles=all_obstacles, iter_max=500, config_goal=goal_pos, step_len=0.8)
    rrt.planning()
    path_to_goal = np.array(rrt.find_path())
    rrt.visualize_path(path_to_goal)
    print("PATH SHAPE: ", path_to_goal.shape)
    print("Path Type", type(path_to_goal))
    ###/RRT####


    print(f"Initial observation : {ob}")
    linear_actions = []  # List to store action[0] values
    linear_errors = []
    final_reach_sent = False

    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    pid_controller = PIDBase(kp=[1, 1], ki=[0.0, 0.0], kd=[0.01, 0.01], dt=0.01)
    prev_action = np.zeros(env.n())




    for step in range(n_steps):

        ob, *_ = env.step(action)
        current_obs  = ob['robot_0']['joint_state']['position']

        # Track reached positions
        if pid_controller.count_reached != len(path_to_goal):
            action = pid_controller.update(current_pos=current_obs[0:3], goal_pos=path_to_goal[:,1,:][pid_controller.count_reached], action=prev_action)
            prev_action = action    
            linear_actions.append(action[0])
            linear_errors.append(pid_controller.error_linear)
              # Store the linear action

        else:
            if not final_reach_sent:
                print("All positions reached!")
                final_reach_sent = True
            action = np.zeros(env.n())

        if step % 100 == 0:
            print (f"current_obs: {current_obs[0:3]} ,\n action: {action[0:2]}")
            print(f"PID_ERROR_linear", pid_controller.error_linear)
            print(f"PID_ERROR_angular", pid_controller.error_angular)
            print (f"step: {step}")
            #print (f"lin_error {pid_controller.error_linear}, ang_error {pid_controller.error_angular}\n\n")

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
