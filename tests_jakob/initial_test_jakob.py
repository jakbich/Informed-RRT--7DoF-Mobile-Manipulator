import gymnasium as gym
import warnings
import numpy as np
import argparse
import pybullet as p
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from create_environments import fill_env_with_obstacles


class PIDBase:
    def __init__(self, kp=[0,0], ki=[0,0], kd=[0,0], dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.last_error_linear = 0
        self.last_error_angular= 0
        
        self.integral_linear = 0
        self.integral_angular = 0


    def update(self, current_pos=[0,0,0], goal_pos=[10,10,10]):

        action = np.zeros(12)

        # Linear
        error_linear = np.sqrt((goal_pos[0] - current_pos[0]) ** 2 + (goal_pos[1] - current_pos[1]) ** 2)

        self.integral_linear += error_linear * self.dt
        derivative_linear = (error_linear - self.last_error_linear) / self.dt
        self.last_error_linear = error_linear

        action[0] = self.kp[0] * error_linear + self.ki[0] * self.integral_linear + self.kd[0] * derivative_linear

        # Angular
        error_angular = np.arctan2(goal_pos[1] - current_pos[1], goal_pos[0] - current_pos[0]) - current_pos[2]

        self.integral_angular += error_angular * self.dt
        derivative_angular = (error_angular - self.last_error_angular) / self.dt
        self.last_error_angular = error_angular

        action[1] = self.kp[1] * error_angular + self.ki[1] * self.integral_angular + self.kd[1] * derivative_angular

        return action

def run_albert(n_steps=10000, render=False, goal=True, obstacles=True, env_type='empty', sphere_density=1.0):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494, 
            spawn_rotation = 0,       # in degrees
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )


    # Fill the environment with obstacles, argument passed to determine which one (empty, easy, hard):
    #fill_env_with_obstacles(env, env_type, sphere_density)

    action = np.zeros(env.n())
    action[0] = 0.0
    action[1] = 0.5
    action[5] = -0.1
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )

    print(ob)


    
    print(f"Initial observation : {ob}")
    history = []

    pid_controller = PIDBase(kp=[0.5, 0.5], ki=[0.0, 0.0], kd=[0.0, 0.0], dt=0.01)


    for _ in range(n_steps):

        ob, *_ = env.step(action)
        current_obs  = ob['robot_0']['joint_state']['position']
        action = pid_controller.update(current_pos=current_obs[0:3], goal_pos=[1, 1, 0])

        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False

    parser = argparse.ArgumentParser(description='Fill environment with obstacles.')
    parser.add_argument('--env_type', type=str, help='Type of the environment to create', default='empty')
    parser.add_argument('--sphere_density', type=float, help='sphere_density to make spheres overlap', default=1.0)
    args = parser.parse_args()
    env_type = args.env_type
    sphere_density = args.sphere_density


    # pid_controller = PIDBase(kp=[0.5, 0.5], ki=[0.0, 0.0], kd=[0.0, 0.0], dt=0.01)
    # print(pid_controller.update(current_pos=[0,0,0], goal_pos=[-10,-10,-10])[0:3])


    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True, env_type=env_type, sphere_density=sphere_density)
