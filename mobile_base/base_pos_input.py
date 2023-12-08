import gymnasium as gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import numpy as np
import time

from urdfenvs.scene_examples.obstacles import *
from urdfenvs.scene_examples.goal import *
from urdfenvs.urdf_common.urdf_env import UrdfEnv

def run_point_robot(n_steps=10000, render=False, goal=True, obstacles=True):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Define a list of actions
    actions = [
        np.array([1, 1, 0.0]),
        np.array([0, -1, 0.0]),
        np.array([-1, 0, 0.0]),
        # Add more actions as needed
    ]
    pos0 = np.array([0., 0., 0.])
    vel0 = np.array([0., 0., 0.])
    ob = env.reset(pos=pos0, vel=vel0)
    print(f"Initial observation : {ob}")

    history = []
    env.reconfigure_camera(2.0, 0.0, -90.01, (0, 0, 0))

    current_action_index = 0
    action_switch_interval = 2  # Time in seconds to switch actions
    last_switch_time = time.time()  # Record the start time

    for _ in range(n_steps):
        current_time = time.time()
        if current_time - last_switch_time >= action_switch_interval:
            # Switch action every 2 seconds
            current_action_index = (current_action_index + 1) % len(actions)
            last_switch_time = current_time

        action = actions[current_action_index]
        ob, _, terminated, _, info = env.step(action)
        if terminated:
            print(info)
            break
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_point_robot(render=True)
