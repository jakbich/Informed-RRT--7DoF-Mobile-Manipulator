import gymnasium as gym
import numpy as np

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def run_albert(n_steps=1000, render=False, joint_speeds=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
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
    action[0] = 0.2
    action[1] = 0.0
    action[2] = joint_speeds[0]
    action[3] = joint_speeds[1]
    action[4] = joint_speeds[2]
    action[5] = joint_speeds[3]
    action[6] = joint_speeds[4]
    action[7] = joint_speeds[5]
    # action[8] = joint_speeds[6]
    # action[9] = joint_speeds[7]
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    print(f"Initial observation : {ob}")
    history = []
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    joint_speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    run_albert(render=True, joint_speeds=joint_speeds)