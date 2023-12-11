import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from armcontrol import ArmControl


def run_albert(n_steps=1000, render=False, goal=True, obstacles=True, albert_radius=0.3):
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
    action[0] = 0.0
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    ob, *_ = env.step(action)
    # robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
    # current_joint_angles = robot_config[0][3:10]
    current_joint_angles = ob['robot_0']['joint_state']['position'][3:10]


    target_position = np.array([1.02, 5.02, 0.02])


    history = []
    for _ in range(n_steps):
        joint_action = ArmControl().task_space_to_joint_space(current_joint_angles, target_position)
        padded_joint_action = np.pad(joint_action, (3, 12 - len(joint_action) - 3), 'constant')
        ob, *_ = env.step(padded_joint_action)
        current_joint_angles = ob['robot_0']['joint_state']['position'][3:10]
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)