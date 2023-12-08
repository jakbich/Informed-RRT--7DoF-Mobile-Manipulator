import warnings
import gymnasium as gym
import numpy as np
import pybullet as p
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.collision_obstacle import CollisionObstacle




"""
TEST TO ADD A BOX OBSTACLE ONLY BASED ON WHAT IS FOUND IN 
/home/jakob/anaconda3/envs/PDM/lib/python3.8/site-packages/mpscenes/obstacles
"""

from mpscenes.obstacles.box_obstacle import BoxObstacle
box_obs = []
box_obs.append(BoxObstacle(name='box_obstacle', content_dict={'type': 'box', 'movable': False, 'geometry': {'position': [0, -1.0, 0.0], 'length': 20.0, 'width': .2, 'height': 1}}, ))
box_obs.append(BoxObstacle(name='box_obstacle', content_dict={'type': 'box', 'movable': False, 'geometry': {'position': [0, 1.0, 0.0], 'length': 10.0, 'width': .2, 'height': 1}}))
"""
END OF TEST TO ADD A BOX OBSTACLE ONLY BASED ON WHAT IS FOUND IN 
/home/jakob/anaconda3/envs/PDM/lib/python3.8/site-packages/mpscenes/obstacles
"""


def add_sphere(env, pos, radius):
    sphere_obst_dict = {
        "type": "sphere",
        'movable': False,
        "geometry": {"position": pos, "radius": radius},
    }
    from mpscenes.obstacles.sphere_obstacle import SphereObstacle
    sphere_obst = SphereObstacle(name=f'obstacle_{pos[0]}_{pos[1]}_{pos[2]}', content_dict=sphere_obst_dict)
    env.add_obstacle(sphere_obst)


def add_wall(env, begin_pos, end_pos, horizontal=True, radius=0.5):
    if horizontal:
        assert begin_pos[1] == end_pos[1]
    else:
        assert begin_pos[0] == end_pos[0]

    if horizontal:
        n_spheres = abs(np.round((end_pos[0] - begin_pos[0]) / (radius * 2)).astype(int))
    else:
        n_spheres = abs(np.round((end_pos[1] - begin_pos[1]) / (radius * 2)).astype(int))

    # add obstacles
    for i in range(n_spheres):
        if horizontal:
            add_sphere(env, [begin_pos[0] + i, begin_pos[1], 0.0], radius)
        else:
            add_sphere(env, [begin_pos[0], begin_pos[1] + i, 0.0], radius)

    # add covering wall
    height = radius
    if horizontal:
        width = n_spheres - (radius * 2)
        length = radius * 2
        pos = [[(begin_pos[0] + end_pos[0]) / 2 - radius, (begin_pos[1] + end_pos[1]) / 2, 0]]
    else:
        width = radius * 2
        length = n_spheres - (radius * 2)
        pos = [[(begin_pos[0] + end_pos[0]) / 2, (begin_pos[1] + end_pos[1]) / 2 - radius, 0]]

    size = [width, length, height]






def run_albert(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation = 90,
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )

   # add_wall(env, [1, 2], [5, 2])

    for box_ob in box_obs:
        env.add_obstacle(box_ob)


    action = np.zeros(env.n())
    action[0] = 0.2
    action[1] = 0.0
    action[5] = -0.1
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
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
