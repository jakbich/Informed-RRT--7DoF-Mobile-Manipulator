import math
import numpy as np
from tqdm import tqdm
import warnings
import gymnasium as gym
import pybullet as p
import sys
import os
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests_jakob.create_environments import fill_env_with_obstacles, add_sphere


class RRTStar:

    def __init__(self, config_start = (0,0), config_goal = (0,0), step_len = 0.1,iter_max = 100, obstacles= []):
        self.config_start = config_start
        self.config_goal = config_goal
        self.step_len = step_len
        self.iter_max = iter_max
        self.obstacles = obstacles

        self.node_list = []
        self.path = []
        self.robot_radius = 1

    # def sample_new_configuration(self):
    #     for _ in range(100): # Try 100 times to sample a valid config.
    #         sampled_config = np.random.uniform([domain['xmin'], domain['ymin'], domain['zmin']],
    #                                             [domain['xmax'], domain['ymax'], domain['zmax']], size=3) * scale
    #         if not self.collision_manager.is_in_obstacle(sampled_config):
    #             return sampled_config


    def sample_new_node(self, range):
        random_node  = np.random.uniform([-range, -range], [range, range], size=2)
        
        if not self.check_collision(random_node):
            # Return (x,y,0) if valid point
            return np.append(random_node, 0)


    # Collision checking
    def check_collision(self, node):
        for wall in self.obstacles:
            for sphere in wall:
                sphere_radius = sphere[3]

                # Check in 2D (x,y) for collision
                if self.distance(node, sphere[:2]) <= self.robot_radius + sphere_radius:
                    return True  # Collision detected
        return False  # No collision

    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))




def run_albert(n_steps=1000, render=False, goal=True, obstacles=True):
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

    env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    
    
    
    # Filling with obstacles and creating the list with al spheres [x,y,z,radius]
    all_obstacles = np.array(fill_env_with_obstacles(env, 'easy',1))

    print('all obstavels shape\n')
    print((all_obstacles))

    history = []

    # Create instance of RRTStar
    rrt = RRTStar(obstacles=all_obstacles)

    # Sample random nodes
    random_nodes = [rrt.sample_new_node(5) for _ in range(1000)]
    print(random_nodes)

    # Define point appearance
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
    
    
    # Plot random nodes
    for node in tqdm(random_nodes):
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=node)


        # print(ob)


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

