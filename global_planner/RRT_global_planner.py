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

    def __init__(self, config_start = (0,0,0), config_goal = (0,0,0), step_len = 0.1,iter_max = 100, obstacles= []):
        self.config_start = config_start
        self.config_goal = config_goal
        self.step_len = step_len
        self.iter_max = iter_max
        self.obstacles = obstacles

        self.node_list = [np.array(config_start)]
        self.path = []
        self.robot_radius = 0.5
        self.goal_epsilon = 1.0
        self.parent ={}


        self.visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])


    def sample_new_node(self, sampling_range):
        
        for i in range(100): # Try 100 times to sample a valid config.
            random_node  = np.random.uniform([-sampling_range, -sampling_range,0], [sampling_range, sampling_range,0], size=3)
            if not self.check_collision(random_node):
                # Return (x,y,0) if valid point
                return random_node



    # Collision checking
    def check_collision(self, node):
        for wall in self.obstacles:
            for sphere in wall:
                sphere_radius = sphere[3]

                # Check in 2D (x,y) for collision
                if self.distance(node, sphere[:3]) <= self.robot_radius + sphere_radius:
                    return True  # Collision detected
        return False  # No collision
    
        
    def distance(self, point1, point2):
        print("\n Points: ", point1, point2)
        return np.linalg.norm(np.array(point1[:2]) - np.array(point2[:2]))
    
##############################################################################################################
    def planning(self, sample_range=5):
        while not self.is_goal_reached(self.node_list[-1]):
            new_node = self.sample_new_node(sample_range)
            if new_node is not None:
                nearest_node = self.find_nearest_node(new_node)
                if self.check_path(new_node, nearest_node):
                    self.add_node(new_node, nearest_node)

        self.add_node(self.config_goal, self.node_list[-1])

    def find_nearest_node(self, new_node):
        # Find the nearest node in the existing tree to the new node
        closest_dist = float('inf')
        nearest_node = None
        for node in self.node_list:
            dist = self.distance(node, new_node)
            if dist < closest_dist:
                closest_dist = dist
                nearest_node = node
        return nearest_node


    def check_path(self, new_node, nearest_node):
        # Number of subpoints to check along the path
        num_subpoints = 5

        # Calculate the vector from nearest_node to new_node
        vector = np.array(new_node) - np.array(nearest_node)

        # Check each subpoint for collision
        for i in range(1, num_subpoints + 1):
            # Calculate the subpoint's position
            subpoint = np.array(nearest_node) + vector * (i / num_subpoints)
            if self.check_collision(subpoint):
                return False  # Collision detected at subpoint
        return True  # No collision detected along the path
    


    def add_node(self, new_node, parent_node):
        # Add the new node to the tree and path
        self.node_list.append(new_node)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.visual_shape_id, basePosition=new_node)

        self.path.append((np.array(parent_node), new_node))
        p.addUserDebugLine(parent_node,new_node, [0.2, 0.2, 0.2], lineWidth=3)

        self.parent[tuple(new_node)] = tuple(parent_node)

    def is_goal_reached(self, node):
        # Check if the node is within a certain threshold of the goal
        return self.distance(node, self.config_goal) < self.goal_epsilon

    def find_path(self):
        path = []
        current_node = tuple(self.config_goal)
        while current_node != tuple(self.config_start):
            parent_node = self.parent[current_node]
            path.append((parent_node, current_node))
            current_node = parent_node
        path.reverse()  # Reverse the path to start from the beginning
        return path
    
    def visualize_path(self, path):
        for i in range(len(path) - 1):
            start_point, end_point = path[i], path[i + 1]
            # Unpack the start and end points of each line segment
            p.addUserDebugLine(start_point[1], end_point[1], [1, 0, 0], lineWidth=10)


def run_albert(n_steps=10000, render=False, goal=True, obstacles=True):
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
    
    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    
    # Filling with obstacles and creating the list with al spheres [x,y,z,radius]
    all_obstacles = np.array(fill_env_with_obstacles(env, 'easy',1))

    print('all obstavels shape\n')
    print((all_obstacles))

    history = []

    goal_pos = (3,3,0)
    # Create instance of RRTStar
    rrt = RRTStar(obstacles=all_obstacles, iter_max=500, config_goal=goal_pos)
    rrt.planning()
    path_to_goal = rrt.find_path()
    rrt.visualize_path(path_to_goal)

    print(f"\n\nPath: {rrt.path}")
    print(f"\n\nNodes: {rrt.node_list}")


    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=goal_pos)
    # # # Plot path
    # for node in tqdm(rrt.path):
    #     p.addUserDebugLine(node[0], node[1], [0, 0, 1], lineWidth=5.0)

    # # # Plot nodes
    # for node in tqdm(rrt.node_list):
    #     p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=node)


    # # Sample random nodes
    # random_nodes = [rrt.sample_new_node(5) for _ in range(1000)]
    # print(random_nodes)

    # # Define point appearance
    
    
    # # Plot random nodes
    # for node in tqdm(random_nodes):
    #     p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=node)


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

