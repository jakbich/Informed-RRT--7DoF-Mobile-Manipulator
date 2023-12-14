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
        self.goal_epsilon = 1
        self.parent ={tuple(self.config_start): None}

        # For RRT* algorithm
        self.cost = {tuple(self.config_start): 0}


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
        #print("\n Points: ", point1, point2)
        return np.linalg.norm(np.array(point1[:2]) - np.array(point2[:2]))
    
##############################################################################################################
    # def planning(self, sample_range=5):
    #     while not self.is_goal_reached(self.node_list[-1]):
    #         new_node = self.sample_new_node(sample_range)
    #         if new_node is not None:
    #             nearest_node = self.find_nearest_node(new_node)
    #             if self.check_path(new_node, nearest_node):
    #                 # Calculate intermediate node if the new node is too far
    #                 if self.distance(new_node, nearest_node) > self.step_len:
    #                     new_node = self.calculate_intermediate_node(nearest_node, new_node)
    #                 self.add_node(new_node, nearest_node)

    #                 # Check if the new node is close enough to the goal
    #                 if self.is_goal_reached(new_node):
    #                     self.add_node(np.array(self.config_goal), new_node)
    #                     break

    #     # Ensure the goal is added to the tree
    #     if not self.is_goal_reached(self.node_list[-1]):
    #         nearest_node_to_goal = self.find_nearest_node(self.config_goal)
    #         self.add_node(self.config_goal, nearest_node_to_goal)

    def add_node(self, new_node, parent_node):
        # Add the new node to the tree and path
        self.node_list.append(new_node)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.visual_shape_id, basePosition=new_node)

        self.path.append((np.array(parent_node), new_node))
        p.addUserDebugLine(parent_node,new_node, [0.2, 0.2, 0.2], lineWidth=3)
        p.addUserDebugLine(parent_node+[0,0,0.01],new_node+[0,0,0.01], [0.2, 0.2, 0.2], lineWidth=3)
        p.addUserDebugLine(parent_node+[0,0,0.02],new_node+[0,0,0.02], [0.2, 0.2, 0.2], lineWidth=3)

        self.parent[tuple(new_node)] = tuple(parent_node)
        self.cost[tuple(new_node)] = self.cost[tuple(parent_node)] + self.distance(parent_node, new_node)

    def planning(self, sample_range=5, search_radius=2):
        while not self.is_goal_reached(self.node_list[-1]):
            new_node = self.sample_new_node(sample_range)
            if new_node is not None:
                nearby_nodes = self.find_nearby_nodes(new_node, search_radius)
                best_parent, min_cost = self.choose_best_parent(new_node, nearby_nodes)
                if best_parent is not None:
                    self.add_node(new_node, best_parent)
                    self.rewire(new_node, nearby_nodes)

        # Last connection to goal node
        print("\n\n Came here manually add stuff")
        nearest_node_to_goal = self.find_nearest_node(self.config_goal)
        self.add_node(np.array(self.config_goal), nearest_node_to_goal)
        nearby_nodes = self.find_nearby_nodes(self.config_goal, search_radius)
        self.rewire(np.array(self.config_goal), nearby_nodes)

    # def planning(self, sample_range=5, search_radius=2):
    #     goal_connected = False

    #     while not goal_connected:
    #         new_node = self.sample_new_node(sample_range)
    #         if new_node is not None:
    #             nearby_nodes = self.find_nearby_nodes(new_node, search_radius)
    #             best_parent, min_cost = self.choose_best_parent(new_node, nearby_nodes)
    #             if best_parent is not None:
    #                 self.add_node(new_node, best_parent)
    #                 self.rewire(new_node, nearby_nodes)

    #                 if self.is_goal_reached(new_node):
    #                     goal_connected = True

    #     # Now explicitly add the goal node
    #     nearest_node_to_goal = self.find_nearest_node(self.config_goal)
    #     self.add_node(np.array(self.config_goal), nearest_node_to_goal)


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

    # Find nodes within a radius
    def find_nearby_nodes(self, new_node, radius):
        nearby_nodes = []
        for node in self.node_list:
            if self.distance(node, new_node) < radius:
                nearby_nodes.append(node)
        return nearby_nodes

    # Find the best node out of the nearby nodes and calculate cost
    def choose_best_parent(self, new_node, nearby_nodes):
        min_cost = float('inf')
        best_parent = None
        for node in nearby_nodes:
            if self.check_path(new_node, node):
                cost = self.cost[tuple(node)] + self.distance(node, new_node)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = node
        return best_parent, min_cost

    # Rewire the tree to minimize the costs
    def rewire(self, new_node, nearby_nodes):
        for node in nearby_nodes:
            if node is not self.parent[tuple(new_node)] and self.check_path(node, new_node):
                new_cost = self.cost[tuple(new_node)] + self.distance(new_node, node)
                if new_cost < self.cost[tuple(node)]:
                    self.parent[tuple(node)] = tuple(new_node)
                    self.cost[tuple(node)] = new_cost
                    # Update visualization if necessary


    def calculate_intermediate_node(self, nearest_node, new_node):
        direction = np.array(new_node) - np.array(nearest_node)
        norm_direction = direction / np.linalg.norm(direction)
        intermediate_node = np.array(nearest_node) + norm_direction * self.step_len
        return intermediate_node

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
    

    def is_goal_reached(self, node):
        # Check if the node is within a certain threshold of the goal
        return self.distance(node, self.config_goal) < self.goal_epsilon

    def find_path(self):
        path = []
        current_node = tuple(self.config_goal)
        while current_node != tuple(self.config_start):
            print('current node: ', current_node)
            parent_node = self.parent[current_node]
            print('parent node: ', parent_node)
            path.append((parent_node, current_node))
            current_node = parent_node
        path.reverse()  # Reverse the path to start from the beginning
        return path
    
    def visualize_path(self, path):
        for i in range(len(path)-1):
            start_point, end_point = path[i], path[i + 1]
            # Convert list to tuple for concatenation
            offset = (0, 0, 0.01)
            p.addUserDebugLine(np.array(start_point[0]) + offset, np.array(end_point[0]) + offset, [1, 0, 0], lineWidth=10)
            offset = (0, 0, 0.02)
            p.addUserDebugLine(np.array(start_point[0]) + offset, np.array(end_point[0]) + offset, [1, 0, 0], lineWidth=10)



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
        pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    
    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    
    # Filling with obstacles and creating the list with al spheres [x,y,z,radius]
    all_obstacles = np.array(fill_env_with_obstacles(env, 'easy',1))

    # print('all obstavels shape\n')
    # print((all_obstacles))

    history = []

    goal_pos = (3,-3,0)
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=goal_pos)
    
    # Create instance of RRTStar
    rrt = RRTStar(obstacles=all_obstacles, iter_max=500, config_goal=goal_pos, step_len=0.5)
    rrt.planning()
    path_to_goal = rrt.find_path()
    rrt.visualize_path(path_to_goal)

    # print(f"\n\nPath: {rrt.path}")
    # print(f"\n\nNodes: {rrt.node_list}")


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

