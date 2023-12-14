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

    def __init__(self, config_start = (0,0,0), config_goal = (0,0,0), step_len = 0.1,iter_max = 100, obstacles= [], sampling_range=5, rewire_radius= 2):
        self.config_start = config_start
        self.config_goal = config_goal
        self.step_len = step_len
        self.iter_max = iter_max
        self.obstacles = obstacles

        self.node_list = [np.array(config_start)]
        self.path = []

        # Sphere around robot for collision checking safety margin
        self.robot_radius = 0.4

        # Allowed distance to goal
        self.goal_epsilon = 1

        # Initializing dictionaries
        self.parent ={tuple(self.config_start): None}
        self.cost = {tuple(self.config_start): 0}

        # Total quadratic field of potential sampled nodes
        self.rrt_sampling_range = sampling_range

        # Radius for rewiring, how far to look for better paths
        self.rewire_radius = rewire_radius

        # Visual shape for points
        self.visual_shape_nodes = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.04, rgbaColor=[1, 0, 0, 1])
        self.visual_shape_goal = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1])
        self.visual_shape_spline = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
    
    def sample_new_node(self):
        """
        Sample a new node in the configuration space.
        """
        for i in range(100): # Try 100 times to sample a valid config.
            random_node  = np.random.uniform([-self.rrt_sampling_range, -self.rrt_sampling_range,0], [self.rrt_sampling_range, self.rrt_sampling_range,0], size=3)
            if not self.check_collision(random_node):
                # Return (x,y,0) if valid point
                return random_node


    def check_collision(self, node):
        """
        Check if a node is in collision with any of the obstacles.
        """
        for wall in self.obstacles:
            # Format of sphere: [x,y,z,radius]
            for sphere in wall:
                sphere_radius = sphere[3]

                # Check in 2D (x,y) for collision
                if self.distance(node, sphere[:3]) <= self.robot_radius + sphere_radius:
                    return True  # Collision detected
        return False  # No collision
    
        
    def distance(self, point1, point2):
        """
        Calculate the distance between two points.
        """
        return np.linalg.norm(np.array(point1[:2]) - np.array(point2[:2]))
    

    def add_node(self, new_node, parent_node):
        """
        Add a new node to the random exploring tree
        """
        self.node_list.append(new_node)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.visual_shape_nodes, basePosition=new_node)

        # Add new nodes to path and plot nodes
        self.path.append((np.array(parent_node), new_node))
        p.addUserDebugLine(parent_node,new_node, [0.2, 0.2, 0.2], lineWidth=3)
        p.addUserDebugLine(parent_node+[0,0,0.01],new_node+[0,0,0.01], [0.2, 0.2, 0.2], lineWidth=3)
        p.addUserDebugLine(parent_node+[0,0,0.02],new_node+[0,0,0.02], [0.2, 0.2, 0.2], lineWidth=3)

        # Add new node to parent and cost dictionary
        self.parent[tuple(new_node)] = tuple(parent_node)
        self.cost[tuple(new_node)] = self.cost[tuple(parent_node)] + self.distance(parent_node, new_node)


    def planning(self):
        """ 
        Main function for RRT* algorithm
        """
        while not self.is_goal_reached(self.node_list[-1]):
            new_node = self.sample_new_node()
            if new_node is not None:
                nearby_nodes = self.find_nearby_nodes(new_node, self.rewire_radius)
                best_parent, min_cost = self.choose_best_parent(new_node, nearby_nodes)
                if best_parent is not None:
                    self.add_node(new_node, best_parent)
                    self.rewire(new_node, nearby_nodes)

        # Last connection to goal node
        nearest_node_to_goal = self.find_nearest_node(self.config_goal)
        self.add_node(np.array(self.config_goal), nearest_node_to_goal)
        nearby_nodes = self.find_nearby_nodes(self.config_goal, self.rewire_radius)
        self.rewire(np.array(self.config_goal), nearby_nodes)


    def find_nearest_node(self, new_node):
        """ 
        Find the nearest node in the existing tree to the new node
        """
        closest_dist = float('inf')
        nearest_node = None
        for node in self.node_list:
            dist = self.distance(node, new_node)
            if dist < closest_dist:
                closest_dist = dist
                nearest_node = node
        return nearest_node


    def find_nearby_nodes(self, new_node, radius):
        """ 
        Find all nodes within a certain radius of the new node
        """
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
            
            parent_node = self.parent[current_node]
            
            path.append((current_node,parent_node))
            current_node = parent_node
        
        # Add the start node
        path.append((current_node, tuple(self.config_start)))
        path.reverse()  # Reverse the path to start from the beginning
        return path
    
    def visualize_path(self, path, spline=False):

        if spline:
            for i in range(len(path)-1):
                # Convert list to tuple for concatenation
                start_point, end_point = path[i], path[i+1]
                

                # Visualize all statr points as green spheres
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.visual_shape_spline, basePosition=start_point)


        else:
            print("path: ", path)   
            for i in range(len(path)-1):
                # Convert list to tuple for concatenation
                start_point, end_point = path[i], path[i+1]
                offset = (0, 0, 0.01)
                p.addUserDebugLine(np.array(start_point) + offset, np.array(end_point) + offset, [1, 0, 0], lineWidth=10)



    def calculate_path_cost(self, path):
        """
        Calculate the total cost of a given path.
        """
        total_cost = 0
        for i in range(len(path) - 1):
            start_node = tuple(path[i])
            end_node = tuple(path[i + 1])
            # Add the cost of moving from start_node to end_node
            total_cost += self.distance(start_node, end_node)
        return total_cost

