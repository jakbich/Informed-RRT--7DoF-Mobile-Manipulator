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

    def __init__(self, config_start = (0,0,0), config_goal = (0,0,0), step_len = 0.1,iter_max = 100, obstacles= [], sampling_range=5, rewire_radius= 2, arm  = False):
        self.config_start = config_start
        self.config_goal = config_goal
        self.step_len = step_len
        self.iter_max = iter_max
        self.obstacles = obstacles

        self.node_list = [np.array(config_start)]
        self.path = []
        self.all_path_costs = []

        # Arm or mobile base
        self.arm = arm

        # Sphere around robot for collision checking safety margin
        self.robot_radius = 0.4

        # Allowed distance to goal
        self.goal_epsilon_base = 1
        self.goal_epsilon_arm = 0.4

        # Initializing dictionaries
        self.parent ={tuple(self.config_start): None}
        self.cost = {tuple(self.config_start): 0}

        # Total quadratic field of potential sampled nodes
        self.rrt_sampling_range = sampling_range

        # Radius for rewiring, how far to look for better paths
        self.rewire_radius = rewire_radius


        # Visual shape for points
        self.visual_shape_nodes = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.04, rgbaColor=[1, 0, 0, 1])
        self.visual_shape_spline = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
    
    def sample_new_node(self):
        """
        Sample a new node in the configuration space.
        """
        for i in range(100): # Try 100 times to sample a valid config.

            if not self.arm:
                random_node  = np.random.uniform([-self.rrt_sampling_range, -self.rrt_sampling_range,0], [self.rrt_sampling_range, self.rrt_sampling_range,0], size=3)
                if not self.check_collision(random_node):
                    # Return (x,y,0) if valid point
                    return random_node
            # If arm True, sample 3d point
            else:
                random_node  = np.random.uniform([-self.rrt_sampling_range, -self.rrt_sampling_range,0], [self.rrt_sampling_range, self.rrt_sampling_range,self.rrt_sampling_range], size=3)
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

                # Check in 3D (x,y,z) for collision
                if self.distance(node, sphere[:3]) <= self.robot_radius + sphere_radius:
                    return True  # Collision detected
        return False  # No collision
    
        
    def distance(self, point1, point2):
        """
        Calculate the distance between two points.
        """
        return np.linalg.norm(np.array(point1[:3]) - np.array(point2[:3]))
    

    def add_node(self, new_node, parent_node):
        """
        Add a new node to the random exploring tree
        """
        # Plot the node 
        self.node_list.append(new_node)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.visual_shape_nodes, basePosition=new_node)

        # Add new nodes to path
        self.path.append((np.array(parent_node), new_node))

        # Add new node to parent and cost dictionary
        self.parent[tuple(new_node)] = tuple(parent_node)
        self.cost[tuple(new_node)] = self.cost[tuple(parent_node)] + self.distance(parent_node, new_node)

    def plot_edge(self, new_node, parent_node): 
        # Plot the edge
        p.addUserDebugLine(np.array(parent_node),np.array(new_node), [0.2, 0.2, 0.2], lineWidth=3)
        p.addUserDebugLine(np.array(parent_node)+[0,0,0.01],np.array(new_node)+(0,0,0.01), [0.2, 0.2, 0.2], lineWidth=3)
  

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
                    self.plot_edge(new_node, best_parent)

        # Last connection to goal node
        nearest_node_to_goal = self.find_nearest_node(self.config_goal)
        self.add_node(np.array(self.config_goal), nearest_node_to_goal)
        nearby_nodes = self.find_nearby_nodes(self.config_goal, self.rewire_radius)
        self.rewire(np.array(self.config_goal), nearby_nodes)  
        self.plot_edge(self.config_goal, nearest_node_to_goal)

    
      


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
        if not self.arm:
            return self.distance(node, self.config_goal) < self.goal_epsilon_base
        
        else:
            return self.distance(node, self.config_goal) < self.goal_epsilon_arm

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
    
    def visualize_path(self, path, spline=False, color=[1, 0, 0]):

        if spline:
            for i in range(len(path)-1):
                # Convert list to tuple for concatenation
                start_point, end_point = path[i], path[i+1]
                

                # Visualize all statr points as green spheres
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.visual_shape_spline, basePosition=start_point)


        else:
            for i in range(len(path)-1):
                # Convert list to tuple for concatenation
                start_point, end_point = path[i], path[i+1]
                offset = (0, 0, 0.02)

                # Draw blue line from start to goal for old and new path
                p.addUserDebugLine(np.array(start_point) + offset, np.array(end_point) + offset, color, lineWidth=10)



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

class InformedRRTStar (RRTStar):
    def __init__(self, config_start = (0,0,0), config_goal = (0,0,0), step_len = 0.1,iter_max = 100, obstacles= [], sampling_range=5, rewire_radius= 2):
        super().__init__(config_start, config_goal, step_len, iter_max, obstacles, sampling_range, rewire_radius)

        # Ellipsoid data
        self.best_path_cost = float('inf')  # Initialize the best path cost
        self.ellipsoid = None  # Placeholder for the ellipsoid data

    def planning(self):
        super().planning()
        # Visualize first path
        first_path = np.array(self.find_path())
        
        # After finding a path to the goal
        path_to_goal = np.array(self.find_path())
        path_cost = self.calculate_path_cost(path_to_goal)
        self.all_path_costs.append(path_cost)  # Record the path cost

        self.visualize_path(first_path[:,0,:], color=[0, 0, 1])

        self.calculate_ellipsoid()
        self.planning_ellipsoid()

    def planning_ellipsoid(self):
        """ 
        Main function for RRT* algorithm
        """
        paths_found = 0

        for _ in tqdm(range(200)):
            new_node = self.sample_new_node_ellipsoid()
            if new_node is not None:
                nearby_nodes = self.find_nearby_nodes(new_node, self.rewire_radius)
                best_parent, min_cost = self.choose_best_parent(new_node, nearby_nodes)
                if best_parent is not None:
                    self.add_node(new_node, best_parent)
                    self.rewire(new_node, nearby_nodes)
                    self.plot_edge(new_node, best_parent)
                
                    path_to_goal = np.array(self.find_path())
                    path_cost = self.calculate_path_cost(path_to_goal)
                    if path_cost < self.all_path_costs[-1]:
                        self.all_path_costs.append(path_cost)  # Record the path cost

    def sample_new_node_ellipsoid(self):
        """
        Sample a new node in the ellipsoidal space
        """
        for i in range(100): # Try 100 times to sample a valid config.
            random_node  = np.random.uniform([-self.rrt_sampling_range, -self.rrt_sampling_range,0], [self.rrt_sampling_range, self.rrt_sampling_range,0], size=3)
            if not self.check_collision(random_node) and self.is_point_in_ellipse(random_node[0], random_node[1], self.distance_start_goal, self.width_ellipse, self.config_start, self.direction_vector):
                # Return (x,y,0) if valid point and inside of the ellipsoid shape
                return random_node

    def calculate_ellipsoid(self):
        """
        Calculate the ellipsoid data.
        """
        # Find the best path
        self.path_to_goal = self.find_path()
        # Calculate the cost of the best path
        self.path_to_goal_cost = self.calculate_path_cost(self.path_to_goal)
        
        # Create an initial ellipsoid that has the goal and start as its foci
        self.direction_vector = np.array(self.config_goal) - np.array(self.config_start)
        self.distance_start_goal = np.linalg.norm(self.direction_vector)
        
        self.width_ellipse = self.distance_start_goal / 2

        # Rotate and translate the ellipse
        rotation_angle = np.arctan2(self.direction_vector[1], self.direction_vector[0])
        self.rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])
        

        # Adjust the ellipse until all points in the path are inside the ellipse
        self.adjust_ellipsoid()
        # Draw the ellipsoid
        self.draw_ellipsoid()


    def draw_ellipsoid(self):

        # Draw ellipsoid
        t = np.linspace(0, 2 * np.pi, 100)
        ellipsoid_points = np.array([self.distance_start_goal * np.cos(t), self.width_ellipse * np.sin(t)])

        ellipsoid_points = np.matmul(self.rotation_matrix, ellipsoid_points)
        ellipsoid_points[0, :] += self.config_start[0] + self.direction_vector[0] / 2
        ellipsoid_points[1, :] += self.config_start[1] + self.direction_vector[1] / 2


        # Plot all points in the ellipse
        visual_shape_goal = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.3, 0.5, 0.3, 1])

        for i in range(len(t)):
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_goal, basePosition=[ellipsoid_points[0, i], ellipsoid_points[1, i], 0])


    def is_point_in_ellipse(self, point_x, point_y, distance_start_goal, width_ellipse, config_start, direction_vector):
        """
        Check if the point (x, y) is inside the rotated and shifted ellipse.

        :param point_x: x-coordinate of the point
        :param point_y: y-coordinate of the point
        :param distance_start_goal: length of the semi-major axis of the ellipse
        :param width_ellipse: length of the semi-minor axis of the ellipse
        :param config_start: starting point of the ellipse
        :param direction_vector: vector from start to goal of the ellipse
        :return: True if the point is inside the ellipse, False otherwise
        """
        # Translate the point back
        translated_point_x = point_x - (config_start[0] + direction_vector[0] / 2)
        translated_point_y = point_y - (config_start[1] + direction_vector[1] / 2)

        # Calculate the rotation angle
        rotation_angle = np.arctan2(direction_vector[1], direction_vector[0])
        # Create the inverse rotation matrix
        inverse_rotation_matrix = np.array([[np.cos(rotation_angle), np.sin(rotation_angle)],
                                            [-np.sin(rotation_angle), np.cos(rotation_angle)]])
        # Rotate the point back
        rotated_point = np.matmul(inverse_rotation_matrix, np.array([translated_point_x, translated_point_y]))

        # Check if the point is inside the standard ellipse
        return (rotated_point[0]**2 / distance_start_goal**2) + (rotated_point[1]**2 / width_ellipse**2) <= 1


    def adjust_ellipsoid(self):
        # Inside calculate_ellipsoid method
        counter_in_elipse = 0
        while counter_in_elipse < len(self.path_to_goal):
            counter_in_elipse = 0
            for i in range(len(self.path_to_goal)):
                if not self.is_point_in_ellipse(self.path_to_goal[i][0][0], self.path_to_goal[i][0][1], self.distance_start_goal, self.width_ellipse, self.config_start, self.direction_vector):
                    print(f"Point {i} not in ellipse")
                    self.width_ellipse *= 1.1
                else:
                    counter_in_elipse += 1

        
