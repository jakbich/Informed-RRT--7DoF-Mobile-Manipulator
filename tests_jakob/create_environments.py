import gymnasium as gym
import numpy as np
import pybullet as p
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.collision_obstacle import CollisionObstacle

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle



def create_walls(env, walls, density=1, sphere_radius=0.2):
    all_obstacles = []
    for wall in walls:
        pos, length, width, height = wall
        begin_pos = [pos[0] - length / 2, pos[1] - width / 2, pos[2] - height / 2]
        end_pos = [pos[0] + length / 2, pos[1] + width / 2, pos[2] + height / 2]
        sphere_wall = add_3d_wall(env, begin_pos, end_pos, sphere_radius, density)
        all_obstacles.append(sphere_wall)

    return all_obstacles   
        

def fill_env_with_obstacles(env, obstacle_setup, density=1):

    if obstacle_setup == 'empty':
        all_obstacles = []

    if obstacle_setup == 'video':
        walls = [[[0, 0.4, 0.5], 1, 0.1, 0.7], 
                 [[0.5, 0.9, 0.5], 0.1, 1, 0.7], 
                 [[-0.5, 0.9, 0.5], 0.1, 1, 0.7],
                 [[0, 1.4, 0.5], 1, 0.1, 0.7]
               ]
        sphere_radius = 0.05
        all_obstacles1 = create_walls(env, walls, density, sphere_radius)

        walls = [[[0, -1.0, 0.5], 5.0, 0.3, 0.5],
        [[3, -3.0, 0.5], 5.0, 0.3, 0.5],
        [[0.6, -4.5, 0.5], 0.3, 3, 0.5]]

        sphere_radius = 0.2
        all_obstacles2 = create_walls(env, walls, density, sphere_radius)

        return all_obstacles1 + all_obstacles2
            

    if obstacle_setup == 'easy':

        sphere_radius = 0.2
        
        # Wall specifications [position, length, width, height]
        walls = [
            [[0, -1.0, 0.5], 5.0, 0.3, 1],
            [[2.5, 0, 0.5], 0.3, 5, 1]]
        
        all_obstacles = create_walls(env, walls, density, sphere_radius)
            

    if obstacle_setup == 'medium':

        sphere_radius = 0.2
        
        # Wall specifications [position, length, width, height]
        walls = [
            [[0, -1.0, 0.5], 5.0, 0.3, 0.5],
            [[3, -3.0, 0.5], 5.0, 0.3, 0.5],
            [[0.6, -4.5, 0.5], 0.3, 3, 0.5]]
        
        all_obstacles = create_walls(env, walls, density, sphere_radius)
            

    if obstacle_setup == 'advanced':
        
        sphere_radius = 0.2
        
        walls = [
        # Boxes
        [[-2.5, -3, 0.5], 1, 1, 1],  # Box 1
        [[0, -1.5, 0.5], 1.5, 1.5, 1],  # Box 2

        [[3.5, -1.5, 0.5], 2, 0.3, 1],  # Horizontal wall


        # Additional walls to create paths
        
        ]
        all_obstacles = create_walls(env, walls, density, sphere_radius)

    if obstacle_setup == 'labyrinth':

        sphere_radius = 0.1
            # Wall specifications [position, length, width, height]
        walls = [
            # Horizontal walls
            [[0, -1.0, 0.1], 3.0, 0.2, 0.2],
            [[-0.4, -2.5, 0.1], 4.8, 0.2, 0.2],
            [[0, 2.0, 0.1], 3.0, 0.2, 0.2],
            [[1.5, 5.0, 0.1], 3.0, 0.2, 0.2],

            # Vertical walls
            [[-1.5, 0.4, 0.1], 0.2, 3.0, 0.2],
            [[1.5, -0.35, 0.1], 0.2, 1.5, 0.2],
            [[3, 1.5, 0.1], 0.2, 7, 0.2],
            [[6, 1.5, 0.1], 0.2, 10, 0.2],
            [[-2.7, 1.3, 0.1], 0.2, 7, 0.2],

            # Box
            [[4.5, -1, 0.1], 1.2, 1.2, 0.2]
            
        ]

        all_obstacles = create_walls(env, walls, density, sphere_radius)


    if obstacle_setup == 'boxes':

        sphere_radius = 0.1
            # Wall specifications [position, length, width, height]
        walls = [
            # Box
            [[2.5, -1.15, 0.1], 1.2, 1.2, 0.2],
            [[2.5, 1.15, 0.1], 1.2, 1.2, 0.2]
            
            
        ]

        all_obstacles = create_walls(env, walls, density, sphere_radius)



    if obstacle_setup == 'hard':

        sphere_radius = 0.15
            # Wall specifications [position, length, width, height]
        walls = [
            [[0, -1.0, 0.5], 10.0, 0.2, 1],
            [[0, 1.0, 0.5], 10.0, 0.2, 1],
            [[0, -3.0, 0.5], 10.0, 0.2,1],
            [[0, 3.0, 0.5], 10.0, 0.2, 1],
            [[0, -6.0, 0.5], 16.0, 0.2, 1],
            [[0, 6.0,0.5], 16.0, 0.2, 1],
            [[8, 0, 0.5], 0.2, 12, 1],
            [[-8, 0, 0.5], 0.2, 12, 1]]
            
        all_obstacles = create_walls(env, walls, density, sphere_radius)

    return all_obstacles

def add_sphere(env, pos, radius):

    sphere_obst = SphereObstacle(name=f'obstacle_{pos[0]}_{pos[1]}_{pos[2]}', 
                                 content_dict={
                                    "type": "sphere",
                                    'movable': False,
                                    "geometry": {"position": pos, "radius": radius},
                                })
    env.add_obstacle(sphere_obst)



def add_3d_wall(env, start, end, radius=0.2, density = 1.0):
    # Calculate the number of spheres needed in each dimension
    n_spheres_x = max(1, abs(np.round((end[0] - start[0]) * density)/ (radius * 2)).astype(int))
    n_spheres_y = max(1, abs(np.round((end[1] - start[1]) * density/ (radius * 2))).astype(int))
    n_spheres_z = max(1, abs(np.round((end[2] - start[2]) * density/ (radius * 2))).astype(int))

    #print (f"n_spheres_x: {n_spheres_x} , n_spheres_y: {n_spheres_y} , n_spheres_z: {n_spheres_z}")

    spheres_wall = []
    # Add obstacles (spheres) to fill the volume
    for i in range(n_spheres_x):
        for j in range(n_spheres_y):
            for k in range(n_spheres_z):
                sphere_x = start[0] + i/density * radius * 2
                sphere_y = start[1] + j/density * radius * 2
                sphere_z = start[2] + k/density * radius * 2
                add_sphere(env, [sphere_x, sphere_y, sphere_z], radius)
                spheres_wall.append([sphere_x, sphere_y, sphere_z, radius])

    return spheres_wall



def add_box_obstacle(env, name, position, length, width, height, movable=False):
    """
    Add a box obstacle to the environment.

    Args:
        env (UrdfEnv): The URDF environment.
        name (str): The name of the obstacle.
        position (list): The position of the box [x, y, z].
        length (float): The length of the box.
        width (float): The width of the box.
        height (float): The height of the box.
        movable (bool): Whether the box is movable or not.

    Returns:
        None
    """
    box_obst_dict = {
        "type": "box",
        'movable': movable,
        "geometry": {
            "position": position,
            "length": length,
            "width": width,
            "height": height,
        },
    }

    box_obst = BoxObstacle(name=name, content_dict=box_obst_dict)
    env.add_obstacle(box_obst)

