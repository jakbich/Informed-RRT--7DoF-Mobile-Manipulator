import gymnasium as gym
import numpy as np
import pybullet as p
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.collision_obstacle import CollisionObstacle

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle


def fill_env_with_obstacles(env, obstacle_setup, density=1):

    if obstacle_setup == 'empty':
        pass

    if obstacle_setup == 'easy':

        sphere_radius = 0.2
        
        # Wall specifications [position, length, width, height]
        walls = [
            [[0, -1.0, 0.0], 5.0, 0.3, 2],
            [[2.5, 0, 0.0], 0.3, 5, 2]]
        
        for wall in walls:
            pos, length, width, height = wall
            begin_pos = [pos[0] - length / 2, pos[1] - width / 2, pos[2] - height / 2]
            end_pos = [pos[0] + length / 2, pos[1] + width / 2, pos[2] + height / 2]
            add_3d_wall(env, begin_pos, end_pos, sphere_radius, density)


    if obstacle_setup == 'hard':

        sphere_radius = 0.1
            # Wall specifications [position, length, width, height]
        walls = [
            [[0, -1.0, 0.0], 10.0, 0.2, 2],
            [[0, 1.0, 0.0], 10.0, 0.2, 2],
            [[0, -3.0, 0.0], 10.0, 0.2, 2],
            [[0, 3.0, 0.0], 10.0, 0.2, 2],
            [[0, -6.0, 0.0], 16.0, 0.2, 2],
            [[0, 6.0, 0.0], 16.0, 0.2, 2],
            [[8, 0.0, 0.0], 0.2, 12, 2],
            [[-8, 0.0, 0.0], 0.2, 12, 2]]

        # Create each wall
        for wall in walls:
            pos, length, width, height = wall
            begin_pos = [pos[0] - length / 2, pos[1] - width / 2, pos[2] - height / 2]
            end_pos = [pos[0] + length / 2, pos[1] + width / 2, pos[2] + height / 2]
            add_3d_wall(env, begin_pos, end_pos, sphere_radius, density)


        ########## Adding obstacles to the environment ##########
        # add_box_obstacle(
        #     env=env,
        #     name='wall_1',
        #     position=[0, -1.0, 0.0],
        #     length=10.0,
        #     width=0.2,
        #     height=2,
        #     movable=False
        # )
        ################################################################




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
    n_spheres_x = abs(np.round((end[0] - start[0]) * density)/ (radius * 2)).astype(int)
    n_spheres_y = abs(np.round((end[1] - start[1]) * density/ (radius * 2)).astype(int))
    n_spheres_z = abs(np.round((end[2] - start[2]) * density/ (radius * 2)).astype(int))

    print (f"n_spheres_x: {n_spheres_x} , n_spheres_y: {n_spheres_y} , n_spheres_z: {n_spheres_z}")

    # Add obstacles (spheres) to fill the volume
    for i in range(n_spheres_x):
        for j in range(n_spheres_y):
            for k in range(n_spheres_z):
                sphere_x = start[0] + i/density * radius * 2
                sphere_y = start[1] + j/density * radius * 2
                sphere_z = start[2] + k/density * radius * 2
                add_sphere(env, [sphere_x, sphere_y, sphere_z], radius)



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

