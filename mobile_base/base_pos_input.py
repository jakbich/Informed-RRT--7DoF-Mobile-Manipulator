import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher

def run_point_robot_to_targets(targets, n_steps=1000, render=False, constant_velocity=0.1, tolerance=1.0):
    # Create the point robot
    robot = GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel")

    # Create the environment
    env = gym.make("urdf-env-v0", dt=0.01, robots=[robot], render=render)

    # Set the initial robot position and velocity
    initial_position = np.array([1.0, 0.1, 0.0])
    initial_velocity = np.array([1.0, 0.0, 0.0])
    ob = env.reset(pos=initial_position, vel=initial_velocity)

    # Print initial observation
    print(f"Initial observation: {ob}")

    for target_position in targets:
        for _ in range(n_steps):
            # Get the first key-value pair in the observation dictionary
            robot_key, robot_observation = next(iter(ob[0].items()))

            # Ensure the key is what we expect it to be
            if robot_key == 'robot_0':
                robot_position = robot_observation['joint_state']['position'][:3]
                direction = target_position - robot_position
                distance = np.linalg.norm(direction)
                if distance < 0.1:  # Assuming a small threshold for reaching the target
                    break

                direction /= np.linalg.norm(direction)

                # Apply constant velocity in the direction of the target
                action = direction * constant_velocity

                ob, _, terminated, _, info = env.step(action)

                if terminated:
                    print(info)
                    break

    env.close()

if __name__ == "__main__":
    # Define the target positions (x, y, z)
    targets = [np.array([200.0, 0.0, 0.0]), np.array([0.0, 200.0, 0.0])]

    # Run the point robot to the target positions
    run_point_robot_to_targets(targets, render=True)
