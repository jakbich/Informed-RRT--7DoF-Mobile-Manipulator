import gym
import numpy as np
import time
from kinematics import Kinematics

from urdfenvs.robots.albert import AlbertRobot
from urdfenvs.sensors.full_sensor import FullSensor

def test_robot_arm_control_with_env():
    # Initialize the Albert robot with velocity control mode
    robot = AlbertRobot(mode="vel")
    
    # Add a full sensor to the robot
    sensor = FullSensor(goal_mask=['position', 'radius'], obstacle_mask=['position', 'radius'])
    robot.add_sensor(sensor)

    # Create the Gym environment
    env = gym.make("urdf-env-v0", dt=0.01, robots=[robot], render=True)

    # Define joint configurations
    joint_configurations = [
        [0, 0, 0, -1, 0, 0, 0],
        [0, 0.5, 0, -1, 0, 0, 0],
        # ... more configurations ...
    ]

    for joint_angles in joint_configurations:
        # Initialize Kinematics with the joint angles
        kinematics = Kinematics(joint_angles)

        # Calculate the desired end-effector position
        position = kinematics.forward_kinematics()

        # Move the robot in the environment
        # Assuming the first 3 joints are not controllable and the last 7 are
        action = np.hstack((np.zeros(3), joint_angles))
        ob, _, _, _ = env.step(action)

        # Wait for a short period to simulate the motion
        time.sleep(1)

        print(f"Moved to joint configuration: {joint_angles}")
        print(f"End effector position should be: {position}")

    env.close()

if __name__ == "__main__":
    test_robot_arm_control_with_env()
