import gymnasium as gym
import warnings
import numpy as np
import argparse
import math
import pybullet as p
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
 
from create_environments import fill_env_with_obstacles


def normalize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi].
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def get_robot_config(ob):
    return ob['robot_0']['joint_state']['position']


def get_robot_velocity(ob):
    return ob['robot_0']['joint_state']['forward_velocity'][0]



class PIDBase:
    def __init__(self, kp=[0,0], ki=[0,0], kd=[0,0], dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.last_error_linear = 0
        self.last_error_angular= 0
        
        self.integral_linear = 0
        self.integral_angular = 0

        self.count_reached = 0


    def update(self, current_pos=[0,0,0], goal_pos=[10,10,10]):


        action = np.zeros(12)
        angular_thresh = math.pi/100
        linear_thresh = 0.1
        max_linear_vel = 1.5


         
        # Angular
        self.error_angular = normalize_angle(np.arctan2(goal_pos[1] - current_pos[1], goal_pos[0] - current_pos[0]) - current_pos[2])
        self.error_linear = np.linalg.norm(np.array(goal_pos[0:2]) - np.array(current_pos[0:2]))


        # if we are close enough to the goal
        if self.error_linear < linear_thresh:
            self.error_linear = 0
            self.error_angular = 0  
            action[0] = -0.01
            print("backward motion")
            self.count_reached += 1

            return action
        
        # Calculate vector from our pos to goal
        vector_to_goal = np.array(goal_pos[0:2]) - np.array(current_pos[0:2])

        # Calculate vector that the robot is facing
        facing_vector = np.array([np.cos(current_pos[2]), np.sin(current_pos[2])])

        # Calculate scalar product to determine whether vectors face the same direction
        scalar_product = np.dot(vector_to_goal, facing_vector)


        if abs(self.error_angular) > angular_thresh:

            self.integral_angular += self.error_angular * self.dt
            derivative_angular = (self.error_angular - self.last_error_angular) / self.dt
            self.last_error_angular = self.error_angular
            action[1] = self.kp[1] * self.error_angular + self.ki[1] * self.integral_angular + self.kd[1] * derivative_angular

        else:
            # Linear

            self.integral_linear += self.error_linear * self.dt
            derivative_linear = (self.error_linear - self.last_error_linear) / self.dt
            self.last_error_linear = self.error_linear

            #action[0] = min(self.kp[0] * self.error_linear + self.ki[0] * self.integral_linear + self.kd[0] * derivative_linear,1)
            action[0] =  self.kp[0] * self.error_linear + self.ki[0] * self.integral_linear + self.kd[0] * derivative_linear
            action[0] = np.clip(action[0], 0, max_linear_vel)

            if scalar_product < 0:
                action[0] = -action[0]

       
        return action



def run_albert(n_steps=10000, render=False, goal=True, obstacles=True, env_type='empty', sphere_density=1.0):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494, 
            spawn_rotation = math.pi/2,       # in radians
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )


    # Set the target position
    target_positions = np.array([[-0.5,-0.5,0], [1,-1,0]])#,[0,5, 0], [5,0,0 ]])

    # Add axes at the origin (you can change the position as needed)
    origin = [0, 0, 0]
    axis_length = 10.0 # Length of each axis
    p.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0], 2.0)  # X-axis in red
    p.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0], 2.0)  # Y-axis in green
    p.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1], 2.0)  # Z-axis in blue
    # Add a visual marker at the target position
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])

    for target in target_positions:
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=target)




    # Fill the environment with obstacles, argument passed to determine which one (empty, easy, hard):
    fill_env_with_obstacles(env, env_type, sphere_density)

    action = np.zeros(env.n())

    ob = env.reset(
        pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    
    print(f"Initial observation : {ob}")
    history = []

    pid_controller = PIDBase(kp=[1, 3], ki=[0.0, 0.0], kd=[0, 0], dt=0.01)

    for step in range(n_steps):

        ob, *_ = env.step(action)
        current_obs  = ob['robot_0']['joint_state']['position']

        # Track reached positions
        if pid_controller.count_reached != len(target_positions):
            action = pid_controller.update(current_pos=current_obs[0:3], goal_pos=target_positions[pid_controller.count_reached])

        else:
            print("All positions reached!")
            action = np.zeros(env.n())

        if step % 100 == 0:
            print (f"current_obs: {current_obs[0:3]} ,\n action: {action[0:2]}")
            print(f"PID_ERROR_linear", pid_controller.error_linear)
            print(f"PID_ERROR_angular", pid_controller.error_angular)
            print (f"step: {step}")
            #print (f"lin_error {pid_controller.error_linear}, ang_error {pid_controller.error_angular}\n\n")

        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False

    parser = argparse.ArgumentParser(description='Fill environment with obstacles.')
    parser.add_argument('--env_type', type=str, help='Type of the environment to create', default='empty')
    parser.add_argument('--sphere_density', type=float, help='sphere_density to make spheres overlap', default=1.0)
    args = parser.parse_args()
    env_type = args.env_type
    sphere_density = args.sphere_density


    # pid_controller = PIDBase(kp=[0.5, 0.5], ki=[0.0, 0.0], kd=[0.0, 0.0], dt=0.01)
    # print(pid_controller.update(current_pos=[0,0,0], goal_pos=[-10,-10,-10])[0:3])


    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True, env_type=env_type, sphere_density=sphere_density)
