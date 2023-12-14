import numpy as np
import math


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


    def update(self, current_pos=[0,0,0], goal_pos=[10,10,10], action=np.zeros(12)):

        angular_thresh = math.pi/120
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

            control_velocity = self.kp[0] * self.error_linear + self.ki[0] * self.integral_linear + self.kd[0] * derivative_linear            
            control_velocity = np.clip(control_velocity, 0, max_linear_vel)

            action[0] = action[0] * 0.95 + control_velocity * 0.05


            if scalar_product < 0:
                action[0] = -action[0]

       
        return action