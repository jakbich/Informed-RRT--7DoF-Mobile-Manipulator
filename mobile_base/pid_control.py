import numpy as np
import math
from scipy import interpolate

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


    def update(self, current_pos=[0,0,0], goal_pos=[10,10,0], action=np.zeros(12), last_step=False):

        # Thresholds and max values
        angular_thresh = math.pi/10
        linear_thresh = 0.2
        max_linear_vel = 1.5

        if last_step:
            min_linear_vel = 0.0

        else:
            min_linear_vel = 0.8

        # Angular
        self.error_angular = normalize_angle(np.arctan2(goal_pos[1] - current_pos[1], goal_pos[0] - current_pos[0]) - current_pos[2])
        self.error_linear = np.linalg.norm(np.array(goal_pos[0:2]) - np.array(current_pos[0:2]))


        # if we are close enough to the goal
        if self.error_linear < linear_thresh:
            self.error_linear = 0
            self.error_angular = 0  
            action[0] = -0.01
            action[1] = 0.0

            if not last_step:
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
            control_velocity = np.clip(control_velocity, max_linear_vel, max_linear_vel)

            
            action[0] = action[0] * 0.95 + control_velocity * 0.05


            if scalar_product < 0:
                action[0] = -action[0]

       
        return action
    

def interpolate_path(path, max_dist=5.0):
    """
    This function interpolates points bet between nodes in case nodes are far apart
    """
    interpolated_path = []

    for i in range(len(path) - 1):
        x1 = path[i][0][0]
        y1 = path[i][0][1]
        z1 = path[i][0][2]
        x2 = path[i][1][0]
        y2 = path[i][1][1]
        z2 = path[i][1][2]

        x_dist = x2 - x1
        y_dist = y2 - y1
        z_dist = z2 - z1
        edge_length = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
        interpolated_path.append(path[i])

        if edge_length > max_dist:
            n_nodes = (edge_length // max_dist).astype(int)
            for ii in range(n_nodes):
                interpolated_path.append(
                    np.array([x1 + (ii + 1) * x_dist / (n_nodes + 1), y1 + (ii + 1) * y_dist / (n_nodes + 1), z1 + (ii + 1) * z_dist / (n_nodes + 1)]))

    interpolated_path.append(path[-1])
    return interpolated_path
    



def path_smoother(shortest_path_configs, total_cost_path=10.0):
    x = []
    y = []
    z = []

 
    for point in shortest_path_configs:

        x.append(point[0][0])
        y.append(point[0][1])
        z.append(point[0][2])
    tck, *rest = interpolate.splprep([x, y, z], s=0.1)

    # Create a number of points on the spline according to path cost
    num_steps = int(total_cost_path) * 30
    u = np.linspace(0, 1, num=num_steps)

    x_smooth, y_smooth, z_smooth = interpolate.splev(u, tck)
    smooth_configs = [np.array([x_smooth[i], y_smooth[i], z_smooth[i]]) for i in range(len(x_smooth))]
    return np.array(smooth_configs)



