# https://frankaemika.github.io/docs/control_parameters.html

import numpy as np

class Kinematics:
    def __init__(self, joint_angles):
        if len(joint_angles) != 7:
            raise ValueError("The wrong amount of angles was provided.")
        
        self.dh_parameters = [
            [0, 0, 0.333, joint_angles[0]],
            [0, -np.pi/2, 0, joint_angles[1]],
            [0, np.pi/2, 0.316, joint_angles[2]],
            [0.0825, np.pi/2, 0, joint_angles[3]],
            [-0.0825, -np.pi/2, 0.384, joint_angles[4]],
            [0, np.pi/2, 0, joint_angles[5]],
            [0.088, np.pi/2, 0, joint_angles[6]],
            [0, 0, 0.107, 0]]                               #extra for the flange, not for the joints!
        
        self.joint_limits = [
            (-2.8973, 2.8973),  # Joint 1 limits
            (-1.7628, 1.7628),  # Joint 2 limits
            (-2.8973, 2.8973),  # Joint 3 limits 
            (-3.0718, -0.0698), # Joint 4 limits 
            (-2.8973, 2.8973),  # Joint 5 lim
            (-0.0175, 3.7525),  # Joint 6 limits
            (-2.8973, 2.8973),  # Joint 7 limits
        ]

        # Check joint limits
        for idx, (angle, (min_limit, max_limit)) in enumerate(zip(joint_angles, self.joint_limits)):
            if not (min_limit <= angle <= max_limit):
                raise ValueError(f"Joint angle {angle} for joint {idx+1} is out of limits: {min_limit} to {max_limit}.")

    def transformation_matrix(self, a, alpha, d, theta):
        T = np.array([  
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]])
        return T

    def forward_kinematics(self):
        T = np.eye(4)
        for parameters in self.dh_parameters:
            a = parameters[0]
            alpha = parameters[1]
            d = parameters[2]
            theta = parameters[3]

            Ti = self.transformation_matrix(a, alpha, d, theta)
            T = np.dot(T, Ti)

        x, y, z = T[0, 3], T[1, 3], T[2, 3]

        return x, y, z
    
    def jacobian(self, eps=1e-6):
        J = np.zeros((3, len(self.dh_parameters) - 1))  # 3 rows for x, y, z; columns for each joint without the flange
        original_position = self.forward_kinematics()

        for i in range(len(self.dh_parameters) - 1):
            # Change each joint angle slightly
            theta = self.dh_parameters[i][3]
            self.dh_parameters[i][3] += eps
            changed_position = self.forward_kinematics()
            self.dh_parameters[i][3] = theta  # Reset joint angle

            # Compute partial derivative for each joint
            J[:, i] = (changed_position - original_position) / eps

        return J

    def inverse_kinematics(self, x, y, z):
        q = []
        return q
    
if __name__ == "__main__":
    # Define the joint angles
    joint_angles = [0, 0, 0, -1, 0, 0, 0]

    kinematics = Kinematics(joint_angles)

    position = kinematics.forward_kinematics()
    print(f"The end effector position is: {position}")


    # x, y, z = 10, 0, 5 
    # print(kinematics.inverse_kinematics(x, y, z))
    

