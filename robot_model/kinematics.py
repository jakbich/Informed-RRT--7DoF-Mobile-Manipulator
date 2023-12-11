# https://frankaemika.github.io/docs/control_parameters.html

import numpy as np
import warnings

class Kinematics:
    def __init__(self, joint_angles):
        # Throw error when wrong amount of angles is provided
        if len(joint_angles) != 7:
            raise ValueError("The wrong amount of angles was provided.")
        
        # Add joint angle limits and speed limits
        self.angle_limits = [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973),
        ]
        self.speed_limits = [
            (2.1750,),
            (2.1750,),
            (2.1750,),
            (2.1750,),
            (2.6100,),
            (2.6100,),
            (2.6100,)
        ]
        
        # Make sure joint angle is within limits
        for i, (angle, (min, max)) in enumerate(zip(joint_angles, self.angle_limits)):
            if not (min <= angle <= max):
                corrected_angle = np.clip(angle, min, max)
                warnings.warn(f"Joint angle {angle} for joint {i+1} is out of limits: {min} to {max}. Setting to {corrected_angle}.")
                joint_angles[i] = corrected_angle
        
        self.dh_parameters = [
            [0, 0, 0.333, joint_angles[0]],
            [0, -np.pi/2, 0, joint_angles[1]],
            [0, np.pi/2, 0.316, joint_angles[2]],
            [0.0825, np.pi/2, 0, joint_angles[3]],
            [-0.0825, -np.pi/2, 0.384, joint_angles[4]],
            [0, np.pi/2, 0, joint_angles[5]],
            [0.088, np.pi/2, 0, joint_angles[6]],
            [0, 0, 0.107, 0]]                               #extra for the flange, not for the joints!

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

    def jacobian(self):
        numjoints = len(self.dh_parameters) - 1  # Excluding the extra for the flange
        J = np.zeros((6, numjoints))

        T_all = [np.eye(4)]  # Start with the identity matrix

        # Calculate transforms from the base frame to each joint
        for i in range(numjoints):
            a, alpha, d, theta = self.dh_parameters[i]
            T = self.transformation_matrix(a, alpha, d, theta)
            T_0_i = np.matmul(T_all[-1], T)
            T_all.append(T_0_i)

        T_0_end = T_all[-1]
        t_end = T_0_end[0:3, 3]

        # Build Jacobian
        for i in range(numjoints):
            T_0_i = T_all[i]
            z_i = T_0_i[0:3, 2]
            t_i = T_0_i[0:3, 3]

            J[0:3, i] = np.round(np.cross(z_i, (t_end - t_i)), 3)
            J[3:6, i] = np.round(z_i, 3)

        return J
    
if __name__ == "__main__":
    # Define the joint angles
    joint_angles = [0, 0, 0, -1, 0, 0, 0]

    kinematics = Kinematics(joint_angles)

    position = kinematics.forward_kinematics()
    print(f"The end effector position is: {position}")

    jacobian_matrix = kinematics.jacobian()
    print("Jacobian Matrix:")
    print(jacobian_matrix)

    

