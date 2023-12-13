import numpy as np
import warnings
from sympy import symbols, cos, sin, Matrix, lambdify, pi


class Kinematics3joints:
    def __init__(self, joint_angles, base_transformation=np.eye(4)):
        if len(joint_angles) != 3:
            raise ValueError("The wrong amount of angles was provided.")
        
        self.angle_limits = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973)]
        
        for i, (angle, (min_angle, max_angle)) in enumerate(zip(joint_angles, self.angle_limits)):
            if not (min_angle <= angle <= max_angle):
                corrected_angle = np.clip(angle, min_angle, max_angle)
                warnings.warn(f"Joint angle {angle} for joint {i+1} is out of limits: {min_angle} to {max_angle}. Setting to {corrected_angle}.")
                joint_angles[i] = corrected_angle

        self.dh_parameters = [
            [0, 0, 0.333, joint_angles[0]],
            [0, -np.pi/2, 0, joint_angles[1]],
            [0, np.pi/2, 0.316, joint_angles[2]]

        ]
        self.base_transformation = base_transformation

    def dh_transform(self, a, alpha, d, theta):
        T = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        return T
    
    def calculate_transformation_matrix(self, a, alpha, d, theta):
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cq = np.cos(theta)
        sq = np.sin(theta)

        # Define transformation matrix for the current joint
        transform = Matrix([
            [cq, -sq, 0, a],
            [ca * sq, ca * cq, -sa, -d * sa],
            [sa * sq, cq * sa, ca, d * ca],
            [0, 0, 0, 1]
        ])

        return transform

    # def forward_kinematics(self):
    #     T_all = [self.base_transformation]
    #     for a, alpha, d, theta in self.dh_parameters:
    #         T = self.dh_transform(a, alpha, d, theta)
    #         T_0_i = np.dot(T_all[-1], T)
    #         T_all.append(T_0_i)

    #     return T_all[-1]
    
    def forward_kinematics(self):
        # Initialize the transformation matrix
        T_total = self.base_transformation

        # Iterate over the three joints
        for dh_params in self.dh_parameters:
            a, alpha, d, theta = dh_params

            # Calculate the transformation matrix for the current joint
            T_joint = self.calculate_transformation_matrix(a, alpha, d, theta)

            # Multiply the total transformation matrix by the current joint's transformation
            T_total = np.dot(T_total, T_joint)

        return T_total