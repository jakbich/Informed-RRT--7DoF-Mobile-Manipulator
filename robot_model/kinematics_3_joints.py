# import numpy as np
# import warnings

# class Kinematics3joints:
#     def __init__(self, joint_angles):
#         # Throw error when wrong amount of angles is provided
#         if len(joint_angles) != 3:
#             raise ValueError("The wrong amount of angles was provided.")
        
#         # Update joint angle limits and speed limits for three joints
#         self.angle_limits = [
#             (-2.8973, 2.8973),
#             (-1.7628, 1.7628),
#             (-2.8973, 2.8973)
#         ]
#         self.speed_limits = [
#             (2.1750,),
#             (2.1750,),
#             (2.1750,)
#         ]
        
#         # Ensure joint angles are within limits
#         for i, (angle, (min, max)) in enumerate(zip(joint_angles, self.angle_limits)):
#             if not (min <= angle <= max):
#                 corrected_angle = np.clip(angle, min, max)
#                 warnings.warn(f"Joint angle {angle} for joint {i+1} is out of limits: {min} to {max}. Setting to {corrected_angle}.")
#                 joint_angles[i] = corrected_angle
        
#         # Update DH parameters for the first three joints only
#         self.dh_parameters = [
#             [0, 0, 0.333, joint_angles[0]],
#             [0, -np.pi/2, 0, joint_angles[1]],
#             [0, np.pi/2, 0.316, joint_angles[2]]
#         ]

    
#     def transformation_matrix(self, a, alpha, d, theta):
#         T = np.array([  
#             [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
#             [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
#             [0, np.sin(alpha), np.cos(alpha), d],
#             [0, 0, 0, 1]])
#         return T

#     def forward_kinematics(self):
#         T = np.eye(4)
#         for parameters in self.dh_parameters:
#             a = parameters[0]
#             alpha = parameters[1]
#             d = parameters[2]
#             theta = parameters[3]

#             Ti = self.transformation_matrix(a, alpha, d, theta)
#             T = np.dot(T, Ti)

#         x, y, z = T[0, 3], T[1, 3], T[2, 3]

#         return x, y, z

#     def jacobian(self):
#         numjoints = len(self.dh_parameters)
#         J = np.zeros((6, numjoints))

#         T_all = [np.eye(4)]

#         # Calculate transforms for the first three joints
#         for i in range(numjoints):
#             a, alpha, d, theta = self.dh_parameters[i]
#             T = self.transformation_matrix(a, alpha, d, theta)
#             T_0_i = np.matmul(T_all[-1], T)
#             T_all.append(T_0_i)

#         T_0_end = T_all[-1]
#         t_end = T_0_end[0:3, 3]

#         # Build Jacobian for the first three joints
#         for i in range(numjoints):
#             T_0_i = T_all[i]
#             z_i = T_0_i[0:3, 2]
#             t_i = T_0_i[0:3, 3]

#             J[0:3, i] = np.round(np.cross(z_i, (t_end - t_i)), 3)
#             J[3:6, i] = np.round(z_i, 3)

#         return J

# # Usage example
# # joint_angles = [0, 0, 0]
# # kinematics = Kinematics(joint_angles)
# # position = kinematics.forward_kinematics()
# # print(f"The end effector position is: {position}")
# # jacobian_matrix = kinematics.jacobian()
# # print("Jacobian Matrix:")
# # print(jacobian_matrix)

import numpy as np
import warnings

class Kinematics3joints:
    def __init__(self, joint_angles):
        # Validate the number of joint angles provided
        if len(joint_angles) != 3:
            raise ValueError("The wrong amount of angles was provided.")
        
        # Define joint angle limits for the three joints
        self.angle_limits = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973)]
        
        # Ensure joint angles are within limits
        for i, (angle, (min_angle, max_angle)) in enumerate(zip(joint_angles, self.angle_limits)):
            if not (min_angle <= angle <= max_angle):
                corrected_angle = np.clip(angle, min_angle, max_angle)
                warnings.warn(f"Joint angle {angle} for joint {i+1} is out of limits: {min_angle} to {max_angle}. Setting to {corrected_angle}.")
                joint_angles[i] = corrected_angle

        # Define DH parameters for the first three joints
        self.dh_parameters = [
            [0, 0, 0.333, joint_angles[0]],
            [0, -np.pi/2, 0, joint_angles[1]],
            [0, np.pi/2, 0.316, joint_angles[2]]
        ]

    def dh_transform(self, alpha, a, d, theta):
        # Calculate the DH transformation matrix
        T = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        return T

    def forward_kinematics(self):
        # Calculate forward kinematics
        T = np.eye(4)
        for alpha, a, d, theta in self.dh_parameters:
            Ti = self.dh_transform(alpha, a, d, theta)
            T = np.dot(T, Ti)

        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        return x, y, z

# Usage example
# joint_angles = [0, 0, 0]
# kinematics = Kinematics3joints(joint_angles)
# position = kinematics.forward_kinematics()
# print(f"The end effector position is: {position}")
