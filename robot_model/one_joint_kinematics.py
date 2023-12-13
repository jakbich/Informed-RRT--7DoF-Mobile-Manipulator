# # https://frankaemika.github.io/docs/control_parameters.html

# import numpy as np
# import warnings
# import math

# class Kinematics:
#     def __init__(self, joint_angles):
#         # Throw error when wrong amount of angles is provided
#         if len(joint_angles) != 7:
#             raise ValueError("The wrong amount of angles was provided.")
        
#         # Add joint angle limits and speed limits
#         self.angle_limits = [
#             (-2.8973, 2.8973),
#             (-1.7628, 1.7628),
#             (-2.8973, 2.8973),
#             (-3.0718, -0.0698),
#             (-2.8973, 2.8973),
#             (-0.0175, 3.7525),
#             (-2.8973, 2.8973),
#         ]
#         self.speed_limits = [
#             (2.1750,),
#             (2.1750,),
#             (2.1750,),
#             (2.1750,),
#             (2.6100,),
#             (2.6100,),
#             (2.6100,)
#         ]
        
#         # Make sure joint angle is within limits
#         for i, (angle, (min, max)) in enumerate(zip(joint_angles, self.angle_limits)):
#             if not (min <= angle <= max):
#                 corrected_angle = np.clip(angle, min, max)
#                 # warnings.warn(f"Joint angle {angle} for joint {i+1} is out of limits: {min} to {max}. Setting to {corrected_angle}.")
#                 joint_angles[i] = corrected_angle
        
#         self.dh_parameters = [
#             [0, 0, 0.333, joint_angles[0]],
#             [0, -np.pi/2, 0, joint_angles[1]],
#             [0, np.pi/2, 0.316, joint_angles[2]],
#             [0.0825, np.pi/2, 0, joint_angles[3]],
#             [-0.0825, -np.pi/2, 0.384, joint_angles[4]],
#             [0, np.pi/2, 0, joint_angles[5]],
#             [0.088, np.pi/2, 0, joint_angles[6]],
#             [0, 0, 0.107, 0]]                               #extra for the flange, not for the joints!

#     def transformation_matrix(self, a, alpha, d, q):
#         ca = math.cos(alpha)
#         sa = math.sin(alpha)
#         cq = math.cos(q)
#         sq = math.sin(q)

#         T = [[cq, -sq, 0, a],
#             [ca * sq, ca * cq, -sa, -d * sa],
#             [sa * sq, cq * sa, ca, d * ca],
#             [0, 0, 0, 1]]
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
#         num_joints = len(self.dh_parameters) - 1  # Exclude the extra for the flange
#         J = np.zeros((6, num_joints))

#         # Calculate transforms from the base frame to each joint
#         T_all = [np.eye(4)]
#         for i in range(num_joints):
#             a, alpha, d, theta = self.dh_parameters[i]
#             T = self.transformation_matrix(a, alpha, d, theta)
#             T_all.append(np.dot(T_all[-1], T))

#         # End-effector position
#         T_0_end = T_all[-1]
#         p_end = T_0_end[0:3, 3]

#         # Build Jacobian matrix
#         for i in range(num_joints):
#             T_0_i = T_all[i]
#             z_i = T_0_i[0:3, 2]
#             p_i = T_0_i[0:3, 3]

#             # Linear velocity part
#             J[0:3, i] = np.cross(z_i, (p_end - p_i))

#             # Angular velocity part
#             J[3:6, i] = z_i

#         # Normalize the Jacobian
#         norm_J = np.linalg.norm(J, axis=0)
#         norm_J[norm_J == 0] = 1  # Avoid division by zero
#         J_normalized = J / norm_J

#         return J_normalized
 
# if __name__ == "__main__":
#     # Define the joint angles
#     joint_angles = [0, 0, 0, -1, 0, 0, 0]

#     kinematics = Kinematics(joint_angles)

#     position = kinematics.forward_kinematics()
#     print(f"The end effector position is: {position}")

#     jacobian_matrix = kinematics.jacobian()
#     print("Jacobian Matrix:")
#     print(jacobian_matrix)

from sympy import symbols, init_printing, Matrix, eye, sin, cos, pi
init_printing(use_unicode=True)
import numpy as np
from sympy import lambdify


class Kinematics:
    def __init__(self, joint_angles):

        self.joint_angles = joint_angles
            
        # create joint angles as symbols
        q1, q2, q3, q4, q5, q6, q7 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6 theta_7')
        joint_angles = [q1, q2, q3, q4, q5, q6, q7]

        # construct symbolic direct kinematics  from Craig's DH parameters
        # see https://frankaemika.github.io/docs/control_parameters.html
        dh_craig = [
            {'a':  0,      'd': 0.333, 'alpha':  0,  },
            {'a':  0,      'd': 0,     'alpha': -pi/2},
            {'a':  0,      'd': 0.316, 'alpha':  pi/2},
            {'a':  0.0825, 'd': 0,     'alpha':  pi/2},
            {'a': -0.0825, 'd': 0.384, 'alpha': -pi/2},
            {'a':  0,      'd': 0,     'alpha':  pi/2},
            {'a':  0.088,  'd': 0.195, 'alpha':  pi/2},
        ]

        DK = eye(4)
        for i, (p, q) in enumerate(zip(reversed(dh_craig), reversed(joint_angles))):
            d = p['d']
            a = p['a']
            alpha = p['alpha']
            ca = cos(alpha)
            sa = sin(alpha)
            cq = cos(q)
            sq = sin(q)
            transform = Matrix(
                [
                    [cq, -sq, 0, a],
                    [ca * sq, ca * cq, -sa, -d * sa],
                    [sa * sq, cq * sa, ca, d * ca],
                    [0, 0, 0, 1],
                ]
            )

            DK = transform @ DK
        A = DK[0:3, 0:4]  # crop last row
        A = A.transpose().reshape(12,1)  # reshape to column vector A = [a11, a21, a31, ..., a34]

        Q = Matrix(joint_angles)
        J = A.jacobian(Q)  # compute Jacobian symbolically

        self.A_lamb = lambdify((q1, q2, q3, q4, q5, q6, q7), A, 'numpy')
        self.J_lamb = lambdify((q1, q2, q3, q4, q5, q6, q7), J, 'numpy')

        self.joint_limits = [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973)
        ]
        self.max_joint_speed = np.array([
            [2.1750],
            [2.1750],
            [2.1750],
            [2.1750],
            [2.6100],
            [2.6100],
            [2.6100]
            ])
        self.inital_pose = np.array([l+(u-l)/2 for l, u in self.joint_limits], dtype=np.float64)
        

    def forward_kinematics(self, xyz=False):
        q = self.joint_angles
        A = self.A_lamb(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        if xyz:
            return A.flatten()[-3:]
        else:
            return A

    def jacobian(self):
        q = self.joint_angles
        J = self.J_lamb(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        J = J/np.linalg.norm(J)
        return J