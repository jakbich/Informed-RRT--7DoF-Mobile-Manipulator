# https://frankaemika.github.io/docs/control_parameters.html

import numpy as np
import math
import warnings

class Kinematics:
    def __init__(self, joint_angles):
        # Throw error when wrong amount of angles is provided
        if len(joint_angles) != 7:
            raise ValueError("The wrong amount of angles was provided.", len(joint_angles), "instead of 7.")
        
        self.joint_angles = joint_angles

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
            [0.088, np.pi/2, 0, joint_angles[6]]]
            # [0, 0, 0.107, 0]]                    #extra for the flange, not for the joints!

    def transformation_matrix(self, a, alpha, d, q):
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        cq = math.cos(q)
        sq = math.sin(q)

        T = [[cq, -sq, 0, a],
            [ca * sq, ca * cq, -sa, -d * sa],
            [sa * sq, cq * sa, ca, d * ca],
            [0, 0, 0, 1]]
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

    def jacobian2(self):
        numjoints = len(self.dh_parameters) 
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
    
    def get_tf_mat(self, i, dh):
        a = dh[i][0]
        d = dh[i][1]
        alpha = dh[i][2]
        theta = dh[i][3]
        q = theta

        return np.array([[np.cos(q), -np.sin(q), 0, a],
                        [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                        [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                        [0, 0, 0, 1]])
    
    def jacobian(self):
        dh_params = np.array([[0, 0.333, 0, self.joint_angles[0]],
            [0, 0, -np.pi / 2, self.joint_angles[1]],
            [0, 0.316, np.pi / 2, self.joint_angles[2]],
            [0.0825, 0, np.pi / 2, self.joint_angles[3]],
            [-0.0825, 0.384, -np.pi / 2, self.joint_angles[4]],
            [0, 0, np.pi / 2, self.joint_angles[5]],
            [0.088, 0, np.pi / 2, self.joint_angles[6]],
            [0, 0.107, 0, 0],
            [0, 0, 0, -np.pi / 4],
            [0.0, 0.1034, 0, 0]], dtype=np.float64)

        T_EE = np.identity(4)
        for i in range(7 + 3):
            T_EE = T_EE @ self.get_tf_mat(i, dh_params)

        J = np.zeros((6, 10))
        T = np.identity(4)
        for i in range(7 + 3):
            T = T @ self.get_tf_mat(i, dh_params)

            p = T_EE[:3, 3] - T[:3, 3]
            z = T[:3, 2]

            J[:3, i] = np.cross(z, p)
            J[3:, i] = z

        return J[:, :7]

    def debug_transformations(self):
        T = np.eye(4)
        transformation_matrices = []

        for i, parameters in enumerate(self.dh_parameters):
            a = parameters[0]
            alpha = parameters[1]
            d = parameters[2]
            theta = parameters[3]

            Ti = self.transformation_matrix(a, alpha, d, theta)
            T = np.dot(T, Ti)
            transformation_matrices.append(T)

            x, y, z = T[0, 3], T[1, 3], T[2, 3]
            print(f"Joint {i + 1}:")
            print("Transformation Matrix:\n", T)
            print(f"Position (x, y, z): ({x}, {y}, {z})\n")


from sympy import symbols, Matrix, sin, cos, lambdify
import numpy as np

class Kinematics2:
    def __init__(self, joint_angles):
        if len(joint_angles) != 7:
            raise ValueError("Incorrect number of joint angles provided. Expected 7.")

        self.joint_angles = joint_angles

        # Create symbolic joint angles
        self.q = symbols('q1:8')  # q1, q2, ..., q7

        # Define DH parameters using symbolic joint angles
        self.dh_parameters = [
            [0, 0, 0.333, self.q[0]],
            [0, -np.pi/2, 0, self.q[1]],
            [0, np.pi/2, 0.316, self.q[2]],
            [0.0825, np.pi/2, 0, self.q[3]],
            [-0.0825, -np.pi/2, 0.384, self.q[4]],
            [0, np.pi/2, 0, self.q[5]],
            [0.088, np.pi/2, 0, self.q[6]]
        ]

        self.init_symbolic_computations()

    def init_symbolic_computations(self):
        T = Matrix.eye(4)
        for parameters in self.dh_parameters:
            a, alpha, d, theta = parameters
            Ti = self.symbolic_transformation_matrix(a, alpha, d, theta)
            T = T * Ti

        # Extract end-effector position (x, y, z)
        self.end_effector_pos = T[:3, 3]

        # Compute Jacobian
        self.J = self.end_effector_pos.jacobian(self.q)

        # Create lambda functions for numerical computation
        self.J_numeric = lambdify(self.q, self.J, 'numpy')

    def symbolic_transformation_matrix(self, a, alpha, d, q):
        ca = cos(alpha)
        sa = sin(alpha)
        cq = cos(q)
        sq = sin(q)

        return Matrix([
            [cq, -sq, 0, a],
            [ca * sq, ca * cq, -sa, -d * sa],
            [sa * sq, cq * sa, ca, d * ca],
            [0, 0, 0, 1]
        ])

    def jacobian(self, joint_angles):
        return self.J_numeric(*joint_angles)

    
if __name__ == "__main__":

    joint_angles = [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0]

    kinematics = Kinematics(joint_angles)
    # kinematics.debug_transformations()

    jacobian = kinematics.jacobian()
    jacobian2 = kinematics.jacobian2()

    print('jacobian', jacobian) 
    print('jacobian2', jacobian2)


    kinematics2 = Kinematics2(joint_angles)
    jacobian3 = kinematics2.jacobian(joint_angles)
    print("jacobian3", jacobian3)
















    # angle_limits = [
    #     (-2.8973, 2.8973),
    #     (-1.7628, 1.7628),
    #     (-2.8973, 2.8973),
    #     (-3.0718, -0.0698),
    #     (-2.8973, 2.8973),
    #     (-0.0175, 3.7525),
    #     (-2.8973, 2.8973),
    # ]

    # inital_pose = np.array([min+(max-min)/2 for min, max in angle_limits], dtype=np.float64)







    # # ----------------------------------------------------------------------BASE TRANSFORMATION    
    # # Define the translation matrix for moving to (0, 0.15, 0.63)
    # T = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0.15],
    #     [0, 0, 1, 0.63],
    #     [0, 0, 0, 1]
    # ])

    # # Define the rotation matrix for -90 degrees around Z-axis
    # theta = np.radians(-90)  # Convert -90 degrees to radians
    # R = np.array([
    #     [np.cos(theta), -np.sin(theta), 0, 0],
    #     [np.sin(theta), np.cos(theta), 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ])

    # # Compute the combined transformation matrix for Translation first, then Rotation
    # M = np.dot(R, T)

    # # Print the transformation matrix
    # print("Transformation Matrix:\n", M)

    # # Define the point as a 4D vector (x, y, z, 1)
    # P = np.array([0.5506552752335201, -1.6886238525761472e-17, 0.7572267968606404, 1])  # Replace x, y, z with the coordinates of your point

    # # Apply the transformation matrix to the point
    # P_transformed = np.dot(M, P)

    # # Print the transformed point
    # print("Transformed Point:", P_transformed)




    

