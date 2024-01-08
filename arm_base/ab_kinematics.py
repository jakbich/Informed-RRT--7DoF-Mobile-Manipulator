#code based on https://gist.github.com/mlaves/a60cbc5541bd6c9974358fbaad9e4c51 (thank you very much)

# Starting position of the end effector: 5.80310599e-01, 6.08140775e-07, 6.89718851e-01

from sympy import symbols, cos, sin, pi, Matrix, eye, lambdify
import numpy as np

class Kinematics:
    def __init__(self):
        # Define joint angles as symbols
        self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6 theta_7')
        self.joint_angles = [self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7]

        # Craig's DH parameters
        self.dh_craig = [
            {'a':  0,      'd': 0.333, 'alpha':  0    },
            {'a':  0,      'd': 0,     'alpha': -pi/2 },
            {'a':  0,      'd': 0.316, 'alpha':  pi/2 },
            {'a':  0.0825, 'd': 0,     'alpha':  pi/2 },
            {'a': -0.0825, 'd': 0.384, 'alpha': -pi/2 },
            {'a':  0,      'd': 0,     'alpha':  pi/2 },
            {'a':  0.088,  'd': 0.24, 'alpha':  pi/2 },
        ]

        self.speed = np.array([
            [2.1750],
            [2.1750],
            [2.1750],
            [2.1750],
            [2.6100],
            [2.6100],
            [2.6100]
            ])

        # Compute direct kinematics matrix
        DK = eye(4)
        for p, q in zip(reversed(self.dh_craig), reversed(self.joint_angles)):
            d, a, alpha = p['d'], p['a'], p['alpha']
            ca, sa = cos(alpha), sin(alpha)
            cq, sq = cos(q), sin(q)

            transform = Matrix([
                [cq, -sq, 0, a],
                [ca * sq, ca * cq, -sa, -d * sa],
                [sa * sq, cq * sa, ca, d * ca],
                [0, 0, 0, 1],
            ])

            DK = transform @ DK

        # Compute A matrix and Jacobian
        A = DK[0:3, 0:4].transpose().reshape(12, 1)
        J = A.jacobian(Matrix(self.joint_angles))

        # Lambdify the matrices for faster computation
        self.A_lamb = lambdify(self.joint_angles, A, 'numpy')
        self.J_lamb = lambdify(self.joint_angles, J, 'numpy')
    
    def matrices(self, joint_positions):
        A = self.A_lamb(*joint_positions)
        J = self.J_lamb(*joint_positions)
        J = J/np.linalg.norm(J)
        position = A.flatten()[-3:]
        return position, A, J

    def transform_world_to_arm(self, rotation_angle_z=0, base_position=np.array([0,0])):
        # Create translation matrix
        translation_x = -0.19 + base_position[0]
        translation_y = 0 + base_position[1]
        translation_z = 0.64
        translation_matrix = np.array([
            [1, 0, 0, translation_x],
            [0, 1, 0, translation_y],
            [0, 0, 1, translation_z],
            [0, 0, 0, 1]
        ])

        # Rotation around Z-axis
        cos_a = np.cos(rotation_angle_z)
        sin_a = np.sin(rotation_angle_z)
        rotation_matrix_z = np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Combine translation and rotation
        transformation_matrix = np.dot(rotation_matrix_z, translation_matrix)
        transformation_matrix = np.linalg.inv(transformation_matrix)

        return transformation_matrix
        
    

        # def base_to_arm(self, target_position, base_position=np.array([0,0]), theta=0):
        #     rotation_angle_z = theta
        #     translation_x = -0.19
        #     translation_y = 0
        #     translation_z = 0.64
        #     transformation_matrix = np.identity(4)

        #     cos_a = np.cos(rotation_angle_z)
        #     sin_a = np.sin(rotation_angle_z)
        #     Rz = np.array([[cos_a, -sin_a, 0, 0],
        #                    [sin_a, cos_a, 0, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]])

        #     # Adjust translation based on the new origin
        #     transformation_matrix[0, 3] = translation_x + base_position[0]
        #     transformation_matrix[1, 3] = translation_y + base_position[1]
        #     transformation_matrix[2, 3] = translation_z 

        #     # Combine rotation and translation
        #     transformation_matrix = np.dot(Rz, transformation_matrix)   
        #     # transformation_matrix = np.linalg.inv(transformation_matrix)

        #     # Convert the target_position to a 4x1 vector
        #     target_vector = np.append(target_position, 1)

        #     # Apply the transformation
        #     target_position_transformed = np.dot(transformation_matrix, target_vector)

        #     return target_position_transformed[:-1]
        
