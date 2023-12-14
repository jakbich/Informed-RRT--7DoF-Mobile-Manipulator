import numpy as np
from sympy import symbols, Matrix, eye, sin, cos, pi, lambdify
# from numba import jit

class Kinematics:
    def __init__(self):
        # Initialize joint angle symbols
        self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6 theta_7')
        self.joint_angles = [self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7]

        # Define DH parameters and construct direct kinematics
        self.DK = self.direct_kinematics()

        # Flatten to a column vector
        A = self.DK[0:3, 0:4]
        A = A.transpose().reshape(12, 1)

        # Compute Jacobian symbolically
        J = A.jacobian(Matrix(self.joint_angles))

        # Convert to numerical functions using lambdify and jit for speed
        self.A_lamb = lambdify(self.joint_angles, A, 'numpy')
        self.J_lamb = lambdify(self.joint_angles, J, 'numpy')

    def direct_kinematics(self):
        # DH parameters for the Panda robot
        dh_craig = [
            {'a':  0,      'd': 0.333, 'alpha':  0},
            {'a':  0,      'd': 0,     'alpha': -pi/2},
            {'a':  0,      'd': 0.316, 'alpha':  pi/2},
            {'a':  0.0825, 'd': 0,     'alpha':  pi/2},
            {'a': -0.0825, 'd': 0.384, 'alpha': -pi/2},
            {'a':  0,      'd': 0,     'alpha':  pi/2},
            {'a':  0.088,  'd': 0.107, 'alpha':  pi/2},
        ]

        DK = eye(4)

        for p, q in zip(reversed(dh_craig), reversed(self.joint_angles)):
            d = p['d']
            a = p['a']
            alpha = p['alpha']

            ca = cos(alpha)
            sa = sin(alpha)
            cq = cos(q)
            sq = sin(q)

            transform = Matrix([
                [cq, -sq, 0, a],
                [ca * sq, ca * cq, -sa, -d * sa],
                [sa * sq, cq * sa, ca, d * ca],
                [0, 0, 0, 1],
            ])

            DK = transform @ DK
        
        return DK

    def get_matrices(self, joint_values):
        A = self.A_lamb(*joint_values)
        J = self.J_lamb(*joint_values)
        return A, J


# Main function
if __name__ == "__main__":
    robot = Kinematics()

    # Example joint angles (in radians)
    joint_angles = [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0]

    A_matrix, J_matrix = robot.get_matrices(joint_angles)
    position = A_matrix.flat[-3:]
    # print("A Matrix:\n", A_matrix)
    print("Position:", position)

    print("Jacobian Matrix:\n", J_matrix)

    print("A Matrix Shape:", A_matrix.shape)
    print("Jacobian Matrix Shape:", J_matrix.shape)