from sympy import symbols, cos, sin, pi, Matrix, eye, lambdify
import numpy as np
from numba import jit

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
            {'a':  0.088,  'd': 0.107, 'alpha':  pi/2 },
        ]

        self.speed_limits = [
                                (-2.8973, 2.8973),
                                (-1.7628, 1.7628),
                                (-2.8973, 2.8973),
                                (-3.0718, -0.0698),
                                (-2.8973, 2.8973),
                                (-0.0175, 3.7525),
                                (-2.8973, 2.8973)
                            ]

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
    
    def FK(self, joint_positions, xyz=False):
        q = joint_positions
        A = self.A_lamb(*q)
        return A.flatten()[-3:], A
    
    # def jacobian(self, joint_positions):
    #     q = joint_positions
    #     J = self.J_lamb(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
    #     return J
    
    def jacobian(self, joint_positions):
        q = joint_positions
        J = self.J_lamb(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        J = J/np.linalg.norm(J)
        return J