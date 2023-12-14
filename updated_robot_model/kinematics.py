# from sympy import symbols, init_printing, Matrix, eye, sin, cos, pi
# init_printing(use_unicode=True)
# import numpy as np
# from sympy import lambdify
# from numba import jit

# # create joint angles as symbols

# q1, q2, q3, q4, q5, q6, q7 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6 theta_7')
# joint_angles = [q1, q2, q3, q4, q5, q6, q7]

# # construct symbolic direct kinematics from Craig's DH parameters
# # see https://frankaemika.github.io/docs/control_parameters.html

# dh_craig = [
#     {'a':  0,      'd': 0.333, 'alpha':  0,  },
#     {'a':  0,      'd': 0,     'alpha': -pi/2},
#     {'a':  0,      'd': 0.316, 'alpha':  pi/2},
#     {'a':  0.0825, 'd': 0,     'alpha':  pi/2},
#     {'a': -0.0825, 'd': 0.384, 'alpha': -pi/2},
#     {'a':  0,      'd': 0,     'alpha':  pi/2},
#     {'a':  0.088,  'd': 0.107, 'alpha':  pi/2},
# ]

# DK = eye(4)

# for i, (p, q) in enumerate(zip(reversed(dh_craig), reversed(joint_angles))):
#     d = p['d']
#     a = p['a']
#     alpha = p['alpha']

#     ca = cos(alpha)
#     sa = sin(alpha)
#     cq = cos(q)
#     sq = sin(q)

#     transform = Matrix(
#         [
#             [cq, -sq, 0, a],
#             [ca * sq, ca * cq, -sa, -d * sa],
#             [sa * sq, cq * sa, ca, d * ca],
#             [0, 0, 0, 1],
#         ]
#     )

#     DK = transform @ DK

# # test direct kinematics

# DK.evalf(subs={
#     'theta_1': 0,
#     'theta_2': 0,
#     'theta_3': 0,
#     'theta_4': 0,
#     'theta_5': 0,
#     'theta_6': 0,
#     'theta_7': 0,
# })

# A = DK[0:3, 0:4]  # crop last row
# A = A.transpose().reshape(12,1)  # reshape to column vector A = [a11, a21, a31, ..., a34]

# Q = Matrix(joint_angles)
# J = A.jacobian(Q)  # compute Jacobian symbolically

# A_lamb = jit(lambdify((q1, q2, q3, q4, q5, q6, q7), A, 'numpy'))
# J_lamb = jit(lambdify((q1, q2, q3, q4, q5, q6, q7), J, 'numpy'))

# @jit
# def incremental_ik(q, A, A_final, step=0.1, atol=1e-4):
#     while True:
#         delta_A = (A_final - A)
#         if np.max(np.abs(delta_A)) <= atol:
#             break
#         J_q = J_lamb(q[0,0], q[1,0], q[2,0], q[3,0], q[4,0], q[5,0], q[6,0])
#         J_q = J_q / np.linalg.norm(J_q)  # normalize Jacobian
        
#         # multiply by step to interpolate between current and target pose
#         delta_q = np.linalg.pinv(J_q) @ (delta_A*step)
        
#         q = q + delta_q
#         A = A_lamb(q[0,0], q[1,0],q[2,0],q[3,0],q[4,0],q[5,0],q[6,0])
#     return q, np.max(np.abs(delta_A))

# # define joint limits for the Panda robot
# limits = [
#     (-2.8973, 2.8973),
#     (-1.7628, 1.7628),
#     (-2.8973, 2.8973),
#     (-3.0718, -0.0698),
#     (-2.8973, 2.8973),
#     (-0.0175, 3.7525),
#     (-2.8973, 2.8973)
# ]

# # create initial pose
# q_init = np.array([l+(u-l)/2 for l, u in limits], dtype=np.float64).reshape(7, 1)
# A_init = A_lamb(*(q_init.flatten()))
# print(A_init.reshape(3, 4, order='F'))

# # generate random final pose within joint limits

# np.random.seed(0)

# q_rand = np.array([np.random.uniform(l, u) for l, u in limits], dtype=np.float64).reshape(7, 1)
# A_final = A_lamb(*(q_rand).flatten())
# print(A_final.reshape(3, 4, order='F'))

# q_final, _ = incremental_ik(q_init, A_init, A_final, atol=1e-6)
# q_final.flatten()

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
        self.DK = self.compute_direct_kinematics()

        # Compute A matrix and Jacobian
        A = self.DK[0:3, 0:4].transpose().reshape(12, 1)
        self.Jacobian = A.jacobian(Matrix(self.joint_angles))

        # Lambdify the matrices for faster computation
        self.A_lamb = jit(lambdify(self.joint_angles, A, 'numpy'))
        self.J_lamb = jit(lambdify(self.joint_angles, self.Jacobian, 'numpy'))

    def compute_direct_kinematics(self):
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
        return DK
    
    def FK(self, joint_positions, xyz=False):
        q = joint_positions
        A = self.A_lamb(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        if xyz:
            return A.flatten()[-3:]
        else:
            return A
        
    def forward_kinematics(self, joint_angles):
        """
        Compute the XYZ position in the task space given the current joint angles.

        :param joint_angles: A list or array of joint angles (in radians).
        :return: A 3-element array representing the XYZ position in the task space.
        """
        # Evaluate the direct kinematics matrix with the current joint angles
        DK_evaluated = self.DK.evalf(subs={self.q1: joint_angles[0], 
                                           self.q2: joint_angles[1], 
                                           self.q3: joint_angles[2], 
                                           self.q4: joint_angles[3], 
                                           self.q5: joint_angles[4], 
                                           self.q6: joint_angles[5], 
                                           self.q7: joint_angles[6]})

        # Extract the XYZ position from the last column of the DK matrix
        xyz_position = Matrix(DK_evaluated)[:3, 3]

        return np.array(xyz_position).astype(np.float64) 

    @jit
    def incremental_ik(self, q, A, A_final, step=0.1, atol=1e-4):
        while True:
            delta_A = A_final - A
            if np.max(np.abs(delta_A)) <= atol:
                break
            J_q = self.J_lamb(*q.flatten())
            J_q /= np.linalg.norm(J_q)

            delta_q = np.linalg.pinv(J_q) @ (delta_A * step)
            q += delta_q
            A = self.A_lamb(*q.flatten())

        return q, np.max(np.abs(delta_A))