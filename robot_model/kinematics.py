import numpy as np

class Kinematics:
    def __init__(self, dh_parameters):
        self.dh_parameters = dh_parameters

    def transformation_matrix(self, a, alpha, d, theta):
        T = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        return T

    def forward_kinematics(self, dh_parameters):

        T = np.eye(4)
        for i, (a, alpha, d, theta) in enumerate(self.dh_params):
            T = np.dot(T, self.dh_matrix(a, alpha, d, q[i]))
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        return x, y, z

    def inverse_kinematics(self, x, y, z):
        q = []
        return q
    
    
q = [0, 0, 0, 0]
dh_parameters = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]    
kinematics = Kinematics(dh_parameters)
print(kinematics.forward_kinematics(q))

x, y, z = 10, 0, 5 
print(kinematics.inverse_kinematics(x, y, z))
    

