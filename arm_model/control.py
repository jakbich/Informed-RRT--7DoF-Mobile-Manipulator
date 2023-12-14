from kinematics import Kinematics
import numpy as np

class ArmControl:
    def __init__(self, kp=2, ki=0, kd=0.2):
        self.kp = kp
        self.kd = kd
        self.errors = [0]
        self.kinematics = Kinematics()

    def control_action(self, joint_angles, target_position):
        _, A, jacobian = self.kinematics.matrices(joint_angles)
        multi_dim_position = np.vstack([np.eye(3).reshape(-1, 1), target_position.reshape(-1, 1)])
        error = multi_dim_position - A
        d_e = error - self.errors[-1]
        task_space_velocity = self.kp * error + self.kd * d_e
        joint_action = np.linalg.pinv(jacobian) @ task_space_velocity
        joint_action = np.clip(joint_action, -self.kinematics.speed, self.kinematics.speed)

        return joint_action