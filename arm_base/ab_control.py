from ab_kinematics import Kinematics
import numpy as np

class ArmControl:
    def __init__(self, kp=1, kd=0.1):
        self.kp = kp
        self.kd = kd
        self.errors = [0]
        self.kinematics = Kinematics()
    
    def control_action(self, joint_angles, target_position):
        _, A, jacobian = self.kinematics.matrices(joint_angles)
        multi_dim_position = np.vstack([-np.eye(3).reshape(-1, 1), target_position.reshape(-1, 1)])
        error = multi_dim_position - A
        error_magnitude = np.linalg.norm(error)

        # print("Error magnitude: ", error_magnitude)

        if error_magnitude < 1:
            joint_action = np.zeros(jacobian.shape[1])
        else:
            d_e = error - self.errors[-1]
            task_space_velocity = self.kp * error + self.kd * d_e
            joint_action = np.linalg.pinv(jacobian) @ task_space_velocity
            joint_action = np.clip(joint_action, -self.kinematics.speed, self.kinematics.speed)

        return joint_action