import numpy as np

class ArmControl:
    """
    PID for arm to follow path
    """

    def __init__(self, arm_model, kp=2.0, ki=0, kd=0):
        self.arm_model = arm_model
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.errors = [0]
        self.integral_error = 0.0

    def PID(self, goal, joint_positions, endpoint_orientation=False):

        A, J = self.arm_model.get_matrices(joint_positions)

        state = A

        if endpoint_orientation:
            goal_state = np.vstack([state[:9], goal.reshape(-1, 1)])
        else:
            orientation = np.array([[1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1]])
            goal_state = np.vstack([orientation.reshape(-1, 1), goal.reshape(-1, 1)])

        error = goal_state - state
        derivative_error = error - self.errors[-1]
        self.integral_error += error

        endpoint_vel = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error

        joint_vel = np.linalg.pinv(J) @ endpoint_vel
        joint_vel = np.clip(joint_vel, -self.arm_model.max_joint_speed, self.arm_model.max_joint_speed)

        return joint_vel.flatten()