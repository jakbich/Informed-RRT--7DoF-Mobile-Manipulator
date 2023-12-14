from korneel_kinematics import Kinematics
import numpy as np

class ArmControl:
    # def __init__(self):
        # Initialize PID controller
        # self.pid = PID(2, 0., 0.0, setpoint=0)

    def __init__(self, kp=1.0, ki=0, kd=0):
        self.arm_model = Kinematics()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.errors = [0]
        self.integral_error = 0.0

        # Initialize Kinematics
        self.kinematics = Kinematics()

    def control_action(self, joint_angles, target_position):
        # Get current position from forward kinematics
        xyz, A = self.kinematics.FK(joint_angles)

        orientation = np.array([[1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1]])

        multi_dim_position = np.vstack([orientation.reshape(-1, 1), target_position.reshape(-1, 1)])

        # Calculate error)
        error = multi_dim_position - A

        jacobian_temp = self.kinematics.jacobian(joint_angles)

        jacobian = np.zeros((jacobian_temp.shape[0], jacobian_temp.shape[1]))

        for i in range(jacobian.shape[0]):  # Loop over rows
            for j in range(jacobian.shape[1]):  # Loop over columns
                jacobian[i, j] = jacobian_temp[i, j]

        # # Apply PID control
        # task_space_velocity = self.pid(error.flatten())
        # joint_action = np.linalg.pinv(jacobian) @ task_space_velocity

        # # Apply speed limits
        # speed_limits = self.kinematics.speed_limits
        # for idx, limit in enumerate(speed_limits):
        #     joint_action[idx] = np.clip(joint_action[idx], -limit[0], limit[0])

        error = multi_dim_position - A
        derivative_error = error - self.errors[-1]
        self.integral_error += error

        J = jacobian

        endpoint_vel = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error

        joint_vel = np.linalg.pinv(J) @ endpoint_vel
        # joint_vel = np.clip(joint_vel, -self.arm_model.max_joint_speed, self.arm_model.max_joint_speed)

        # return joint_vel.flatten()

        return joint_vel.flatten(), error
