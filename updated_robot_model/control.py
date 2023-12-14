from kinematics import Kinematics
import numpy as np
from simple_pid import PID

class ArmControl:
    def __init__(self):
        # Initialize PID controller for each degree of freedom in task space
        self.pid_x = PID(1, 0.1, 0.05, setpoint=0)
        self.pid_y = PID(1, 0.1, 0.05, setpoint=0)
        self.pid_z = PID(1, 0.1, 0.05, setpoint=0)

        # Initialize Kinematics
        self.kinematics = Kinematics()

    def control_action(self, joint_angles, target_position):
        # Get current position from forward kinematics
        A = self.kinematics.A_lamb(*joint_angles.flatten())
        current_position = A[:3, -1]  # Extract position from transformation matrix

        # Calculate error
        error = target_position - current_position

        # Apply PID control
        control = np.array([self.pid_x(error[0]), self.pid_y(error[1]), self.pid_z(error[2])])

        return control

    def task_space_to_joint_space(self, joint_angles, target_position):
        control_action = self.control_action(joint_angles, target_position)

        # Calculate Jacobian
        J = self.kinematics.J_lamb(*joint_angles.flatten())

        # Extract the relevant portion of the Jacobian (for X, Y, Z control)
        J_xyz = J[:3, :]  # Assuming the first 3 rows correspond to X, Y, Z translation

        # Convert task space control action to joint space
        joint_action = np.linalg.pinv(J_xyz) @ control_action

        # Apply speed limits
        speed_limits = self.kinematics.speed_limits
        for idx, limit in enumerate(speed_limits):
            joint_action[idx] = np.clip(joint_action[idx], -limit[0], limit[0])

        return joint_action
