from kinematics import Kinematics
import numpy as np

class ArmControl:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral_error = 0

    def current_position(self, joint_angles):
        kinematics = Kinematics(joint_angles)
        position = kinematics.forward_kinematics()
        jacobian = kinematics.jacobian()
        return jacobian, position
    
    def PID(self, target_position, current_position):
        # Calculate error
        error = target_position - current_position

        # Proportional term
        P_out = self.kp * error

        # Integral term
        self.integral_error += error
        I_out = self.ki * self.integral_error

        # Derivative term
        derivative_error = error - self.previous_error
        D_out = self.kd * derivative_error

        # Total output
        total_output = P_out + I_out + D_out

        # Update previous error
        self.previous_error = error

        return total_output
    
    def control_action(self, joint_angles, target_position):
        jacobian, current_position = self.current_position(joint_angles)
        control_action = self.PID(target_position, current_position)
        return jacobian, control_action
    
    def task_space_to_joint_space(self, joint_angles, target_position):
        jacobian, control_action = self.control_action(joint_angles, target_position)
        joint_action = np.matmul(np.linalg.pinv(jacobian), control_action)
        return joint_action