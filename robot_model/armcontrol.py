from kinematics import Kinematics

class ArmControl:
    def __init__(self, kp, ki, kd):
        """
        Initialize the ArmControl class with PID constants.

        :param kp: Proportional gain.
        :param ki: Integral gain.
        :param kd: Derivative gain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral_error = 0

    def current_position(self, joint_angles):
        """
        Compute the current position of the end-effector.

        :param joint_angles: List of joint angles.
        :return: Current position of the end-effector.
        """
        kinematics = Kinematics(joint_angles)
        position = kinematics.forward_kinematics()
        return position
    
    def PID(self, target_position, current_position):
        """
        Compute the control action using PID algorithm.

        :param target_position: The desired position of the end-effector.
        :param current_position: The current position of the end-effector.
        :return: Control action.
        """
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
        """
        Compute the control action using PID algorithm.

        :param joint_angles: List of joint angles.
        :param target_position: The desired position of the end-effector.
        :return: Control action.
        """
        current_position = self.current_position(joint_angles)
        control_action = self.PID(target_position, current_position)
        return control_action