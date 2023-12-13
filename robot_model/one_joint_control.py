from one_joint_kinematics import Kinematics
import numpy as np
from simple_pid import PID


class ArmControl:
    def __init__(self, kp=0.2, ki=0, kd= 0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral_error = 0
        
        # Initialize PID controller
        self.pid = PID(1, 0.1, 0.05, setpoint=0)
        self.pid_x = PID(1, 0.1, 0.05, setpoint=0)
        self.pid_y = PID(1, 0.1, 0.05, setpoint=0)
        self.pid_z = PID(1, 0.1, 0.05, setpoint=0)

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
        extended_control_action = np.concatenate((control_action, np.zeros(3)))
        joint_action = np.linalg.pinv(jacobian) @ extended_control_action
        speed_limits = Kinematics(joint_angles).speed_limits
        for idx, limit in enumerate(speed_limits):
            joint_action[idx] = np.clip(joint_action[idx], -limit[0], limit[0])
        return joint_action

    # # Assume action is a list with at least 4 elements
    # def control_action(self, joint_angles, target_position):

    #     actual_position = Kinematics(joint_angles).forward_kinematics()

    #     # Use kinematics function to calculate error
    #     error = target_position - actual_position
        
    #     # Use PID controller to calculate control output
    #     control = self.pid(error)
        
    #     return control
    
    def control_action(self, joint_angles, target_position):
        # Compute the current position and the Jacobian
        jacobian, current_position = self.current_position(joint_angles)

        # Calculate the position error (desired - current)
        position_error = target_position - current_position[:3]

        # Use a PID controller for each dimension (X, Y, Z)
        control_signal_x = self.pid_x(position_error[0])
        control_signal_y = self.pid_y(position_error[1])
        control_signal_z = self.pid_z(position_error[2])

        # Desired end-effector velocity (consider scaling factors)
        desired_velocity = np.array([control_signal_x, control_signal_y, control_signal_z])


        # Use the pseudoinverse of the Jacobian to calculate joint velocities
        pseudo_inverse_jacobian = np.linalg.pinv(jacobian)
        print('pseudo_inverse_jacobian', pseudo_inverse_jacobian.shape)
        print(jacobian.shape)
        pseudo_inverse_jacobian = np.linalg.pinv(jacobian[:3, :])  # Considering only the first 3 rows for linear motion
        print('pseudo_inverse_jacobian', pseudo_inverse_jacobian.shape)

        # joint_velocities = np.dot(pseudo_inverse_jacobian, desired_velocity)

        joint_velocities = pseudo_inverse_jacobian @ desired_velocity

        # Clip velocities based on joint speed limits
        # joint_velocities = self.clip_joint_velocities(joint_velocities, joint_angles)

        return joint_velocities.flatten()

    # def clip_joint_velocities(self, velocities, joint_angles):
    #     # Retrieve speed limits from Kinematics
    #     kinematics = Kinematics(joint_angles)
    #     speed_limits = kinematics.speed_limits

    #     # Clip velocities based on joint speed limits
    #     for i, speed_limit in enumerate(speed_limits):
    #         velocities[i] = np.clip(velocities[i], -speed_limit[0], speed_limit[0])

    #     return velocities
