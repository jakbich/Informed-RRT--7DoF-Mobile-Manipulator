from one_joint_kinematics import Kinematics
import numpy as np
from simple_pid import PID


class ArmControl:
    def __init__(self):
        
        # Initialize PID controller
        self.pid = PID(1, 0.1, 0.05, setpoint=0)

    def current_position(self, joint_angles):
        kinematics = Kinematics(joint_angles)
        position = kinematics.forward_kinematics()
        jacobian = kinematics.jacobian()
        return jacobian, position
    
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

    # Assume action is a list with at least 4 elements
    def control_action(self, joint_angles, target_position):

        actual_position = Kinematics(joint_angles).forward_kinematics()

        # Use kinematics function to calculate error
        error = target_position - actual_position
        
        # Use PID controller to calculate control output
        control = self.pid(error)
        
        return control