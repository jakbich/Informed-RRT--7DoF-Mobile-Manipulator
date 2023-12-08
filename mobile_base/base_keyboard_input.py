import gymnasium as gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import numpy as np
from pynput.keyboard import Key, Listener
import threading

from urdfenvs.urdf_common.urdf_env import UrdfEnv

# Global variable to store keyboard input
keyboard_input = np.array([0.0, 0.0, 0.0])

def on_key_press(key):
    global keyboard_input
    if key == Key.left:
        keyboard_input[0] = 1
    elif key == Key.right:
        keyboard_input[0] = -1
    elif key == Key.up:
        keyboard_input[1] = -1
    elif key == Key.down:
        keyboard_input[1] = 1

def on_key_release(key):
    global keyboard_input
    if key == Key.left or key == Key.right:
        keyboard_input[0] = 0.0
    elif key == Key.up or key == Key.down:
        keyboard_input[1] = 0.0

def keyboard_listener():
    with Listener(on_press=on_key_press, on_release=on_key_release) as listener:
        listener.join()

def run_point_robot(n_steps=1000, render=False, goal=False, obstacles=True):
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    
    pos0 = np.array([0., 0., 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    ob = env.reset(pos=pos0, vel=vel0)
    print(f"Initial observation : {ob}")
    
    if goal:
        env.add_goal(splineGoal)
    
    history = []
    env.reconfigure_camera(2.0, 0.0, -90.01, (0, 0, 0))
    
    # Start a thread to listen for keyboard input
    keyboard_thread = threading.Thread(target=keyboard_listener)
    keyboard_thread.start()
    
    for _ in range(n_steps):
        action = keyboard_input.copy()  # Use the global keyboard input
        ob, _, terminated, _, info  = env.step(action)
        if terminated:
            print(info)
            break
        history.append(ob)
    
    # Stop the keyboard listener thread
    keyboard_thread.join()
    
    env.close()
    return history

if __name__ == "__main__":
    run_point_robot(render=True)
