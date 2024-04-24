import retro
from gym import Env
from gym.spaces import Discrete, Box, MultiBinary
import numpy as np
import cv2
import time
from environment import create_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# Load pre-trained PPO model
model = PPO.load('./train_nodelta/new_best_model_2770000.zip')

# Define directories
LOG_DIR = './logs/'
CHECKPOINT_DIR = './train_nodelta/'
OPT_DIR = './nodelta/'

# Create custom environment
env = create_env(LOG_DIR=LOG_DIR)

# Monitor the environment
env = Monitor(env)

# Vectorize environment
env = DummyVecEnv([lambda: env])

# Stack frames
env = VecFrameStack(env, 4, channels_order='last')

# Run episodes
for episode in range(6): 
    obs = env.reset()  # Reset the environment for a new episode
    done = False
    total_reward = 0
    
    # Loop until episode is done
    while not done: 
        action, _ = model.predict(obs)  # Get action from the model
        obs, reward, done, info = env.step(action)  # Take action in the environment
        env.render()  # Render the environment
        time.sleep(.005)  # Delay for visualization
        total_reward += reward  # Accumulate total reward
        
    print('Total Reward for episode {} is {}'.format(episode, total_reward))  # Print total reward for the episode
    time.sleep(2)  # Delay before starting the next episode
