import retro
from gym import Env
from gym.spaces import Discrete, Box, MultiBinary
import numpy as np
import cv2
import time
from environment import create_env
# Import PPO for algos
from stable_baselines3 import PPO
# Evaluate Policy
from stable_baselines3.common.evaluation import evaluate_policy
# Import Wrappers
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

model = PPO.load('./train_nodelta/new_best_model_2770000.zip')

LOG_DIR = './logs/'
CHECKPOINT_DIR = './train_nodelta/'
OPT_DIR = './nodelta/'

env = create_env(LOG_DIR=LOG_DIR)
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')
for episode in range(6): 
    obs = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(.005)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))
    time.sleep(2)
    
    
