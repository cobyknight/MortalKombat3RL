import os
import time
import optuna
from environment import create_env
from callback import Callbacks
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Define directories
LOG_DIR = './logs/'
CHECKPOINT_DIR = './train/'
OPT_DIR = './opt/'

# Create environment
env = create_env(LOG_DIR=LOG_DIR)  # Creating environment with logging directory
env = Monitor(env, LOG_DIR)  # Monitor environment
env = DummyVecEnv([lambda: env])  # Vectorize environment
env = VecFrameStack(env, 4, channels_order='last')  # Stack frames

# Define model parameters
model_params = {'n_steps': 5824,  # Number of steps to run for each batch
                'gamma': 0.8787592152924,  # Discount factor
                'learning_rate': 3.52316500494815e-07,  # Learning rate
                'clip_range': 0.1487683904424553,  # Clip range for clipping gradients
                'gae_lambda': 0.9474456981149016}  # Lambda for Generalized Advantage Estimation

# Initialize PPO model
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)

# Load pre-trained model
model.load(os.path.join(OPT_DIR, 'trial_6_best_model.zip'))

# Define callback for saving checkpoints during training
callback = Callbacks(check_freq=10000, save_path=CHECKPOINT_DIR)

# Train the model
model.learn(total_timesteps=5000000, callback=callback)

# Close the environment
env.close()
