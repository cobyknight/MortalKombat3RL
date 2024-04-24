import os
import time

from environment import create_env
from callback import Callbacks
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
# Import optuna for HPO
import optuna
# Import PPO for algos
from stable_baselines3 import PPO
# Evaluate Policy
from stable_baselines3.common.evaluation import evaluate_policy
# Import wrappers
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


LOG_DIR = './logs/'
CHECKPOINT_DIR = './train_nodelta/'
OPT_DIR = './nodelta/'

env = create_env(LOG_DIR=LOG_DIR)
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')
model_params = {'n_steps': 5824, 'gamma': 0.8787592152924, 'learning_rate': 3.52316500494815e-07, 'clip_range': 0.1487683904424553, 'gae_lambda': 0.9474456981149016}
#model_params = {'n_steps': 8960, 'gamma': 0.906, 'learning_rate': 2e-03, 'clip_range': 0.369, 'gae_lambda': 0.891}
# model_params = study.best_params
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
model.load(os.path.join(OPT_DIR, 'trial_6_best_model.zip'))
#model.load('./train_nodelta_backup/best_model_5460000.zip')
callback = Callbacks(check_freq=10000, save_path=CHECKPOINT_DIR)
model.learn(total_timesteps=5000000, callback=callback)
env.close()