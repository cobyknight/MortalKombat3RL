import optuna

from stable_baselines3 import PPO #Algoritmo PPO (Proximal Policy Optimization)

from stable_baselines3.common.evaluation import evaluate_policy
from environment import create_env
from math import floor
import os
    
    
import optuna
# Import PPO for algos
from stable_baselines3 import PPO
# Evaluate Policy
from stable_baselines3.common.evaluation import evaluate_policy
# Import wrappers
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
LOG_DIR = './logs/'
OPT_DIR = './opt/'

def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, .99)
    }
def optimize_agent(trial):
        model_params = optimize_ppo(trial)
        env = create_env(LOG_DIR=LOG_DIR)
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=30000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        env.close()

        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)
        return mean_reward
    
study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=10, n_jobs=1)

best_trial = study.best_trial
print("Best trial:")
print("  Value: ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

best_model_path = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(best_trial.number))
best_model = PPO.load(best_model_path)
print("Best model loaded from:", best_model_path)

# After the optimization process is completed
best_params = study.best_params
print("Best parameters:")
for key, value in best_params.items():
    print("  {}: {}".format(key, value))