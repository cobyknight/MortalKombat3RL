import os
import time
import optuna
from environment import create_env
from callback import Callbacks
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Directories
LOG_DIR = 'MORTALKOMBATAIFIX\logs'
OPT_DIR = 'MORTALKOMBATAIFIX/opt/'

def optimize_ppo(trial):
    """ Function to optimize PPO hyperparameters """
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),  # Number of steps per update
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),  # Discount factor
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),  # Learning rate
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),  # Clip range for policy gradient
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, .99)  # Lambda for Generalized Advantage Estimation
    }

def optimize_agent(trial):
    """ Function to optimize the PPO agent """
    model_params = optimize_ppo(trial)  # Get optimized PPO hyperparameters
    env = create_env(LOG_DIR=LOG_DIR)  # Create environment
    env = Monitor(env, LOG_DIR)  # Monitor environment
    env = DummyVecEnv([lambda: env])  # Vectorize environment
    env = VecFrameStack(env, 4, channels_order='last')  # Stack frames
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)  # Create PPO model
    model.learn(total_timesteps=30000)  # Train the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)  # Evaluate mean reward
    env.close()  # Close the environment

    # Save the best model for the current trial
    SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
    model.save(SAVE_PATH)
    return mean_reward

# Create an optimization study
study = optuna.create_study(direction='maximize')
# Optimize the agent
study.optimize(optimize_agent, n_trials=10, n_jobs=1)

# Get the best trial and print its details
best_trial = study.best_trial
print("Best trial:")
print("  Value: ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

# Load the best model from the best trial
best_model_path = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(best_trial.number))
best_model = PPO.load(best_model_path)
print("Best model loaded from:", best_model_path)

# Print the best parameters found during optimization
best_params = study.best_params
print("Best parameters:")
for key, value in best_params.items():
    print("  {}: {}".format(key, value))
