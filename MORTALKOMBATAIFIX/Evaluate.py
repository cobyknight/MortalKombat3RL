import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from environment import create_env

LOG_DIR = './logs/'
CHECKPOINT_DIR = './train_nodelta/'

# Initialize variables
best_models = []
best_rewards = []

# Load the pre-trained model
model = PPO.load('./train_nodelta/new_best_model_90000')

# Create the environment
env = create_env(LOG_DIR=LOG_DIR)
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# Test models from 10000 to 4900000 checking every 40000 models
for i in range(10000, 4900001, 60000):
    # Load the model
    model_path = os.path.join(CHECKPOINT_DIR, f"new_best_model_{i}")
    model = PPO.load(model_path)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, render=False)

    # Append the results
    best_models.append(model_path)
    best_rewards.append(mean_reward)

# Sort the models by reward in descending order
sorted_models = [model for _, model in sorted(zip(best_rewards, best_models), reverse=True)]

# Print the highest performing models
for i, model_path in enumerate(sorted_models):
    print(f"Model {i+1}: {model_path} - Mean Reward: {best_rewards[i]}")
