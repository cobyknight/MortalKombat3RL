from gym import Env
from gym.spaces import Box, MultiBinary
import numpy as np
import cv2
import retro

class MortalKombat(Env):
    def __init__(self):
        super().__init__()
        
        # Define observation and action spaces
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)  # Multi-binary action space
        
        # Initialize Retro environment for Mortal Kombat
        self.game = retro.make(game='MortalKombat3-Genesis', use_restricted_actions=retro.Actions.FILTERED)
        
        # Initialize variables for computing rewards
        self.prev_info = {
            'enemy_health': 120,
            'health': 120
        }

    def compute_reward(self, info, prev_info):
        """Compute reward based on changes in game state."""
        # Initialize reward
        reward = 0
        
        # Reward when enemy's health decreases
        if info['enemy_health'] < prev_info['enemy_health']:
            reward += 15

        # Penalty when player's health decreases
        if info['health'] < prev_info['health']:
            reward -= 10
        
        '''    
        # Reward for hitting combos    
        if self.newi >= 2 & self.newi > self.previ:
            reward += 10
            self.previ == self.newi
         
        # Reward for matches won    
        if info["matches_won"] > self.x:
            reward += 100
            self.x += 1
        
        # Penalty for matches lost
        if info["enemy_matches_won"] > self.i:
            #print(info["enemy_matches_won"])
            reward -= 80
            self.i += 1 
        '''
        
        return reward

    def step(self, action):
        """Take a step in the environment."""
        obs, _, done, info = self.game.step(action)
        obs = self.preprocess(obs)  # Preprocess observation
        
        # Calculate reward using the compute_reward method
        reward = self.compute_reward(info, self.prev_info)
        self.prev_info = info  # Update previous info
        
        return obs, reward, done, info

    def render(self, *args, **kwargs): 
        """Render the game environment."""
        self.game.render()
    
    def reset(self):
        """Reset the environment."""
        obs = self.game.reset()  # Reset the game
        obs = self.preprocess(obs)  # Preprocess observation
        self.prev_info = {
            'enemy_health': 120,
            'health': 120
        }  # Reset previous info
        
        return obs
    
    def preprocess(self, observation): 
        """Preprocess observation."""
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)  # Resize
        state = np.reshape(resize, (84,84,1))  # Reshape
        return state
    
    def close(self): 
        """Close the environment."""
        self.game.close()

def create_env(LOG_DIR):
    """Create the environment."""
    env = MortalKombat()  # Create Mortal Kombat environment
    return env