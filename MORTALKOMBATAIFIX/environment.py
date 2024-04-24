from gym import Env
from gym.spaces import Box, MultiBinary
import numpy as np
import cv2
import retro
class MortalKombat(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game='MortalKombat3-Genesis', use_restricted_actions=retro.Actions.FILTERED)
        self.i = 0
        self.x = 0
        self.previ = 0
        self.newi = 0
        self.prev_info = {
            'enemy_health': 120,
            'health': 120
        }
        
    def compute_reward(self, info, prev_info):
        # Initialize reward
        reward = 0

        
        enemywin = info["enemy_matches_won"]

        # Reward when enemy's health decreases
        if info['enemy_health'] < prev_info['enemy_health']:
            reward += 15
            self.newi + 1

        # Penalty when player's health decreases
        if info['health'] < prev_info['health']:
            reward -= 10
            self.newi = 0
            self.previ = 0
        
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
        obs, _, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        frame_delta = obs
        
        # Calculate reward using the compute_reward method
        reward = self.compute_reward(info, self.prev_info)
        self.prev_info = info

        return obs, reward, done, info
    
    def render(self, *args, **kwargs): 
        self.game.render()
    
    def reset(self):
        self.previous_frame = np.zeros(self.game.observation_space.shape)
        
        # Frame delta
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        
        self.health = 120
        self.enemy_health = 120
        self.matches_won = 0
        self.enemy_matches_won = 0

        return obs
    
    def preprocess(self, observation): 
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (84,84,1))
        return state
    
    def close(self): 
        self.game.close()
        
def create_env(LOG_DIR):
    env = MortalKombat()

    return env

