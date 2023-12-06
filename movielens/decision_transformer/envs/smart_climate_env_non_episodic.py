import random
import pickle
import numpy as np
import gym
from gym.spaces import Box, Space
from tqdm import tqdm
import time
# from atari.mingpt.utils import set_seed
# set_seed(123)

# This is from a global pool of trajectories

class CustomActionSpace(Space):
    def __init__(self, shape=None, dtype=None):
        super().__init__(shape, dtype)
        actions = np.arange(16, 28.5, 0.5)
        # actions = np.array(['16-18', '18-20', '20-22', '22-24', '24-26', '26-28'])
        self.actions_map = {idx: action for idx, action in enumerate(actions)}
        self.actions = list(self.actions_map.keys())
    def sample(self):
        """Randomly sample an element of this space. Can be 
        uniform or non-uniform sampling based on boundedness of space."""
        return np.random.choice(self.actions)
    
class SmartClimateEnv(gym.Env):
    def __init__(self, test_traj_path, pbar=None):
        with open(test_traj_path, 'rb') as f:
            self.dataset = pickle.load(f)

        super(SmartClimateEnv, self).__init__()
        self.current_step = 0
        self.max_steps = sum(len(traj['observations']) for traj in self.dataset)
        self.action_space = CustomActionSpace(shape=(1, 1))  # You need to define CustomActionSpace
        self.observation_space = Box(low=0, high=1, shape=(self.dataset[0]['observations'].shape[1],), dtype=np.float32)
        self.sampled_idx = None
        self.action = None
        self.reward = None
        self.pbar = pbar
        self.total_steps = 0

    def step(self, action):
        self.action = action
        temperature = self.dataset[self.sampled_idx]['actions'][self.current_step]
        target_temperature = self.action_space.actions_map[temperature]
        predicted_temperature = self.action_space.actions_map[action]
        
        # print(f"target_temperature: {target_temperature} | predicted_temperature: {predicted_temperature}")
        acc = 0
        if predicted_temperature == target_temperature:
            acc = 1
        #     reward = 1
        # else:
        #     reward = 0
        
        # Reward Scheme 4
        # -----------------------------------------------
        # error = abs(target_temperature - predicted_temperature)
        # if error <= 0.5:
        #     reward = 1.00
        # elif error <= 1.0:
        #     reward = 0.90
        # elif error <= 2.0:
        #     reward = 0.80
        # elif error <= 3.0:
        #     reward = 0.70
        # elif error <= 4.0:
        #     reward = 0.60
        # elif error <= 5.0:
        #     reward = 0.50
        # elif error <= 6.0:
        #     reward = 0.40
        # elif error <= 7.0:
        #     reward = 0.30
        # elif error <= 8.0:
        #     reward = 0.20
        # elif error <= 9.0:
        #     reward = 0.10
        # else:
        #     reward = 0
        # -----------------------------------------------
        # print the env
        
        # Rewards scheme 5
        # -------------------------------
        error = abs(target_temperature - predicted_temperature)
        self.reward = (1- (error / 12)) ** 2
        # # -------------------------------
        done = False
        
        if self.pbar is not None:
            self.pbar.set_description(f"(idx, step): ({self.sampled_idx}, {self.current_step}) | True temperature: {target_temperature} | Predicted temperature: {predicted_temperature} | reward: {self.reward:.2f}")
        
        self.sampled_idx = random.randint(0, len(self.dataset) - 1)
        self.current_step = random.randint(0, self.dataset[self.sampled_idx]['observations'].shape[0] - 1)
        
        obs, done = self._next_observation()
        self.total_steps += 1
        return obs, self.reward, done, acc, target_temperature, predicted_temperature, self.total_steps

    def reset(self):
        self.sampled_idx = random.randint(0, len(self.dataset) - 1)
        self.current_step = random.randint(0, self.dataset[self.sampled_idx]['observations'].shape[0] - 1)
        
        return self.dataset[self.sampled_idx]['observations'][self.current_step]

    def _next_observation(self):
        # if self.dataset[self.sampled_idx]['terminals'][self.current_step]:
        #     done = True
        #     obs = self.reset()
        #     return obs, done
            
        obs = self.dataset[self.sampled_idx]['observations'][self.current_step]
        done = False
        return obs, done

    def eval(self):
        self.training = False
        
    def get_true_temperature(self):
        target_temperature = self.dataset[self.sampled_idx]['actions'][self.current_step]
        target_temperature = self.action_space.actions_map[target_temperature]
        return target_temperature
        
    # def render(self, pbar):
    #     target_temperature = self.get_true_temperature()
    #     if self.action is not None:
    #         predicted_temperature = self.action_space.actions_map[self.action]
    #     else:
    #         predicted_temperature = None
            
    #     pbar.set_description(f"Sampled idx: {self.sampled_idx} | Target Temperature: {target_temperature} | Predicted Temperature: {predicted_temperature} | reward: {self.reward:.2f}")
