import random
import pickle
import numpy as np
import gym
from gym.spaces import Box, Space
from tqdm import tqdm
import time
# from atari.mingpt.utils import set_seed
# set_seed(123)
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
    def __init__(self, test_traj_path, use_prev_temp_as_feature=False, van_specific_embeddings=None, pbar=None):
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
        self.use_prev_temp = use_prev_temp_as_feature
        self.idx_of_prev_temp_feat = np.where(self.dataset[0]['features'] == 'd_prev_target_temp')[0][0]
        self.personalized_features = van_specific_embeddings

    def step(self, action):
        self.action = action
        temperature = self.dataset[self.sampled_idx]['targets'][self.current_step]
        # target_temperature = self.action_space.actions_map[temperature]
        target_temperature = temperature

        prev_temperature = self.dataset[self.sampled_idx]['observations'][self.current_step][self.idx_of_prev_temp_feat]
        
        predicted_temperature = self.action_space.actions_map[action]
        
        # print(f"target_temperature: {target_temperature} | predicted_temperature: {predicted_temperature}")
        acc = 0
        if predicted_temperature == target_temperature:
            acc = 1

        # Rewards scheme 5
        # -------------------------------
        error = abs(target_temperature - predicted_temperature)
        reward = (1- (error / 12)) ** 2
        # # -------------------------------
        
        # Reward for special cases
        if target_temperature != prev_temperature:
            special_reward = reward
        else:
            special_reward = 0
        
        self.reward = reward
        done = False
        
        if self.pbar is not None:
            self.pbar.set_description(f"(idx, step): ({self.sampled_idx}, {self.current_step}) | True temperature: {target_temperature} | Predicted temperature: {predicted_temperature} | Prev temperature: {prev_temperature} | reward: {self.reward:.2f}")
            # time.sleep(0.25)
        self.current_step += 1
        obs, prev_target_temp, done = self._next_observation()
        self.total_steps += 1
        return obs, reward, special_reward, done, acc, target_temperature, predicted_temperature, prev_target_temp, self.total_steps

    def reset(self):
        self.sampled_idx = random.randint(0, len(self.dataset) - 1)
        self.current_step = 0
        traj = self.dataset[self.sampled_idx]
        van_id = traj['van_id']


        obs = traj['observations'][self.current_step]
        prev_target_temp = obs[39]
        if self.use_prev_temp == 'no':
            obs = obs[0:39]
        if self.personalized_features is not None:
            obs = np.hstack((obs, self.personalized_features[van_id]))
        
        return obs, prev_target_temp
    
    def _next_observation(self):
        if self.dataset[self.sampled_idx]['terminals'][self.current_step]:
            done = True
            obs, prev_target_temp = self.reset()
            return obs, prev_target_temp, done
        
        traj = self.dataset[self.sampled_idx]
        van_id = traj['van_id']
        obs = traj['observations'][self.current_step]
        prev_target_temp = obs[39]
        if self.use_prev_temp == 'no':
            obs = obs[0:39]
        if self.personalized_features is not None:
            obs = np.hstack((obs, self.personalized_features[van_id]))
        done = False
        return obs, prev_target_temp, done

    def eval(self):
        self.training = False
        
    def get_true_temperature(self):
        target_temperature = self.dataset[self.sampled_idx]['actions'][self.current_step]
        target_temperature = self.action_space.actions_map[target_temperature]
        return target_temperature
        