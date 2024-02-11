import random
import pickle
import numpy as np
import gym
from gym.spaces import Box, Space
from tqdm import tqdm
import time


class CustomActionSpace(Space):
    def __init__(self, shape=None, dtype=None):
        super().__init__(shape, dtype)
        actions = np.arange(0.5, 5.5, 0.5)
        self.actions_map = {idx: action for idx, action in enumerate(actions)}
        self.actions = list(self.actions_map.keys())
    
class MovieLensEnv(gym.Env):
    
    def __init__(self, test_traj_path, use_prev_temp_as_feature=False, user_specific_features=None, reward_scheme=None, pbar=None):
        # print("__init__ method")
        # with open('../gym/data/mlens/mlens-test-trajectories-v1.pkl', 'rb') as f:
        with open(test_traj_path, 'rb') as f:
            self.dataset = pickle.load(f)

        super(MovieLensEnv, self).__init__()

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
        self.personalized_features = user_specific_features
        self.reward_scheme = reward_scheme

    def step(self, action):
        self.action = action
        target_rating = self.dataset[self.sampled_idx]['targets'][self.current_step]
        
        pred_rating = self.action_space.actions_map[action]
        
        acc = 0
        if pred_rating == target_rating:
            acc = 1

        # Binary Rewards scheme 
        # -------------------------------
        if self.reward_scheme == 'binary':
            if target_rating >= 3.5 and pred_rating >= 3.5:
                self.reward = 1
            else:
                self.reward = 0
        
        # # -------------------------------
        # # Rewards scheme 5
        # # -------------------------------
        else:
            error = abs(target_rating - pred_rating)
            self.reward = (1- (error / 4.5)) ** 2
        # # # -------------------------------


        # # Reward for special cases
        # if target_rating != pred_rating:
        #     special_reward = reward
        # else:
        #     special_reward = 0
        
        done = False
        
        if self.pbar is not None:
            self.pbar.set_description(f"(idx, step): ({self.sampled_idx}, {self.current_step}) | True rating: {target_rating} | Predicted rating: {pred_rating} | reward: {self.reward:.2f}")
            # time.sleep(0.25)
        self.current_step += 1
        self.total_steps += 1
        if self.personalized_features is not None:
            obs, done, user_feature = self._next_observation()
            return obs, self.reward, done, acc, target_rating, pred_rating, self.total_steps, user_feature
        else:
            obs, done = self._next_observation()
            return obs, self.reward, done, acc, target_rating, pred_rating, self.total_steps

    
    def reset(self):
        self.sampled_idx = random.randint(0, len(self.dataset) - 1)
        self.current_step = 0
        traj = self.dataset[self.sampled_idx]
        user_id = traj['user_id']


        obs = traj['observations'][self.current_step]

        if self.personalized_features is not None:
            user_feature = self.personalized_features[self.personalized_features['userId'] == user_id].values.reshape(-1)
            return obs, user_feature
        else:
            return obs
    
    def _next_observation(self):
        if self.dataset[self.sampled_idx]['terminals'][self.current_step]:
            done = True
            if self.personalized_features is not None:
                obs, user_feature = self.reset()
                return obs, done, user_feature
            else:
                obs = self.reset()
                return obs, done
        
        traj = self.dataset[self.sampled_idx]
        user_id = traj['user_id']
        obs = traj['observations'][self.current_step]
        done = False
        if self.personalized_features is not None:
            user_feature = self.personalized_features[self.personalized_features['userId'] == user_id].values.reshape(-1)
            return obs, done, user_feature
        else:
            return obs, done

    def eval(self):
        self.training = False
        
    def get_true_temperature(self):
        target_temperature = self.dataset[self.sampled_idx]['actions'][self.current_step]
        target_temperature = self.action_space.actions_map[target_temperature]
        return target_temperature
        