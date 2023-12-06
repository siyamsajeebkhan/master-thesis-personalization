from gym.spaces import Space
import numpy as np
import gym
from gym import utils
# from gym.envs.mujoco import mujoco_env
import os
import random
import pickle
class CustomActionSpace(Space):
    def __init__(self, shape=None, dtype=None):
        super().__init__(shape, dtype)
    def sample(self):
        """Randomly sample an element of this space. Can be 
        uniform or non-uniform sampling based on boundedness of space."""
        actions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        actions_map = {idx: action for idx, action in enumerate(actions)}
        actions = list(actions_map.keys())
        return np.random.choice(actions)
    
class MovieLensEnv(gym.Env):
    
    def __init__(self):
        # print("__init__ method")
        # with open('../gym/data/mlens/mlens-test-trajectories-v1.pkl', 'rb') as f:
        with open('/home/q621464/Desktop/Thesis/code/decision-transformer/gym/data/mlens/mlens-test-trajectories-v1.pkl', 'rb') as f:
            trajectories = pickle.load(f)
        super(MovieLensEnv, self).__init__()
        self.trajectories = trajectories
        self.action_space = CustomActionSpace(shape=(1, 1))
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.trajectories['observations'].shape[1], ), dtype=np.float32) 
        self.sampled_idx = None

    def step(self, action):
        # print(f'sampled index: {self.sampled_idx}')
        # print(f"Original Action: {self.trajectories['actions'][self.sampled_idx]} and Taken action: {action} with type: {type(action)}")
        # self.current_step += 1

        target_action = self.trajectories['actions'][self.sampled_idx]
        pred_action = action[0]
        if action > target_action:
            reward = target_action / pred_action
        else:
            reward = pred_action / target_action

        # reward = abs(action - self.trajectories['actions'][self.sampled_idx]) # idx is the index of the row which was selected as a state
        done = False
        obs = self._next_observation()
        return obs, reward, done, {}
    
    def reset(self):
        self.sampled_idx = random.randint(0, self.trajectories['observations'].shape[0]-1)
        return self.trajectories['observations'][self.sampled_idx]

    def _next_observation(self):
        self.sampled_idx = random.randint(0, self.trajectories['observations'].shape[0]-1)
        obs = self.trajectories['observations'][self.sampled_idx]
        return obs