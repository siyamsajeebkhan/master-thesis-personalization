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
    
    def __init__(self, test_traj_path, movie_embed_to_id, movies_ratings_and_tags, use_prev_temp_as_feature=False, user_specific_features=None, pbar=None):
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
        self.movies_ratings_and_tags = movies_ratings_and_tags
        self.movie_embed_to_id = movie_embed_to_id

    def step(self, action):
        self.action = action.cpu().numpy()

        # Need to create a mapping between actions and rewards
        # If the movie is actually rated by the user: then the reward is the user's rating
        # Else, the reward is the average rating of all users for the movie

        # First let's find the movie_id from the embed
        movie_id = self.movie_embed_to_id[tuple(self.action)]
        user_id = self.dataset[self.sampled_idx]['user_id']

        rating_by_user = self.movies_ratings_and_tags[(self.movies_ratings_and_tags['movieId'] == movie_id) & (self.movies_ratings_and_tags['userId'] == user_id)]['rating']

        if rating_by_user.any():
            self.reward = rating_by_user.values[0]
        else:
            self.reward = self.movies_ratings_and_tags[self.movies_ratings_and_tags['movieId'] == movie_id]['rating_global'].mean()

        # Round to nearest 0.5
        def round_to_nearest_half(number):
            return round(number * 2) / 2

        # Example usage
        self.reward = round_to_nearest_half(self.reward)


        done = False
        
        if self.pbar is not None:
            self.pbar.set_description(f"(idx, step): ({self.sampled_idx}, {self.current_step}) | predicted movie_id: {movie_id} | reward: {self.reward:.2f}")
            # time.sleep(0.25)
        self.current_step += 1
        self.total_steps += 1
        if self.personalized_features is not None:
            obs, done, user_feature = self._next_observation()
            return obs, self.reward, done, user_feature, self.total_steps, movie_id
        else:
            obs, done = self._next_observation()
            return obs, self.reward, done, self.total_steps, movie_id
        
        # return obs, self.reward, done, user_feature, self.total_steps, movie_id
    
    def reset(self):
        self.sampled_idx = random.randint(0, len(self.dataset) - 1)
        self.current_step = 0
        traj = self.dataset[self.sampled_idx]
        user_id = traj['user_id']

        obs = traj['observations'][self.current_step]

        if self.personalized_features is not None:
            user_feature = self.personalized_features[self.personalized_features['userId'] == user_id].values.reshape(-1)

            # print(f"Before injecting personal features, shape of obs: {obs.shape}")
            # print(f"shape of personal feature: {user_feature.shape}")
        # obs = np.hstack((user_feature, obs))
            # print(f"obs with personal features has shape: {obs.shape}")
        if self.personalized_features is not None:
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
        if self.personalized_features is not None:
            user_feature = self.personalized_features[self.personalized_features['userId'] == user_id].values.reshape(-1)
            # print(f"Before injecting personal features, shape of obs: {obs.shape}")
            # print(f"shape of personal feature: {user_feature.shape}")

            # obs = np.hstack((obs, user_feature))
            # print(f"obs with personal features has shape: {obs.shape}")
        done = False
        if self.personalized_features is not None:
            return obs, done, user_feature
        else:
            return obs, done

    def eval(self):
        self.training = False
        
    def get_true_temperature(self):
        target_temperature = self.dataset[self.sampled_idx]['actions'][self.current_step]
        target_temperature = self.action_space.actions_map[target_temperature]
        return target_temperature
        