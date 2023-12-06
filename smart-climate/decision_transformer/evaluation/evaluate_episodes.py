import numpy as np
import torch
from tqdm import tqdm
import time
def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
 
 
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        use_prev_temp_feat='yes',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, prev_target_temp = env.reset()

    # if use_prev_temp_feat == 'no':
    #     state = state[0:39]
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.empty((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    episode_special_return = 0
    # pbar = tqdm(range(max_ep_len), disable=True)
    episode_corr_samples = 0
    special_case_count, special_case_corr_samples = 0, 0
    pred_actions = []
    target_actions = []
    
    total_special_action_error = 0.0
    for t in range(max_ep_len):
    # for t in pbar:
        # add padding
        # if t==0:
        #     prev_target_temp = states[-1][env.idx_of_prev_temp_feat]

        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            use_action_amebeddings=True,
        )
        
        
        actions[-1] = action
        action = action.detach().cpu().item()
        
        state, reward, special_reward, done, _, target_temp, pred_temp, prev_target_temp, _ = env.step(action)
        
        # special addition: adding the newly pred_temp to replace the prev one
        # states[-1][env.idx_of_prev_temp_feat] = pred_temp
        
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        
        # Calculate eval accuracy
        if target_temp != prev_target_temp:
            special_case_count += 1
            total_special_action_error += (target_temp - pred_temp)**2
        if target_temp == pred_temp:
            episode_corr_samples += 1
            if target_temp != prev_target_temp:
                special_case_corr_samples += 1
                
        pred_actions.append(pred_temp)
        target_actions.append(target_temp)        

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_special_return += special_reward
        episode_length += 1
    # non_special_case_count = max_ep_len - special_case_count
    # non_special_corr_sample_count = episode_corr_samples - special_case_corr_samples
    episode_non_special_return = episode_return - episode_special_return
    eval_action_loss = np.sum((np.array(target_actions) - np.array(pred_actions))**2)
    return episode_return, episode_special_return, episode_non_special_return, episode_length, (episode_corr_samples / max_ep_len), special_case_count, special_case_corr_samples, eval_action_loss, total_special_action_error

