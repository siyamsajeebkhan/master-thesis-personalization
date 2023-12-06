import gym
import numpy as np
import pandas as pd
import torch
# import wandb

import argparse
import pickle
import random
import sys
from tqdm import tqdm
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
np.random.seed(0)

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    dataset_path = "../data/smart-climate-2021/datasets_with_non_filled_events/train-test-sets/smart-climate-train-trajectories-v1-with-prev-target-temp.pkl"

    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    if variant['use_prev_temp_as_feature'] == 'no':
        for traj in trajectories:
            traj['observations'] = traj['observations'][:, 0:39]

    # Uncomment the following lines for running with van specific embeddings
    van_specific_features = None
    if variant['use_personalized_embeddings'] == 'yes':
        print("Using personalized embeddings")
        with open('../data/embeddings/van_specific_embeddings.pkl', 'rb') as f:
            van_specific_features = pickle.load(f)
        for traj in trajectories:
            van_id = traj['van_id']
            van_feature = van_specific_features[van_id]
            
            observations = traj['observations']
            arr_tiled = np.tile(van_feature, (observations.shape[0], 1))
            traj['observations'] = np.concatenate((observations, arr_tiled), axis=1)

    test_set_path = "../data/smart-climate-2021/datasets_with_non_filled_events/train-test-sets/smart-climate-test-trajectories-v1-with-prev-target-temp.pkl"
    
    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations
    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns, actions = [], [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        actions.append(path['actions'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)
    
    state_dim = states.shape[1]
    act_dim = 1
    unique_actions = np.unique(np.concatenate(actions))
    total_actions = unique_actions.shape[0]
    # total_actions = 49
    
    max_return = np.max(returns)
    max_ret_traj_idx = np.argmax(returns)
    traj_len_of_max_ret = len(trajectories[max_ret_traj_idx]['observations'])
    max_reward_normalized = max_return*100/traj_len_of_max_ret
    
    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print(f"Max return traj len: {traj_len_of_max_ret}, Normalized max reward in 100: {max_reward_normalized}")
    print(f"Total actions: {total_actions}")
    print(f"Unique actions are: {sorted(unique_actions)}")
    print(f"Average trajectory length: {traj_lens.mean()}")
    print(f"State dimension: {state_dim}")
    print('=' * 50)
    
    ##---- Define and initialize the envs
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    elif env_name == 'smartclimate':
        from decision_transformer.envs.smart_climate_env import SmartClimateEnv
        max_ep_len = variant['max_ep_len']
        # max_ep_len = 100
        # env_targets = [max_ep_len*1, max_ep_len*0.7]
        env_targets = [max_ep_len]
        pbar = tqdm(range(max_ep_len), disable=False)
        env = SmartClimateEnv(test_set_path, use_prev_temp_as_feature=variant['use_prev_temp_as_feature'], van_specific_embeddings=van_specific_features, pbar=pbar)
        scale = 1
    else:
        raise NotImplementedError

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K, train=True):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            sampled_states = traj['observations'][si:si + max_len]

            # get sequences from dataset
            s.append(sampled_states.reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        # print(f"actions.min and max: {a.min(), a.max()}")
        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths, accuracies = [], [], []
            special_returns, non_special_returns = [], []
            special_case_counts, special_case_corr_counts = [], []
            eval_action_errors, eval_special_case_errors = [], []
            for _ in tqdm(range(num_eval_episodes), disable=False):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, special_ret, non_special_ret, length, acc, special_case_count, special_case_corr_count, eval_action_error, eval_special_case_error = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            use_prev_temp_feat=variant['use_prev_temp_as_feature'],
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                special_returns.append(special_ret)
                lengths.append(length)
                accuracies.append(acc)
                non_special_returns.append(non_special_ret)
                
                special_case_counts.append(special_case_count)
                special_case_corr_counts.append(special_case_corr_count)
                eval_action_errors.append(eval_action_error)
                eval_special_case_errors.append(eval_special_case_error)
            special_cases_acc = np.sum(special_case_corr_counts)/np.sum(special_case_counts)
            non_special_case_count = (num_eval_episodes * max_ep_len) - np.sum(special_case_counts)
            non_special_case_corr_count = int(round(np.sum(accuracies*max_ep_len))) - np.sum(special_case_corr_counts)
            return {
                f'target_{target_rew}_eval_action_error': np.mean(eval_action_errors),
                f'target_{target_rew}_eval_special_action_error': np.mean(eval_special_case_errors),
                f'target_{target_rew}_overall_return_mean': np.mean(returns),
                f'target_{target_rew}_overall_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'target_{target_rew}_overall_accuracy_mean': f'{np.mean(accuracies):.4f}',
                f'target_{target_rew}_special_case_count_total': np.sum(special_case_counts),
                f'target_{target_rew}_special_case_corr_count_total': np.sum(special_case_corr_counts),
                f'target_{target_rew}_special_case_accuracy': f'{np.mean(special_cases_acc):.4f}',
                f'target_{target_rew}_total_special_returns': np.sum(special_returns),
                f'target_{target_rew}_special_return_mean': np.mean(special_returns),
                f'target_{target_rew}_special_return_std': np.std(special_returns),
                f'target_{target_rew}_non_special_case_count_total': np.sum(non_special_case_count),
                f'target_{target_rew}_non_special_case_corr_count_total': np.sum(non_special_case_corr_count),
                f'target_{target_rew}_non_special_case_accuracy': f'{non_special_case_corr_count/non_special_case_count:.4f}',
                f'target_{target_rew}_total_non_special_returns': np.sum(non_special_returns),
                f'target_{target_rew}_non_special_return_mean': np.mean(non_special_returns),
                f'target_{target_rew}_special_return_std': np.std(non_special_returns),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            vocab_size=total_actions,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='smartclimate')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=20)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)
    parser.add_argument('--max_ep_len', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--use_prev_temp_as_feature', type=str, default='yes')
    parser.add_argument('--use_personalized_embeddings', type=str, default='yes')

    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))