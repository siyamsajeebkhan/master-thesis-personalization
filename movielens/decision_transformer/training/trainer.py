import numpy as np
import torch

import time
from tqdm import tqdm

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, use_personalized_embeddings='no'):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()
        self.use_personalized_embeddings = use_personalized_embeddings

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses, val_losses = [], []
        logs = dict()

        train_start = time.time()

        self.model.train()
        total_samples = 0
        total_corr_samples = 0
        for _ in tqdm(range(num_steps)):
            train_loss, num_samples_per_step, num_corr_samples_per_step = self.train_step()
            total_samples += num_samples_per_step
            total_corr_samples += num_corr_samples_per_step
            
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
        
        logs['time/training'] = time.time() - train_start
        logs['training/train_accuracy'] = total_corr_samples / total_samples
        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
                if "accuracy_mean" in k:
                    print("-" * 45)
                if k == "training/train_accuracy":
                    print("-" * 45)
            print('=' * 80)

        return logs

    # def train_iteration_modified(self, num_steps, iter_num=0, print_logs=False):

    #     train_losses, val_losses = [], []
    #     logs = dict()

    #     train_start = time.time()

    #     self.model.train()
    #     total_samples, total_val_samples = 0, 0
    #     total_corr_samples, total_val_corr_samples = 0, 0
    #     for _ in tqdm(range(num_steps)):
    #         train_loss, num_samples_per_step, num_corr_samples_per_step = self.train_step()
    #         total_samples += num_samples_per_step
    #         total_corr_samples += num_corr_samples_per_step
            
    #         train_losses.append(train_loss)
    #         if self.scheduler is not None:
    #             self.scheduler.step()
                
    #     eval_start = time.time()
    #     self.model.eval()
    #     for _ in tqdm(range(100)):
    #         val_loss, num_samples_per_step, num_corr_samples_per_step = self.val_step()
    #         total_val_samples += num_samples_per_step
    #         total_val_corr_samples += num_corr_samples_per_step
            
    #         val_losses.append(val_loss)
    #         if self.scheduler is not None:
    #             self.scheduler.step()
        
    #     logs['time/training'] = time.time() - train_start
    #     logs['training/train_accuracy'] = (total_corr_samples * 100 / total_samples)
    #     logs['evaluation/eval_accuracy'] = (total_val_corr_samples * 100 / total_val_samples)

    #     logs['time/total'] = time.time() - self.start_time
    #     logs['time/evaluation'] = time.time() - eval_start
    #     logs['training/train_loss_mean'] = np.mean(train_losses)
    #     logs['training/train_loss_std'] = np.std(train_losses)
    #     logs['evaluation/val_loss_mean'] = np.mean(val_losses)
    #     logs['evaluation/val_loss_std'] = np.std(val_losses)
    #     for k in self.diagnostics:
    #         logs[k] = self.diagnostics[k]

    #     if print_logs:
    #         print('=' * 80)
    #         print(f'Iteration {iter_num}')
    #         for k, v in logs.items():
    #             print(f'{k}: {v}')
    #             if "accuracy_mean" in k:
    #                 print("-" * 45)
    #             if k == "training/train_accuracy":
    #                 print("-" * 45)
    #         print('=' * 80)

    #     return logs


    # def train_step(self):
    #     print(f"Train step")
    #     states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
    #     state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

    #     state_preds, action_preds, reward_preds = self.model.forward(
    #         states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
    #     )
        
    #     # note: currently indexing & masking is not fully correct
    #     loss = self.loss_fn(
    #         state_preds, action_preds, reward_preds,
    #         state_target[:,1:], action_target, reward_target[:,1:],
    #     )
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     return loss.detach().cpu().item()
