import numpy as np
import torch
from torch.nn import functional as F
from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self, state_mean, state_std):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size, train=True)
        # print(f"rtg of the batch: {rtg[:,:-1]}")
        # print(f"rtg.shape of the batch: {rtg.max()}")
        
        # action_target = torch.clone(actions)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)
        
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]

        logits = action_preds.view(-1, action_preds.size(-1))
        targets = action_target.view(-1).type(torch.int64)
        reward_targets = reward_target.view(-1)
        reward_preds = reward_preds.view(-1)
        # Create a mask tensor where padding elements are False and actual data elements are True
        mask = targets != -10    
        
        
        # Apply the mask to the logits and targets
        masked_logits = logits[mask]
        masked_targets = targets[mask]
        
        masked_reward_targets = reward_targets[mask]
        masked_reward_preds = reward_preds[mask]
        # print(f"masked_logits.shape: {masked_logits.shape}, masked_targets.shape: {masked_targets.shape}")
        # Calculate the cross-entropy loss
        reward_loss = torch.mean((masked_reward_targets - masked_reward_preds) ** 2)
        loss = F.cross_entropy(masked_logits, masked_targets)
        
        # Calculate train accuracy
        probs = F.softmax(masked_logits, dim=-1)
        _, pred_actions = torch.topk(probs, k=1, dim=-1)
        pred_actions = pred_actions.view(-1)
        # print(f"pred_actions.shape: {pred_actions.shape}")
        num_batch_elems = torch.numel(masked_targets)
        num_corr_samples = (masked_targets == pred_actions).sum()
        
        # print(f"targets: {masked_targets}, preds: {pred_actions}")
        # print(f"num_batch_elems: {num_batch_elems}, num_corr_samples: {num_corr_samples}")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(), num_batch_elems, num_corr_samples

    def val_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size, train=False)
        # print(f"rtg of the batch: {rtg[:,:-1]}")
        # print(f"rtg.shape of the batch: {rtg.max()}")
        
        # action_target = torch.clone(actions)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)
        
        with torch.no_grad():
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, train=True,
            )

            logits = action_preds.view(-1, action_preds.size(-1))
            targets = action_target.view(-1).type(torch.int64)
            reward_targets = reward_target.view(-1)
            reward_preds = reward_preds.view(-1)
            # Create a mask tensor where padding elements are False and actual data elements are True
            mask = targets != -10    
            
            # Apply the mask to the logits and targets
            masked_logits = logits[mask]
            masked_targets = targets[mask]
            
            masked_reward_targets = reward_targets[mask]
            masked_reward_preds = reward_preds[mask]
            # print(f"masked_logits.shape: {masked_logits.shape}, masked_targets.shape: {masked_targets.shape}, original targets shape: {targets.shape}")
            # Calculate the cross-entropy loss
            reward_loss = torch.mean((masked_reward_targets - masked_reward_preds) ** 2)
            # loss = F.cross_entropy(masked_logits, masked_targets) + reward_loss
            loss = F.cross_entropy(masked_logits, masked_targets)
            
            # Calculate val accuracy
            probs = F.softmax(masked_logits, dim=-1)
            _, pred_actions = torch.topk(probs, k=1, dim=-1)
            pred_actions = pred_actions.view(-1)
            
            num_batch_elems = torch.numel(masked_targets)
            num_corr_samples = (masked_targets == pred_actions).sum()
        

        with torch.no_grad():
            self.diagnostics['validation/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(), num_batch_elems, num_corr_samples