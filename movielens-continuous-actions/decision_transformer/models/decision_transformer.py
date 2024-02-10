import numpy as np
import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model, Attention, Block
from sklearn.metrics.pairwise import cosine_similarity

class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            action_vocab,
            hidden_size,
            user_embedding_dim=None,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.user_embedding_dim = user_embedding_dim
        self.hidden_size = hidden_size
        self.config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        self.action_vocab = action_vocab
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        self.transformer = GPT2Model(self.config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_users = torch.nn.Linear(self.user_embedding_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

        # self.crossattention = Attention(nx=128, n_ctx=self.config.n_ctx, config=self.config, scale=False, is_cross_attention=True).to('cuda')
        # self.ln_cross_attn = nn.LayerNorm(128, eps=self.config.layer_norm_epsilon).to('cuda')

        # self.config.add_cross_attention=True

        
        # self.block = Block(n_ctx=self.config.n_ctx, config=self.config, scale=False)

    # def cross_attention_based_fusion(states, user_embeddings):
    #     pass

    # def fuse_user_embeddings(states, user_embeddings, strategy='cross'):
    #     if strategy == 'early':
    #         states = states + user_embeddings
    #     if strategy == 'cross':
    #         states = cross_attention_based_fusion(states, user_embeddings)
    
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, user_embeddings=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # print(f"states in forward shape: {states.shape}")
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        # if user_embeddings is not None:
        #     user_embeddings = self.embed_users(user_embeddings)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        if user_embeddings is not None and self.config.fusion_strategy == 'early':
            # print(f"taking steps for early fusion")
            user_embeddings = self.embed_users(user_embeddings)
            state_embeddings = state_embeddings + user_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # print(f"stacked_inputs.shape: {stacked_inputs.shape} and attention_mask shape: {stacked_attention_mask.shape}")
        # we feed in the input embeddings (not word indices as in NLP) to the model

        if self.config.fusion_strategy == 'cross':
            # print(f"Doing cross-attention")
            encoder_hidden_states = nn.functional.pad(user_embeddings, (0, self.hidden_size - user_embeddings.shape[2]))

            transformer_outputs = self.transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                # fusion_strategy = 'cross',
            )
        elif self.config.fusion_strategy == 'late':
            transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        else:
            # print(f"Now training with early fused embeddings")
            transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        

        
            
            
        x = transformer_outputs['last_hidden_state']
        # print(f"x from transformer outputs shape: {x.shape}")


        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        # print(f"transformed state shape: {x[:,1].shape}")
        # transformed_states = x[:,1]

        # print(f"config: {self.config}")
        # if self.config.fusion_strategy == "cross":
        #     stacked_attention_mask = torch.stack(
        #     (attention_mask, attention_mask), dim=1).permute(0, 2, 1).reshape(batch_size, 2*seq_length)
        #     encoder_hidden_states = user_embeddings

        #     print(f"hidden_states shape: {x[:,1].shape}, encoder_hidden_states.shape: {encoder_hidden_states.shape}, attention_mask.shape: {attention_mask.shape}")
        #     # pass

        #     hidden_states = x[:,1]
        #     print(f"hidden_states.device: {hidden_states.device}")
        #     encoder_hidden_states = nn.functional.pad(encoder_hidden_states, (0, hidden_states.shape[2] - encoder_hidden_states.shape[2]))

        #     cross_attn_outputs = self.block(hidden_states=hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states,)
        #     # cross_attn_outputs = crossattention(
        #     #     ln_cross_attn(hidden_states.to('cuda')),
        #     #     attention_mask=stacked_attention_mask,
        #     #     head_mask=None,
        #     #     encoder_hidden_states=encoder_hidden_states,
        #     #     encoder_attention_mask=None,
        #     #     output_attentions=False,
        #     # )
        #     # # pass
        #     print(f"ATTENTION OUTPUT: {cross_attn_outputs.shape}")
        #     x[:,1] = cross_attn_outputs


        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action

        if self.config.fusion_strategy == 'late':
            # print(f"Now predicting with Late Fusion")
            user_embeddings = self.embed_users(user_embeddings)
            x[:,1] = x[:,1] + user_embeddings
        raw_action_preds = self.predict_action(x[:,1])  # predict next action given state

        # Find the most similar action to the predicted raw_action from the action_vocab
        # APPROACH 1: Cosine similarity
        # similarities = cosine_similarity(raw_action_preds.view(-1, self.act_dim).cpu().detach().numpy(), self.action_vocab)

        # # Find indices of closest neighbors for each prediction in the batch
        # top_k = 1  # Number of closest neighbors to find
        # closest_indices = np.argsort(-similarities, axis=1)[:, :top_k]

        # closest_vectors = self.action_vocab[closest_indices.flatten()].reshape(batch_size, seq_length, self.act_dim)

        # action_preds = torch.from_numpy(closest_vectors).cuda().requires_grad_(True)
        
        #### ------ APPROACH 2: Using Euclidean distance ------ ####
        # # Convert the model's prediction tensor to a NumPy array
        # raw_action_preds_np = raw_action_preds.view(-1, self.act_dim).cpu().detach().numpy()

        # # Calculate Euclidean distances between predictions and action vocab
        # distances = np.linalg.norm(self.action_vocab[:, None] - raw_action_preds_np, axis=2)

        # # Find indices of closest neighbors for each prediction in the batch
        # top_k = 1  # Number of closest neighbors to find
        # closest_indices = np.argmin(distances, axis=0)

        # # Gather closest vectors based on indices
        # closest_vectors = self.action_vocab[closest_indices]

        # # Reshape to the desired shape (batch_size, seq_length, self.act_dim)
        # action_preds = torch.from_numpy(closest_vectors.reshape(raw_action_preds.size(0), raw_action_preds.size(1), self.act_dim)).cuda().requires_grad_(True)

        #--------- APPROACH 3 --------#
        similarities = cosine_similarity(raw_action_preds.view(-1, self.act_dim).cpu().detach().numpy(), self.action_vocab)

        # Find indices of closest neighbors for each prediction in the batch
        closest_indices = np.argmax(similarities, axis=1)

        closest_vectors = self.action_vocab[closest_indices.flatten()].reshape(batch_size, seq_length, self.act_dim)

        action_preds = torch.from_numpy(closest_vectors).cuda().requires_grad_(True)
        return state_preds, action_preds, return_preds



    def get_action(self, states, actions, rewards, returns_to_go, timesteps, user_features,**kwargs):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        user_features = user_features.reshape(1, -1, self.user_embedding_dim)
        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]
            user_features = user_features[:,-self.max_length:]
            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
            user_features = torch.cat(
                [torch.zeros((user_features.shape[0], self.max_length-user_features.shape[1], self.user_embedding_dim), device=states.device), user_features],
                dim=1).to(dtype=torch.float32)
            # print(f"in dt.py: user_features.shape: {user_features.shape}, states.shape: {states.shape}")
        else:
            attention_mask = None

        
        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, user_embeddings=user_features, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]