#!/bin/bash

# Ablation studies with num_steps_per_iter
# python experiment.py --use_personalized_embeddings no --num_steps_per_iter 100 --log_to_wandb True

# python experiment.py --use_personalized_embeddings no --num_steps_per_iter 1000 --log_to_wandb True

# python experiment.py --use_personalized_embeddings no --num_steps_per_iter 10000 --log_to_wandb True


# Ablation studies with K
# python experiment.py --use_personalized_embeddings no --num_steps_per_iter 1000 --log_to_wandb True --K 1 --max_iters 5

# python experiment.py --use_personalized_embeddings no --num_steps_per_iter 1000 --log_to_wandb True --K 10 --max_iters 5

# python experiment.py --use_personalized_embeddings no --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5

# python experiment.py --use_personalized_embeddings no --num_steps_per_iter 1000 --log_to_wandb True --K 30 --max_iters 5

# python experiment.py --use_personalized_embeddings no --num_steps_per_iter 1000 --log_to_wandb True --K 40 --max_iters 5


# Ablation studies with different fusion strategies
# python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10

# python experiment.py --use_personalized_embeddings yes --fusion_strategy late --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10

# python experiment.py --use_personalized_embeddings yes --fusion_strategy cross --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10


# Ablation studies with different action embed shapes
# python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5 --action_embed_shape 16 --num_eval_episodes 5

# python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5 --action_embed_shape 32 --num_eval_episodes 5

# python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5 --action_embed_shape 64 --num_eval_episodes 5

# python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5 --action_embed_shape 128 --num_eval_episodes 5


# Ablation studies with K
# python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 1 --max_iters 5 --action_embed_shape 32 --num_eval_episodes 5

# python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 10 --max_iters 5 --action_embed_shape 32 --num_eval_episodes 5

# python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5 --action_embed_shape 32 --num_eval_episodes 5

# python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 30 --max_iters 5 --action_embed_shape 32 --num_eval_episodes 5

# python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 40 --max_iters 5 --action_embed_shape 32 --num_eval_episodes 5


# Remaining ablations
python experiment.py --action_embed_shape 16 --use_personalized_embeddings yes --fusion_strategy late --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5  --num_eval_episodes 5
python experiment.py --action_embed_shape 16 --use_personalized_embeddings yes --fusion_strategy cross --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5  --num_eval_episodes 5
python experiment.py --action_embed_shape 16 --use_personalized_embeddings no --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5  --num_eval_episodes 5

python experiment.py --action_embed_shape 32 --use_personalized_embeddings yes --fusion_strategy late --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5  --num_eval_episodes 5
python experiment.py --action_embed_shape 32 --use_personalized_embeddings no --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5  --num_eval_episodes 5

python experiment.py --action_embed_shape 64 --use_personalized_embeddings yes --fusion_strategy late --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5  --num_eval_episodes 5
python experiment.py --action_embed_shape 64 --use_personalized_embeddings yes --fusion_strategy cross --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5  --num_eval_episodes 5
python experiment.py --action_embed_shape 64 --use_personalized_embeddings no  --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5  --num_eval_episodes 5

python experiment.py --action_embed_shape 128 --use_personalized_embeddings yes --fusion_strategy late --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5  --num_eval_episodes 5
python experiment.py --action_embed_shape 128 --use_personalized_embeddings yes --fusion_strategy cross --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5  --num_eval_episodes 5
python experiment.py --action_embed_shape 128 --use_personalized_embeddings no  --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 5  --num_eval_episodes 5