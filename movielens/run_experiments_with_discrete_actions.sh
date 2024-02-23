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


# Ablation studies with different action embed shapes
python experiment.py --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme naive
python experiment.py --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme binary
python experiment.py --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme range

# Ablation studies with different fusion strategies in discrete action spaces
python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme naive
python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme binary
python experiment.py --use_personalized_embeddings yes --fusion_strategy early --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme range

python experiment.py --use_personalized_embeddings yes --fusion_strategy late --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme naive
python experiment.py --use_personalized_embeddings yes --fusion_strategy late --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme binary
python experiment.py --use_personalized_embeddings yes --fusion_strategy late --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme range

python experiment.py --use_personalized_embeddings yes --fusion_strategy cross --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme naive
python experiment.py --use_personalized_embeddings yes --fusion_strategy cross --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme binary
python experiment.py --use_personalized_embeddings yes --fusion_strategy cross --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10 --reward_scheme range




