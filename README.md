
# Developing personalized AI Solutions through Reinforcement Learning

<!-- Md Siyam Sajeeb Khan (EN-52)\, Dogukan Sonmez (EN-52)\ -->


## Overview

Codebase for [Master Thesis: Developing personalized AI Solutions through Reinforcement Learning ](https://atc.bmwgroup.net/confluence/x/kVKz8).
Contains scripts to reproduce experiments.

![image info](./architecture.png)

## Instructions

### Setting up the environment:
Install miniconda or conda following these instructions: <br>
[Miniconda installation](https://docs.conda.io/projects/miniconda/en/latest/) <br>
[Anaconda installation](https://docs.anaconda.com/free/anaconda/install/index.html)

Install the dependencies and activate the virtual environment:
```
conda env create -f pdt-mlens.yml
conda activate pdt
```

### Running the experiments:<br>
```
cd movielens-continuous-actions
python experiment.py --action_embed_shape 32 --use_personalized_embeddings yes --fusion_strategy cross --num_steps_per_iter 1000 --log_to_wandb True --K 20 --max_iters 10  --num_eval_episodes 10
```
