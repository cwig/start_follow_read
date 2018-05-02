#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --ntasks=24
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2666M

module load cuda/8.0
module load cudnn/6.0_cuda-8.0
module load python
module load python-pytorch
module load opencv/3/0

CUDA_VISIBLE_DEVICES=0 python -u continuous_validation.py sample_config.yaml >validation.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u continuous_sol_training.py sample_config.yaml >sol_training.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u continuous_lf_training.py sample_config.yaml >lf_training.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u continuous_hw_training.py sample_config.yaml >hw_training.out 2>&1 &

wait
