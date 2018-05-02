#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2666M

module load cuda/8.0
module load cudnn/6.0_cuda-8.0
module load python
module load python-pytorch
module load opencv/3/0

python -u continuous_validation.py sample_config.yaml init >validation.out 2>&1
