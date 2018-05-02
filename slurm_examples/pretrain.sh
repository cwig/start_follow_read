#!/bin/bash


#SBATCH --time=24:00:00
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2666M


source activate sfr_env

python -u sol_pretraining.py sample_config.yaml >log_sol_pretrain.out 2>&1 &
python -u lf_pretraining.py sample_config.yaml >log_lf_pretrain.out 2>&1 &
python -u hw_pretraining.py sample_config.yaml >log_hw_pretrain.out 2>&1 &

wait
