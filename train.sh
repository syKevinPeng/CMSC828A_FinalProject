#!/bin/bash
#SBATCH --time=1-24:00:00
#SBATCH --partition=class
#SBATCH --qos=default
#SBATCH --account=class
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/fs/classhomes/spring2023/cmsc828a/c828a010/CMSC828A_FinalProject/slurm_output/slurm-%j.out
#SBATCH --error=/fs/classhomes/spring2023/cmsc828a/c828a010/CMSC828A_FinalProject/slurm_output/slurm-%j.err
set -x

srun bash -c "source /fs/classhomes/spring2023/cmsc828a/c828a010/.profile;\
python3 src/main.py \
    --config src/configs/config_lining.yml \
    --train"
 
