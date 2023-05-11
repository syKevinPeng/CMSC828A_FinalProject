#!/bin/bash
#SBATCH --time=1-24:00:00
#SBATCH --partition=class
#SBATCH --qos=default
#SBATCH --account=class
#SBATCH --gres=gpu:1
#SBATCH --output=/fs/classhomes/spring2023/cmsc828a/c828a006/slurm_output/slurm-%j.out
#SBATCH --error=/fs/classhomes/spring2023/cmsc828a/c828a006/slurm_output/slurm-%j.err
set -x

srun bash -c "source /fs/classhomes/spring2023/cmsc828a/c828a050/.profile;\
python3 src/main.py \
    --config src/configs/config_tayo.yml \
    --train"
 
