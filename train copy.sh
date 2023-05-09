#!/bin/bash
#SBATCH --time=1-24:00:00
#SBATCH --partition=class
#SBATCH --qos=default
#SBATCH --account=class
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err
set -x

srun bash -c "source /fs/classhomes/spring2023/cmsc828a/c828a050/.profile;conda activate cmsc828_final; \
python3 src_mtl/main.py \
    --config src_mtl/configs/config_tayo.yml \
    --train"
 