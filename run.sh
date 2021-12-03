#!/bin/bash
# Submission script for Dragon2
#SBATCH --job-name=VaryMinions
#SBATCH --time=3-01:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=12288 # 12GB
#SBATCH --partition=gpu
#
#SBATCH --mail-user=sophie.fortz@unamur.be
#SBATCH --mail-type=ALL
#
# ------------------------- work -------------------------

echo "Job start at $(date)"
cd scripts/training_NN/
python3 Claroline-dis_10-tensorflow.py
echo "Job end at $(date)"
