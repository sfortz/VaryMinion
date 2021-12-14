#!/bin/bash
# Submission script for Dragon1
#SBATCH --job-name=VaryMinions
#SBATCH --time=3-01:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12288 # 12GB
#SBATCH --partition=gpu
#
#SBATCH --array=3,4,7-40
#
#SBATCH --mail-user=sophie.fortz@unamur.be
#SBATCH --mail-type=ALL
#
# ------------------------- work -------------------------

module load foss
module load releases/2019b
module load TensorFlow/2.4.1-fosscuda-2019b-Python-3.7.4
echo "Job start at $(date)"
cd scripts/training_NN/
python3 job-array-BPIC.py
echo "Job end at $(date)"
