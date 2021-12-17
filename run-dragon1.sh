#!/bin/bash
# Submission script for Dragon1
#SBATCH --job-name=VaryMinions
#SBATCH --time=3-01:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
# For the K20M
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:kepler:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12288 # 12GB
#
#SBATCH --array=2
#
#SBATCH --mail-user=sophie.fortz@unamur.be
#SBATCH --mail-type=ALL
#
# ------------------------- work -------------------------

module load foss
module load releases/2019b
module load TensorFlow/2.4.1-fosscuda-2019b-Python-3.7.4
mkdir -p $LOCALSCRATCH/$SLURM_JOB_ID
rsync -azu $CECIHOME/VaryMinions $LOCALSCRATCH/$SLURM_JOB_ID/
cd $LOCALSCRATCH/$SLURM_JOB_ID/VaryMinions/scripts/training_NN/
echo "Job start at $(date)"
python3 Claroline-dis_10.py
echo "Job end at $(date)"
rsync -azu $LOCALSCRATCH/$SLURM_JOB_ID/VaryMinions/ $CECIHOME/VaryMinions/
rm -rf $LOCALSCRATCH/$SLURM_JOB_ID

