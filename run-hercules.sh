#!/bin/bash
# Submission script for Hercules2
#SBATCH --job-name=VaryMinions
#SBATCH --time=10-00:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12288 # 12GB
#SBATCH --partition=gpu
#
#SBATCH --array=1-40
#SBATCH --output="slurm-output/slurm-%A_%a.out"
#
#SBATCH --mail-user=sophie.fortz@unamur.be
#SBATCH --mail-type=ALL
#
# ------------------------- work -------------------------

module load foss
module load releases/2020b
module load TensorFlow/2.4.1-foss-2020b
module load scikit-learn/0.23.2-foss-2020b
module load SciPy-bundle/2020.11-foss-2020b
mkdir -p $LOCALSCRATCH/$SLURM_JOB_ID
rsync -azu $CECIHOME/VaryMinions $LOCALSCRATCH/$SLURM_JOB_ID/
cd $LOCALSCRATCH/$SLURM_JOB_ID/VaryMinions/scripts/training_NN/
echo "Job start at $(date)"
python3 job-array-BPIC.py
rsync -azu $LOCALSCRATCH/$SLURM_JOB_ID/VaryMinions/results/training_metrics/ $CECIHOME/VaryMinions/results/training_metrics/
echo "Job end at $(date)"
rm -rf $LOCALSCRATCH/$SLURM_JOB_ID