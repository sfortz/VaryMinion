#!/bin/bash
# Submission script for Dragon2
#SBATCH --job-name=VaryMinions
#SBATCH --time=5-00:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=3072 # 3GB
#SBATCH --partition=gpu
#
#SBATCH --array=6-20
#SBATCH --output="slurm-output/slurm-%A_%a.out"
#
#SBATCH --mail-user=sophie.fortz@unamur.be
#SBATCH --mail-type=ALL
#
# ------------------------- work -------------------------

cd
module load foss
module load releases/2021b
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1
module load scikit-learn/1.0.2-foss-2021b
echo "module loaded"

source ceci-venv/bin/activate
echo "ceci-venv activated"
cd $CECIHOME/VaryMinions

mkdir -p $LOCALSCRATCH/$SLURM_JOB_ID
rsync -azu $CECIHOME/VaryMinions $LOCALSCRATCH/$SLURM_JOB_ID/
cd $LOCALSCRATCH/$SLURM_JOB_ID/VaryMinions/scripts/training_NN/
echo "Job start at $(date)"
srun python3 job-array-claroline-50-dis.py
rsync -azu $LOCALSCRATCH/$SLURM_JOB_ID/VaryMinions/results/training_metrics/ $CECIHOME/VaryMinions/results/training_metrics/
echo "Job end at $(date)"
rm -rf $LOCALSCRATCH/$SLURM_JOB_ID
deactivate
