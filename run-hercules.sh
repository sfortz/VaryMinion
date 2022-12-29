#!/bin/bash
# Submission script for Hercules2
#SBATCH --job-name=VaryMinions
#SBATCH --time=10-00:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000 # 4GB
#SBATCH --partition=gpu
#
#SBATCH --array=16-20
#SBATCH --output="slurm-output/slurm-%A_%a.out"
#
#SBATCH --mail-user=sophie.fortz@unamur.be
#SBATCH --mail-type=ALL
#
# ------------------------- work -------------------------

cd
module load releases/2020b
module load Python/3.8.6-GCCcore-10.2.0
module load foss/2020b
module load cuDNN/8.0.4.30-CUDA-11.1.1
module load CUDA/11.1.1-GCC-10.2.0
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