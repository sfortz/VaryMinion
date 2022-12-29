#!/bin/bash
# Submission script for Dragon1
#SBATCH --job-name=VaryMinions
#SBATCH --time=10-00:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
# For the K20M
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:kepler:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000 # 4GB
#
#SBATCH --array=19
#SBATCH --output="slurm-output/slurm-%A_%a.out"
#
#SBATCH --mail-user=sophie.fortz@unamur.be
#SBATCH --mail-type=ALL
#
# ------------------------- work -------------------------

module load releases/2020b
module load Python/3.8.6-GCCcore-10.2.0
module load foss/2020b
module load cuDNN/8.0.4.30-CUDA-11.1.1
module load CUDA/11.1.1-GCC-10.2.0
echo "module loaded"

cd
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