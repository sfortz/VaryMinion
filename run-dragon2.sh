#!/bin/bash
# Submission script for Dragon2
#SBATCH --job-name=VaryMinions
#SBATCH --time=5-01:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12288 # 12GB
#SBATCH --partition=gpu
#
#SBATCH --array=1-20
#
#SBATCH --mail-user=sophie.fortz@unamur.be
#SBATCH --mail-type=ALL
#
# ------------------------- work -------------------------

module load foss
module load releases/2019b
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
module load scikit-learn/0.22.1-fosscuda-2019b-Python-3.7.4
mkdir -p $LOCALSCRATCH/$SLURM_JOB_ID
rsync -azu $CECIHOME/VaryMinions $LOCALSCRATCH/$SLURM_JOB_ID/
cd $LOCALSCRATCH/$SLURM_JOB_ID/VaryMinions/scripts/training_NN/
echo "Job start at $(date)"
n=10
for ((i=0 ; i<$n ; i++));
do
  echo "Begining exécution of the $(i) th configuration: "
  python3 job-array-claroline-50-dis.py
  rsync -azu $LOCALSCRATCH/$SLURM_JOB_ID/VaryMinions/results/training_metrics/ $CECIHOME/VaryMinions/results/training_metrics/
done
echo "Job end at $(date)"
rm -rf $LOCALSCRATCH/$SLURM_JOB_ID

