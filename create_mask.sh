#!/bin/bash
#SBATCH --job-name=create_mask
#SBATCH --output=create_mask_%j.out
#SBATCH --error=create_mask_%j.err
#
#SBATCH -p gpu
#SBATCH -n 16

#SBATCH --cpus-per-task=16
#SBATCH -t 4-4:00
#SBATCH --mem=128G
#SBATCH --gres gpu:1
#SBATCH --mail-type ALL
#SBATCH --mail-user andreea.aldea.18@ucl.ac.uk
source ~/.bashrc
module load mamba

source activate napari-env 
export OMP_NUM_THREADS=1

MICE=($(</ceph/mrsic_flogel/public/projects/AnAl_20240405_Neuromod_PE/brainsaw/LC/A065/mice_list.txt))
MOUSE_ID=${MICE[$SLURM_ARRAY_TASK_ID]}


echo "Starting job for $MOUSE_ID"
python batch_mask.py $MOUSE_ID
