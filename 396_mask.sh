#!/bin/bash

#SBATCH -J 396_mask_1 #job name
#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 120G # memory pool for all cores
#SBATCH --ntasks=10
#SBATCH -t 4-20:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH -o 396_mask_1.out
#SBATCH -e 396_mask_1.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youremail@domain.com

# Activate the appropriate environment
source ~/.bashrc
conda activate napari-env

# Run the script
python /ceph/mrsic_flogel/public/projects/AnAl_20240405_Neuromod_PE/code/quantifysynapses_processing/create_mask_396.py
