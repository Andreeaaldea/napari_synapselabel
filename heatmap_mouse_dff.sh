#!/bin/bash
#SBATCH --job-name=heatmap_dff

#SBATCH --output=barinsaw_dff_%j.out
#SBATCH --error=barinsaw_dff_%j.err
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


echo "[START] $(date)"
python heatmap_mouse.py

echo "[END] $(date)"