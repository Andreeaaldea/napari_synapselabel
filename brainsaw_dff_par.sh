#!/bin/bash
#SBATCH --job-name=barinsaw_dff
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


input_root="/ceph/mrsic_flogel/public/projects/AnAl_20240405_Neuromod_PE/brainsaw/PE_mapping"
output_root="/ceph/mrsic_flogel/public/projects/AnAl_20240405_Neuromod_PE/PE_mapping/dff_data/highrez_3"
search_suffix="stitchedImages_100/2"

dry_run_flag=""           # Set to "--dry_run" to only list folders
test_mode_flag=""         # Set to "--test" to limit each brain to 5 slices
echo "[START] $(date)"
python batch_process_dff_recursive_par.py --input_root $input_root --output_root $output_root --search_suffix $search_suffix $dry_run_flag $test_mode_flag
echo "[END] $(date)"
