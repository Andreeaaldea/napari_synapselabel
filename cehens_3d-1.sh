#!/bin/bash

#SBATCH -J cd-process-2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem 120G
#SBATCH --ntasks=10
#SBATCH -t 4-20:00
#SBATCH --gres=gpu:1
#SBATCH -o cd-process-2.out
#SBATCH -e cd-process-2.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youremail@domain.com

module avail python

