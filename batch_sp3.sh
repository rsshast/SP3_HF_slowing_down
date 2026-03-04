#!/bin/bash

#SBATCH --job-name=sp3_hf
#SBATCH --mail-type=All
#SBATCH --nodes=1
#SBATCH --partition=largemem
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1024Gb
#SBATCH --time=3:00:00
#SBATCH --account=bckiedro0
#SBATCH --export=ALL
#SBATCH --output=sp3_b2_eigen_50000_2.out

srun --cpu-bind=cores python HF_SD_Sp3.py
