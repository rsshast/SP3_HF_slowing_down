#!/bin/bash

#SBATCH --job-name=sp3_hf
#SBATCH --mail-type=All
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64Gb
#SBATCH --time=01:00:00
#SBATCH --account=bckiedro0
#SBATCH --partition=standard
#SBATCH --export=ALL
#SBATCH --output=sp3_finite_diff_25k_hf.out

srun --cpu-bind=cores python HF_SD_Sp3.py
