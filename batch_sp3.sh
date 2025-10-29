#!/bin/bash

#SBATCH --job-name=sp3_hf
#SBATCH --mail-type=All
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#BSATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --account=bckiedro0
#SBATCH --partition=standard
#SBATCH --export=ALL
#SBATCH --output=sp3_hf.out

srun --cpu-bind=cores python HF_SD_Sp3.py
