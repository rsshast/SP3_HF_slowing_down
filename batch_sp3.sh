#!/bin/bash

#SBATCH --job-name=sp3_hf
#SBATCH --mail-type=All
#SBATCH --nodes=1
#SBATCH --partition=gpu 
#SATCH --partition=standard
#SBATCH --gres=gpu:1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=16Gb
#SBATCH --time=00:10:00
#SBATCH --account=bckiedro0
#SBATCH --export=ALL
#SBATCH --output=sp3_gpu.out

srun --cpu-bind=cores python HF_SD_Sp3.py
