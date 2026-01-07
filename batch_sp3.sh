#!/bin/bash

#SBATCH --job-name=sp3_hf
#SBATCH --mail-type=All
#SBATCH --nodes=1
#SATCH --partition=gpu 
#SBATCH --partition=standard
#SATCH --gres=gpu:1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=48Gb
#SBATCH --time=00:45:00
#SBATCH --account=bckiedro0
#SBATCH --export=ALL
#SBATCH --output=sp3_eqns.out

srun --cpu-bind=cores python HF_SD_Sp3.py
