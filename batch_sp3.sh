#!/bin/bash

#SBATCH --job-name=sp3
#SBATCH --mail-type=All
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --account=bckiedro0
#SBATCH --partition=standard
#SBATCH --export=ALL
#SBATCH --output=sp3.out

python HF_SD_Sp3.py
