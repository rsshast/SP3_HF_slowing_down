#!/bin/bash

#SBATCH --job-name=sp3_slowing_down
#SBATCH --mail-type=None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --mem=50G
#SBATCH --time=10:00:00
#SBATCH --account=bckiedro0
#SBATCH --partition=standard
#SBATCH --export=ALL
#SBATCH --output=sp3.out

python run_sp3.py 
