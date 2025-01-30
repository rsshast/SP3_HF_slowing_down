#!/bin/bash
# interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=sp3_slowing_down
#SBATCH --mail-type=None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=24Gb
#SBATCH --time=0:30:00
#SBATCH --account=bckiedro0
#SBATCH --partition=standard
#SBATCH --export=ALL
#SBATCH --output=sp3.out

# The application(s) to execute along with its input arguments and options:
python3 run_sp3.py

display results/charts/phi2.png
