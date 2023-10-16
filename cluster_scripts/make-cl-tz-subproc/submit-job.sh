#!/bin/bash
#SBATCH -A aroman
#SBATCH -t 0-6:00              # time limit: (D-HH:MM) 

sbatch --partition=debugq --ntasks=4 --cpus-per-task=20 --mail-type=ALL --mail-user=aroman@perimeterinstitute.ca slurm-script.sh
# sbatch --partition=defq --ntasks=8 --cpus-per-task=20 --mail-type=ALL --mail-user=aroman@perimeterinstitute.ca slurm-script.sh