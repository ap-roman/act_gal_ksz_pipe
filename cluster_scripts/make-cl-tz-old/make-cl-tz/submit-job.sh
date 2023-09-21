#!/bin/bash
#SBATCH -A aroman

sbatch --partition=defq --ntasks=16 --cpus-per-task=20 --mail-type=ALL --mail-user=aroman@perimeterinstitute.ca slurm-script.sh