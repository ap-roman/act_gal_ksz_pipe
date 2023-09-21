#!/bin/bash
#SBATCH -A aroman

sbatch --partition=defq --ntasks=16 --cpus-per-task=20 slurm-script.sh