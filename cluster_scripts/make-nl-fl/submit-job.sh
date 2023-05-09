#!/bin/bash
#SBATCH -A aroman

# module purge
# # module load openmpi/gcc
# module load slurm
# # module load hdf5_18
# module load anaconda3

# source /home/aroman/venv/bin/activate

sbatch --partition=defq --ntasks=12 --cpus-per-task=20 slurm-script.sh