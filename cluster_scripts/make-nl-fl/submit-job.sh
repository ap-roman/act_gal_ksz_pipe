#!/bin/bash
#SBATCH -A aroman

# module purge
# # module load openmpi/gcc
# module load slurm
# # module load hdf5_18
# module load anaconda3

# source /home/aroman/venv/bin/activate

sbatch --partition=defq --ntasks=16 --cpus-per-task=20 --mail-type=ALL --mail-user=aroman@perimeterinstitute.ca slurm-script.sh