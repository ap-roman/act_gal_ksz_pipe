#!/bin/bash
#SBATCH -A aroman

# module purge
# # module load openmpi/gcc
# module load slurm
# # module load hdf5_18
# module load anaconda3

# source /home/aroman/venv/bin/activate

sbatch --partition=amdq --ntasks=16 --cpus-per-task=16 --mail-type=ALL --mail-user=aroman@perimeterinstitute.ca slurm-script.sh