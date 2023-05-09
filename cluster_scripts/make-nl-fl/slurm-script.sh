#!/bin/bash

# module purge
# # module load openmpi/gcc
# module load slurm
# # module load hdf5_18
# module load anaconda3

# source activate ksz4

# srun python3 make-nl-fl-cluster.py
# srun --mpi=pmi2 -n 4 python3 mpi-test.py
mpiexec python3 make-nl-fl-cluster.py