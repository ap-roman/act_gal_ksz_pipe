from mpi4py import MPI

import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == "__main__":
	print(f'my rank is {rank} of {size} on node {MPI.Get_processor_name()}')