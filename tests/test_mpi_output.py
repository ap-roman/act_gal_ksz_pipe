from mpi4py import MPI

from pixell import enmap, enplot
from pixell.enmap import downgrade

import numpy as np

from fnl_pipe.util import OutputManager, get_fname

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data_path = '/home/aroman/data/'
act_path = data_path + 'act_pub/'
map_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_map_srcfree.fits' # public

if __name__ == "__main__":
    om = OutputManager(base_path='output', title='test_om_mpi', logs=['log',],
                       mpi_rank=rank, mpi_comm=comm)

    om.printlog(f'my rank is {rank}')
    om.printlog(f'importing reference map {get_fname(map_path)}')
    ref_map = enmap.read_map(map_path)[0]
    ref_map[:,:] = 0.

    ndec, nra = ref_map.shape
    assert ndec % size == 0
    assert nra % size == 0

    nfill_dec, nfill_ra = np.floor(np.array(ref_map.shape) // size).astype(int)
    assert abs(nfill_dec) + abs(nfill_ra) >= 2
    ref_map[rank * nfill_dec:(rank + 1) * nfill_dec, rank * nfill_ra:(rank + 1) * nfill_ra] = 1.

    fig = enplot.plot(downgrade(ref_map, 16), ticks=15, colorbar=True)
    om.printlog(f'saving plot, rank {rank}')
    om.savefig(f'test_mpi_pixell_{rank}', mode='pixell', fig=fig)