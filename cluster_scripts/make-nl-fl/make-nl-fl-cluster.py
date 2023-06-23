import numpy as np

import os
# print(os.environ['CONDA_DEFAULT_ENV'])
# print(os.environ["CONDA_PREFIX"])

import fnl_pipe
from fnl_pipe.util import OutputManager
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut

from pixell import enplot, enmap

import mpi4py
from mpi4py import MPI

NTRIAL_NL = 1024

NTRIAL_FL = 32
NAVE_FL = 60
NITER_FL = 40

NTRIAL_MC = 1024

plots = False
make_nl = True
make_fl = True
compute_estimator = False

freq_str = '150'

data_path = '/gpfs/aroman/data/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'
pipe_path = data_path + 'pipe/'

act_path = data_path + 'act_pub/'
map_path = act_path + f'act_dr5.01_s08s18_AA_f{freq_str}_daynight_map_srcfree.fits' # public
ivar_path = act_path + f'act_dr5.01_s08s18_AA_f{freq_str}_daynight_ivar.fits' # public
beam_path = act_path + f'act_planck_dr5.01_s08s18_f{freq_str}_daynight_beam.txt' # public beam
nl_path = data_path + f'desils/nl_desils_pub_actonly_f{freq_str}_{NTRIAL_NL}.npy'
fl_path = data_path + f'desils/fl_desils_pub_actonly_f{freq_str}_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'

gal_mask_path = data_path + 'vr_source/desils/intersect_sdss_desi_mask.fits'

desils_v3_north = data_path + 'vr_source/desils/v03_desils_north_cmass.h5'
desils_v3_south = data_path + 'vr_source/desils/v03_desils_south_cmass.h5'

mc_list_base = data_path + f'desils/desils_cmass_actonly_f{freq_str}_nmc_{NTRIAL_MC}'

planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_enmap_path = mask_path + 'planck_foreground.npy'

catalog_path = data_path + 'vr_summaries/desils/v02_south_lowz.h5'
# catalog_path = data_path + 'vr_summaries/v01_sdss_cmass_north.h5'

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

mem_cap = 245 # GB; system memory less buffer for os and background
max_mem_per_proc = 40 # GB

zerr_max = 0.1
vr_width = '1.0'


def make_nl_mpi(cross_pipe, ntrial_nl):
    cross_pipe.make_nl(ntrial_nl=ntrial_nl, nl_path=None, plots=True)
    my_nl = cross_pipe.nl_tilde

    recvbuf = None
    if rank == 0:
        recvbuf = np.empty((size, len(my_nl)), dtype=np.float64)

    comm.Gather(my_nl, recvbuf, root=0)
    if rank == 0:
        result = recvbuf.sum(axis=0) / size
        return result
    else:
        return None


def compute_estimator_mpi(cross_pipe, ntrial_mc):
    ntrial_per = ntrial_mc // size
    cross_pipe.compute_estimator(ntrial_mc=ntrial_per)
    my_alphas = cross_pipe.alphas

    recvbuf = None
    if rank == 0:
        recvbuf = np.empty((size, len(my_alphas)), dtype=np.float64)

    comm.Gather(my_alphas, recvbuf, root=0)
    if rank == 0:
        return recvbuf.flatten()
    else:
        return None


if __name__ == "__main__":
    assert make_nl or make_fl or compute_estimator # we need to do something

    print(f'my rank is {rank} of size {size}, on node {MPI.Get_processor_name()}, comm {comm.Get_name()}')
    
    max_procs = int(mem_cap / max_mem_per_proc)
    # assert size <= max_procs

    ntrial_nl = NTRIAL_NL
    ntrial_fl = NTRIAL_FL

    if make_nl: assert ntrial_nl % size == 0
    my_ntrial_nl = ntrial_nl // size


    om = OutputManager(base_path='output', title='make-nl-fl-cluster', logs=['log',],
                       mpi_rank=rank, mpi_comm=comm)

    act_md = ACTMetadata(r_fkp=1.56, r_lwidth=0.62)

    act_pipe_150 = ACTPipe(map_path, ivar_path, beam_path, planck_enmap_path,
                           om, freq=150, metadata=act_md, plots=True)
    act_pipe_150.import_data()
    act_pipe_150.init_fkp()
    act_pipe_150.init_lweight()

    ref_map = act_pipe_150.map_t

    desils_cat = DESILSCat(cat_north=desils_v3_north, cat_south=desils_v3_south)
    
    north_cut = NullCut()
    south_cut = NullCut()

    gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)
    gal_pipe.import_data()
    gal_pipe.make_vr_list()

    gal_mask = enmap.read_map(gal_mask_path)

    if plots:
        fig = enplot.plot(enmap.downgrade(gal_mask, 16), ticks=15, colorbar=True)
        om.savefig('galaxy_mask', mode='pixell', fig=fig)

    cross_pipe = CMBxGalPipe(act_pipe_150, gal_pipe, gal_mask, output_manager=om)

    if make_nl:
        nl_tilde = make_nl_mpi(cross_pipe, my_ntrial_nl)
        if rank == 0:
            cross_pipe.update_nl(nl_tilde, nl_path=nl_path, ntrial_nl=ntrial_nl)

    comm.Barrier()

    # synchronize state across all ranks    
    cross_pipe.import_nl(nl_path)

    if make_fl:
        cross_pipe.make_fl_iter(ntrial_fl=NTRIAL_FL, nave_fl=NAVE_FL, niter=NITER_FL, fl_path=fl_path, plots=True, comm=comm)

    comm.Barrier()

    cross_pipe.import_fl(fl_path)
