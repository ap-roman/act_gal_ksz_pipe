from mpi4py import MPI

import numpy as np

from fnl_pipe.util import OutputManager
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.pipe import GalPipe

from pixell import enplot, enmap

NTRIAL_NL = 1024

NTRIAL_FL = 32
NAVE_FL = 60
NITER_FL = 40

NTRIAL_MC = 1024

plots = True
make_nl = True
make_fl = True
compute_estimator = False


data_path = '/home/aroman/data/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'
pipe_path = data_path + 'pipe/'

# act_path = data_path + 'act/'
# map_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits' # private
# ivar_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_ivar.fits' # private
# beam_path = act_path + 'beam_f150_daynight.txt' # proprietary beam file
# nl_path = f'data/nl_mpi_desils_{NTRIAL_NL}.npy'
# fl_path = f'data/fl_mpi_desils_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'

act_path = data_path + 'act_pub/'
map_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_map_srcfree.fits' # public
ivar_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_ivar.fits' # public
beam_path = act_path + 'act_planck_dr5.01_s08s18_f150_daynight_beam.txt' # public beam
nl_path = f'data/nl_desils_pub_{NTRIAL_NL}.npy'
fl_path = f'data/fl_desils_pub_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'

# gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'
gal_mask_path = data_path + 'vr_source/desils/intersect_sdss_desi_mask.fits'

planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_enmap_path = mask_path + 'planck_foreground.npy'

catalog_path = data_path + 'vr_summaries/desils/v02_south_lowz.h5'
# catalog_path = data_path + 'vr_summaries/v01_sdss_cmass_north.h5'

kszpipe_path = data_path + 'kszpipe/sdss_dr12/cmass_north/'
kszpipe_cosmo_path = kszpipe_path + 'cosmology.pkl'
kszpipe_box_path = kszpipe_path + 'bounding_box.pkl'
kszpipe_d0_path = kszpipe_path + 'delta0_DR12v5_CMASS_North.h5'

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

mem_cap = 245 # GB; system memory less buffer for os and background
max_mem_per_proc = 40 # GB


def make_nl_mpi(cross_pipe):
    cross_pipe.make_nl(ntrial_nl=NTRIAL_NL, nl_path=None, plots=True)
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

    ntrial_nl = NTRIAL_NL
    ntrial_fl = NTRIAL_FL

    max_procs = int(mem_cap / max_mem_per_proc)

    assert size <= max_procs
    if make_nl: assert ntrial_nl % size == 0
    if make_fl: assert ntrial_fl % size == 0


    om = OutputManager(base_path='output', title='make-significance-mpi', logs=['log',],
                       mpi_rank=rank, mpi_comm=comm)
    act_md = ACTMetadata(r_fkp=1.56, r_lwidth=0.62)

    act_pipe_150 = ACTPipe(map_path, ivar_path, beam_path, planck_enmap_path,
                           om, freq=150, metadata=act_md, plots=True)
    act_pipe_150.import_data()
    act_pipe_150.init_fkp()
    act_pipe_150.init_lweight()

    ref_map = act_pipe_150.map_t

    gal_pipe = GalPipe(catalog_path, act_pipe_150.map_t, diag_plots=True)
    gal_pipe.import_data()
    gal_pipe.make_vr_list()

    gal_mask = enmap.read_map(gal_mask_path)

    if plots:
        fig = enplot.plot(enmap.downgrade(gal_mask, 16), ticks=15, colorbar=True)
        om.savefig('galaxy_mask', mode='pixell', fig=fig)

    cross_pipe = CMBxGalPipe(act_pipe_150, gal_pipe, gal_mask, output_manager=om)

    if make_nl:
        nl_tilde = make_nl_mpi(cross_pipe)
        cross_pipe.update_nl(nl_tilde, nl_path=nl_path, ntrial_nl=ntrial_nl)

    # synchronize state across all objects    
    cross_pipe.import_nl(nl_path)

    if make_fl:
        cross_pipe.make_fl_iter(ntrial_fl=NTRIAL_FL, nave_fl=NAVE_FL, niter=NITER_FL, fl_path=fl_path, plots=True)

    cross_pipe.import_fl(fl_path)

    if compute_estimator:
        alphas = compute_estimator_mpi(cross_pipe, ntrial_mc=NTRIAL_MC)
        if rank == 0:
            alpha_ksz = cross_pipe.a_ksz_unnorm / np.std(alphas)
            om.printlog(f'MC ksz estimator: {alpha_ksz:.4e}')
            om.printlog(f'ntrial_mc combined: {NTRIAL_MC}')