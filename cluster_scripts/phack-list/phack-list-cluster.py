from fnl_pipe.util import OutputManager, get_fname, ChunkedMaskedReader
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut
from fnl_pipe.catalog import get_files

from pixell import enplot, enmap

import itertools
import time

import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()
node_name = MPI.Get_processor_name()

# from line_profiler import LineProfiler

mem_cap = 245 # GB; system memory less buffer for os and background
max_mem_per_proc = 40 # GB

time_per_iter = 50. # seconds


NTRIAL_NL = 1024

NTRIAL_FL = 32
NAVE_FL = 60
NITER_FL = 40

# NTRIAL_MC = 4096
# NFILE_MC = 16 # number of mpi tasks that generated the mc lists

NTRIAL_MC = 1024
NFILE_MC = 16 # number of mpi tasks that generated the mc lists


freq_str = '150'

plots = True

# data_path = '/gpfs/aroman/data/'
data_path = '/gpfs/aroman/data/'
data_alt = '/gpfs/aroman/data/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'
pipe_path = data_path + 'pipe/'

act_path = data_path + 'act_pub/'
# map_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_map_srcfree.fits' # public
# ivar_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_ivar.fits' # public
# beam_path = act_path + 'act_planck_dr5.01_s08s18_f150_daynight_beam.txt' # public beam
# nl_path = data_alt + f'desils/nl_desils_pub_{NTRIAL_NL}.npy'
# fl_path = data_alt + f'desils/fl_desils_pub_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'

# mc_list_base = data_alt + f'desils/desils_cmass_nmc_{NTRIAL_MC}'

map_path = act_path + f'act_dr5.01_s08s18_AA_f{freq_str}_daynight_map_srcfree.fits' # public
ivar_path = act_path + f'act_dr5.01_s08s18_AA_f{freq_str}_daynight_ivar.fits' # public
beam_path = act_path + f'act_planck_dr5.01_s08s18_f{freq_str}_daynight_beam.txt' # public beam

nl_path = data_path + f'desils/nl_desils_pub_actonly_f{freq_str}_{NTRIAL_NL}.npy'
fl_path = data_path + f'desils/fl_desils_pub_actonly_f{freq_str}_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'

mc_list_base = data_alt + f'desils/desils_cmass_actonly_f{freq_str}_nmc_{NTRIAL_MC}'

gal_mask_path = data_path + 'vr_source/desils/intersect_sdss_desi_mask.fits'

desils_v3_north = data_path + 'vr_source/desils/v03_desils_north_cmass.h5'
desils_v3_south = data_path + 'vr_source/desils/v03_desils_south_cmass.h5'


planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_enmap_path = mask_path + 'planck_foreground.npy'


# # zerr_max, do_lrg_cut, vr_width
# # do the lrg comparison
# params_lrg = [[0.05, True,  '1.0'],
#               [0.05, False, '1.0']]

# # determine the influence of zerr_max
# params_zerr = [[0.025, True,  '1.0'],
#                [0.05,  True,  '1.0'],
#                [0.06,  True,  '1.0'],
#                [0.075, True,  '1.0']]

# determine the influence of vr_width
# params_vr_width = [[0.05, True,  '0.75'],
#                [0.05,  True,  '1.0'],
#                [0.05,  True,  '1.25'],
#                [0.05, True,  '1.5']]

# params_vr_zerr_simul = [[0.0125, True,  '1.0'],
#                         [0.0125, True,  '1.25'],
#                         [0.0125, True,  '1.5'],
#                         [0.025, True,  '1.0'],
#                         [0.025, True,  '1.25'],
#                         [0.025, True,  '1.5'],
#                         [0.0375, True,  '1.0'],
#                         [0.0375, True,  '1.25'],
#                         [0.0375, True,  '1.5'],
#                         [0.05, True,  '1.0'],
#                         [0.05, True,  '1.25'],
#                         [0.05, True,  '1.5'],
#                         [0.0625, True,  '1.0'],
#                         [0.0625, True,  '1.25'],
#                         [0.0625, True,  '1.5'],
#                         [0.075, True,  '1.0'],
#                         [0.075, True,  '1.25'],
#                         [0.075, True,  '1.5'],]

# zerr_eps = 0.001
# params_vr_zerr_simul = [[0.025, True,  '0.25'],
#                         [0.025, True,  '0.75'],
#                         [0.025, True,  '1.0'],
#                         [0.025 + zerr_eps, True,  '0.25'],
#                         [0.025 + zerr_eps, True,  '0.75'],
#                         [0.025 + zerr_eps, True,  '1.0'],
#                         [0.025 - zerr_eps, True,  '0.25'],
#                         [0.025 - zerr_eps, True,  '0.75'],
#                         [0.025 - zerr_eps, True,  '1.0'],]

# zerr_eps = 0.00005
# # zerr_center = 0.0125
# zerr_center = 0.023
# params_vr_zerr_simul = [[zerr_center - delta_ze, True,  '0.75'] for delta_ze in np.arange(10) * zerr_eps ]


# truncated for testing
params_vr_zerr_simul = [[0.0375, True,  '1.0'],
                        [0.0375, True,  '1.25'],
                        [0.0375, True,  '1.5'],
                        [0.05, True,  '1.0'],
                        [0.05, True,  '1.25'],
                        [0.05, True,  '1.5'],
                        [0.0625, True,  '1.0'],
                        [0.0625, True,  '1.25'],
                        [0.0625, True,  '1.5'],
                        [0.075, True,  '1.0'],
                        [0.075, True,  '1.25'],
                        [0.075, True,  '1.5'],]


# zerr_grid = np.linspace(0.0125, 0.1, 20)
# vr_widths = ['0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0']
# do_cuts = [True,]

# zerr_grid = np.linspace(0.02, 0.03, 20)
# vr_widths = ['0.5', '0.75', '1.0', '1.25',]
# do_cuts = [True,]

zerr_grid = np.linspace(0.02, 0.03, 16)
vr_widths = ['1.0',]
do_cuts = [True,]

# class ParamSet:
#     def __init__(self, zerrs, vr_widths, do_cuts):
#         self.shape = np.array((len(zerrs), len(vr_widths), len(do_cuts)))


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
    max_procs = int(mem_cap / max_mem_per_proc)


if __name__ == "__main__":
    assert NFILE_MC % size == 0

    om = OutputManager(base_path='output', title='phack-mc-cluster', logs=['log',],
                       mpi_rank=rank, mpi_comm=comm)
    log = 'log'
    param_set = params_vr_zerr_simul

    assert len(param_set) > 0

    nfiles_per_task = NFILE_MC // size
    my_ifiles = rank * nfiles_per_task + np.arange(nfiles_per_task)
    my_ntrial_mc = NTRIAL_MC // NFILE_MC

    mc_lists = [mc_list_base + f'_rank_{ifile}.h5' for ifile in my_ifiles]

    printlog = om.printlog
    om.set_default_log(log)

    act_md = ACTMetadata(r_fkp=1.56, r_lwidth=0.62)

    act_pipe_150 = ACTPipe(map_path, ivar_path, beam_path, planck_enmap_path,
                           om, freq=150, metadata=act_md, plots=True)
    act_pipe_150.import_data()
    act_pipe_150.init_fkp()
    act_pipe_150.init_lweight()

    ref_map = act_pipe_150.map_t
    printlog(f'importing galaxy mask {get_fname(gal_mask_path)}', rank)
    gal_mask = enmap.read_map(gal_mask_path)

    desils_cat = DESILSCat(cat_north=desils_v3_north, cat_south=desils_v3_south)

    param_set = itertools.product(zerr_grid, do_cuts, vr_widths)

    nparams = len(list(param_set))
    # time_per_run = nparams * time_per_iter / 3600. # hours
    # printlog(f'phack-mc: predicted time for entire run: {time_per_run:2f} hours with {nparams} iterations')

    cut_labels = []
    alphas_opt = []

    param_set = itertools.product(zerr_grid, do_cuts, vr_widths)

    for zerr_max, do_cut, vr_width in param_set:
        t0 = time.time()
        cutstring = f'zerr_max {zerr_max:.5f}, do_lrg_cut {do_cut}, vr_width {vr_width}'
        printlog(cutstring, rank)
        
        zerr_cut = ZerrCut(zerr_max)
        if do_cut:
            north_cut = AndCut([LRGNorthCut(), zerr_cut])
            south_cut = AndCut([LRGSouthCut(), zerr_cut])
        else:
            north_cut = zerr_cut
            south_cut = zerr_cut

        gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)
        gal_pipe.import_data()
        gal_pipe.make_vr_list()
        row_mask = gal_pipe.cut_inbounds_mask

        # iterate over files assigned to this mpi task and append alphas to alphas_mpi
        alphas_mpi = np.array([], dtype=float)
        for mc_list in mc_lists:
            br = ChunkedMaskedReader(mc_list, 1024 * 32, row_mask, bufname='t_mc')

            cross_pipe = CMBxGalPipe(act_pipe_150, gal_pipe, gal_mask, output_manager=om)

            printlog('importing nl', rank)
            cross_pipe.import_nl(nl_path)
            printlog('done', rank)
            printlog('importing fl', rank)
            cross_pipe.import_fl(fl_path)
            printlog('done', rank)

            printlog('phack-mc: computing estimator', rank)
            cross_pipe.compute_estimator(ntrial_mc=my_ntrial_mc, buffered_reader=br)
            printlog('phack-mc: done', rank)
            
            alphas = cross_pipe.alphas
            alphas_mpi = np.concatenate((alphas_mpi, alphas))

        assert len(alphas_mpi) == NTRIAL_MC // size, f'rank {rank}: {len(alphas_mpi)}, {NTRIAL_MC // size}'

        alphas_gather = None
        if rank == 0: alphas_gather = np.empty((size, NTRIAL_MC // size), dtype=np.float64)

        comm.Gather(alphas_mpi, alphas_gather, root=0)

        # only compute the estimator for rank 0
        if rank == 0:
            alpha_ksz = cross_pipe.a_ksz_unnorm / np.std(alphas_gather.flatten())
            printlog(f'MC ksz estimator: {alpha_ksz:.4e}')
            printlog(f'ntrial_mc combined: {NTRIAL_MC}')
            alphas_opt.append(alpha_ksz)

        comm.Barrier()

        dt = time.time() - t0
        printlog(f'time of iteration: {dt:.2f} s', rank)


    if rank == 0:
        param_set = itertools.product(zerr_grid, do_cuts, vr_widths)

        printlog('zerr_max, do_lrg_cut, vr_width, alpha_mc')
        for alpha, params in zip(alphas_opt, param_set):
            zerr_max, do_cut, vr_width = params
            printlog(f'{zerr_max:.7f}, {do_cut:b}, {vr_width}, {alpha:.3e}')
