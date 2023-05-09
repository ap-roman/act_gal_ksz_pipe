from fnl_pipe.util import OutputManager, get_fname, ChunkedMaskedReader
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut
from fnl_pipe.catalog import get_files

from pixell import enplot, enmap

import itertools
import time

import numpy as np

# from line_profiler import LineProfiler

mem_cap = 245 # GB; system memory less buffer for os and background
max_mem_per_proc = 40 # GB

time_per_iter = 50. # seconds


NTRIAL_NL = 1024

NTRIAL_FL = 32
NAVE_FL = 60
NITER_FL = 40

NTRIAL_MC = 1024

plots = True

data_path = '/home/aroman/data/'
base_data_path = '/data/aroman/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'
pipe_path = data_path + 'pipe/'

# act_path = data_path + 'act/'
# map_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits' # private
# ivar_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_ivar.fits' # private
# beam_path = act_path + 'beam_f150_daynight.txt' # proprietary beam file
# # nl_path = f'data/nl_desils_{NTRIAL_NL}.npy'
# # fl_path = f'data/fl_desils_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'
# nl_path = f'data/nl_{NTRIAL_NL}.npy'
# fl_path = f'data/fl_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'


act_path = data_path + 'act_pub/'
map_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_map_srcfree.fits' # public
ivar_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_ivar.fits' # public
beam_path = act_path + 'act_planck_dr5.01_s08s18_f150_daynight_beam.txt' # public beam
nl_path = f'data/nl_desils_pub_{NTRIAL_NL}.npy'
fl_path = f'data/fl_desils_pub_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'


gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'


# gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'
# gal_mask_path = data_path + 'vr_source/desi_ls/intersect_sdss_desi_mask.h5'
desils_v3_north = data_path + 'vr_source/desils/v03_desils_north_cmass.h5'
desils_v3_south = data_path + 'vr_source/desils/v03_desils_south_cmass.h5'


planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_enmap_path = mask_path + 'planck_foreground.npy'

# catalog_path = data_path + 'vr_summaries/v01_sdss_cmass_north.h5'
catalog_base = data_path + 'vr_summaries/desils/'
catalog_files = get_files(catalog_base)

mc_list = base_data_path + f'vr_source/desils/desils_cmass_nmc_{NTRIAL_MC}.h5'


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

zerr_grid = np.linspace(0.022, 0.024, 10)
vr_widths = ['0.5', '0.75', '1.0', '1.25',]
do_cuts = [True,]


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

def do_study(om, param_set, log=None):
    assert len(param_set) > 0
    printlog = om.printlog
    om.set_default_log(log)

    act_md = ACTMetadata(r_fkp=1.56, r_lwidth=0.62)

    act_pipe_150 = ACTPipe(map_path, ivar_path, beam_path, planck_enmap_path,
                           om, freq=150, metadata=act_md, plots=True)
    act_pipe_150.import_data()
    act_pipe_150.init_fkp()
    act_pipe_150.init_lweight()

    ref_map = act_pipe_150.map_t
    printlog(f'importing galaxy mask {get_fname(gal_mask_path)}')
    gal_mask = enmap.read_map(gal_mask_path)

    printlog(f'importing catalog files {catalog_files}')

    desils_cat = DESILSCat(cat_north=desils_v3_north, cat_south=desils_v3_south)

    param_set = itertools.product(zerr_grid, do_cuts, vr_widths)

    nparams = len(list(param_set))
    time_per_run = nparams * time_per_iter / 3600. # hours
    printlog(f'phack-mc: predicted time for entire run: {time_per_run:2f} hours with {nparams} iterations')

    cut_labels = []
    alphas_opt = []

    param_set = itertools.product(zerr_grid, do_cuts, vr_widths)

    for zerr_max, do_cut, vr_width in param_set:
        t0 = time.time()
        cutstring = f'zerr_max {zerr_max:.5f}, do_lrg_cut {do_cut}, vr_width {vr_width}'
        printlog(cutstring)
        
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

        br = ChunkedMaskedReader(mc_list, 1024 * 32, row_mask, bufname='t_mc')

        cross_pipe = CMBxGalPipe(act_pipe_150, gal_pipe, gal_mask, output_manager=om)

        printlog('phack-mc: importing nl')
        cross_pipe.import_nl(nl_path)
        printlog('phack-mc: done')
        cross_pipe.import_fl(fl_path)

        printlog('phack-mc: computing estimator')
        cross_pipe.compute_estimator(ntrial_mc=NTRIAL_MC, buffered_reader=br)
        printlog('phack-mc: done')
        
        alphas = cross_pipe.alphas

        alpha_ksz = cross_pipe.a_ksz_unnorm / np.std(alphas)
        printlog(f'MC ksz estimator: {alpha_ksz:.4e}')
        printlog(f'ntrial_mc combined: {NTRIAL_MC}')
        alphas_opt.append(alpha_ksz)
        dt = time.time() - t0
        printlog(f'time of iteration: {dt:.2f} s')

    param_set = itertools.product(zerr_grid, do_cuts, vr_widths)

    printlog('zerr_max, do_lrg_cut, vr_width, alpha_mc')
    for alpha, params in zip(alphas_opt, param_set):
        zerr_max, do_cut, vr_width = params
        printlog(f'{zerr_max:.7f}, {do_cut:b}, {vr_width}, {alpha:.3e}')


if __name__ == "__main__":
    om = OutputManager(base_path='output', title='phack-mc', logs=['log',])

    do_study(om, params_vr_zerr_simul, 'log')
