from fnl_pipe.util import OutputManager, get_fname
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut
from fnl_pipe.catalog import get_files

from pixell import enplot, enmap
import itertools
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

mem_cap = 245 # GB; system memory less buffer for os and background
max_mem_per_proc = 40 # GB


NTRIAL_NL = 1024

NTRIAL_FL = 32
NAVE_FL = 60
NITER_FL = 40

NTRIAL_MC = 256

plots = True

data_path = '/home/aroman/data/'
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


# zerr_max, do_lrg_cut, vr_width
# do the lrg comparison
params_lrg = [[0.05, True,  '1.0'],
              [0.05, False, '1.0']]

# determine the influence of zerr_max
params_zerr = [[0.025, True,  '1.0'],
               [0.05,  True,  '1.0'],
               [0.06,  True,  '1.0'],
               [0.075, True,  '1.0']]

# determine the influence of vr_width
# params_vr_width = [[0.05, True,  '0.75'],
#                [0.05,  True,  '1.0'],
#                [0.05,  True,  '1.25'],
#                [0.05, True,  '1.5']]
params_vr_width = [[0.025, True,  '1.25'],]

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
    # zerr_grid = np.linspace(0.025, 0.1, 8)
    # vr_widths = ['0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0']
    # do_cuts = [True, False]

    cut_labels = []
    alphas_opt = []

    for zerr_max, do_cut, vr_width in param_set:
        cutstring = f'zerr_max {zerr_max:.3f}, do_lrg_cut {do_cut}, vr_width {vr_width}'
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

        cross_pipe = CMBxGalPipe(act_pipe_150, gal_pipe, gal_mask, output_manager=om)
        cross_pipe.import_nl(nl_path)
        cross_pipe.import_fl(fl_path)

        alphas = compute_estimator_mpi(cross_pipe, ntrial_mc=NTRIAL_MC)

        if rank == 0:
            alpha_ksz = cross_pipe.a_ksz_unnorm / np.std(alphas)
            printlog(f'MC ksz estimator: {alpha_ksz:.4e}')
            printlog(f'ntrial_mc combined: {NTRIAL_MC}')
            alphas_opt.append(alpha_ksz)

    if rank == 0:
        printlog('zerr_max, do_lrg_cut, vr_width, alpha_bs_2')
        for alpha, params in zip(alphas_opt, param_set):
            zerr_max, do_cut, vr_width = params
            printlog(f'{zerr_max:.3f}, {do_cut:b}, {vr_width}, {alpha:.3e}')


if __name__ == "__main__":
    max_procs = int(mem_cap / max_mem_per_proc)
    assert size <= max_procs
    assert NTRIAL_MC % size == 0

    om = OutputManager(base_path='output', title='phack-mc', logs=['log-lrg', 'log-zerr', 'log-vr-width'],
                       mpi_rank=rank, mpi_comm=comm)

    assert len(sys.argv) == 2
    config_file = sys.argv[1]
    printlog('got config file ' + config_file)
    config_dict = get_yaml_dict(config_file)
    local_dict = locals()
    printlog('dumping config')
    for key, value in config_dict.items():
        printlog(key, value)
        local_dict[key] = value
    printlog('################## DONE ##################')



    zerr_grid = 
    do_cuts = 
    vr_widths = "1.0"
    param_set = list(itertools.product(zerr_grid, do_cuts, vr_widths))


    # do_study(om, params_lrg, 'log-lrg')
    # do_study(om, params_zerr, 'log-zerr')
    do_study(om, param_set, 'log-vr-width')
