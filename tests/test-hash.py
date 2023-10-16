from fnl_pipe.util import OutputManager, get_fname, fequal, ChunkedMaskedReader, ChunkedTransposeWriter
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut
from fnl_pipe.catalog import get_files

from pixell import enplot, enmap

import h5py
import matplotlib.pyplot as plt

import itertools

import numpy as np

from fnl_pipe.gal_cmb import CMBxGalHash

NTRIAL_NL = 1024

NTRIAL_FL = 32
NAVE_FL = 60
NITER_FL = 40

NTRIAL_MC = 128

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
beam_path2 = act_path + 'act_planck_dr5.01_s08s18_f090_daynight_beam.txt' # public beam
nl_path = f'data/nl_desils_pub_{NTRIAL_NL}.npy'
fl_path = f'data/fl_desils_pub_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'


gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'


# gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'
# gal_mask_path = data_path + 'vr_source/desi_ls/intersect_sdss_desi_mask.h5'
desils_v3_north = data_path + 'vr_source/desils/v03_desils_north_cmass.h5'
desils_v3_south = data_path + 'vr_source/desils/v03_desils_south_cmass.h5'

mc_list_out = '/data/aroman/tmp/test_mc.h5'

planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_mask_path = mask_path + 'planck_foreground.npy'

zerr_max = 0.0225
vr_width = '1.0'
do_cut = True

LMAX = 12000
freq = 150


def init_ap(act_pipe):
    act_pipe.import_data()
    act_pipe.init_fkp(plots=True)
    act_pipe.init_lweight(plots=True)


if __name__ == "__main__":
    om = OutputManager(base_path='output', title='test-hash', logs=['log',])
    printlog = om.printlog

    act_md = ACTMetadata(r_fkp=1.56, r_lwidth=0.62)

    act_pipe = ACTPipe(map_path, ivar_path, beam_path, planck_mask_path, om,
                       lmax=LMAX, freq=150, metadata=act_md, plots=True)
    init_ap(act_pipe)

    ap_copy = ACTPipe(map_path, ivar_path, beam_path, planck_mask_path, om,
                       lmax=LMAX, freq=150, metadata=act_md, plots=True)
    init_ap(ap_copy)

    ap2 = ACTPipe(map_path, ivar_path, beam_path2, planck_mask_path, om,
                       lmax=LMAX, freq=150, metadata=act_md, plots=True)
    init_ap(ap2)

    ref_map = act_pipe.map_t

    printlog(f'importing galaxy mask {get_fname(gal_mask_path)}')
    gal_mask = enmap.read_map(gal_mask_path)

    zerr_cut = ZerrCut(zerr_max)
    if do_cut:
        north_cut = AndCut([LRGNorthCut(), zerr_cut])
        south_cut = AndCut([LRGSouthCut(), zerr_cut])
    else:
        north_cut = zerr_cut
        south_cut = zerr_cut

    desils_cat = DESILSCat(cat_north=desils_v3_north, cat_south=desils_v3_south)
    gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)

    cutstring = f'zerr_max {zerr_max:.3f}, do_lrg_cut {do_cut}, vr_width {vr_width}'
    printlog(cutstring)

    # printlog('getting gal pipe')
    gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)
    # printlog('done')

    # printlog('importing data')
    gal_pipe.import_data()
    # printlog('done')

    # cross_pipe = CMBxGalPipe(act_pipe, gal_pipe, gal_mask, output_manager=om)

    printlog('testing CMBxGalHash agreement...')
    ch1 = CMBxGalHash(cmb_pipe=act_pipe, gal_pipe=gal_pipe, gal_mask=gal_mask)
    ch2 = CMBxGalHash(cmb_pipe=ap_copy,  gal_pipe=gal_pipe, gal_mask=gal_mask)
    
    # for key in ch1._hashes.keys():
    #     print(f'{key} {ch1._hashes[key]} {ch2._hashes[key]}')

    # dl_rms = np.std(act_pipe.l_weight - ap_copy.l_weight)

    # plt.figure(dpi=300, facecolor='w')
    # plt.title('lweight difference, same inputs')
    # plt.plot(act_pipe.l_weight - ap_copy.l_weight)
    # plt.xlabel(r'$l$')
    # plt.ylabel('delta lweight')
    # om.savefig('delta_lweight.png')

    # print(f'rms variation in l_weight: {dl_rms}')
    
    assert ch1 == ch2
    printlog('PASS')

    printlog('testing CMBxGalHash disagreement...')
    ch1 = CMBxGalHash(cmb_pipe=act_pipe, gal_pipe=gal_pipe, gal_mask=gal_mask)
    ch2 = CMBxGalHash(cmb_pipe=ap2,      gal_pipe=gal_pipe, gal_mask=gal_mask)
    assert ch1 != ch2
    printlog('PASS')

    printlog('testing CMBxGalHash single value disagreement...')
    ch1 = CMBxGalHash(cmb_pipe=act_pipe, gal_pipe=gal_pipe, gal_mask=gal_mask)
    act_pipe.map_t[0,0] = 2 * act_pipe.map_t[0,0] + 1.
    ch2 = CMBxGalHash(cmb_pipe=act_pipe, gal_pipe=gal_pipe, gal_mask=gal_mask)
    assert ch1 != ch2
    printlog('PASS')

    printlog('testing CMBxGalHash pickled save/load agreement...')
    ch1.save('tmp/hash_test.pck')
    ch1_load = CMBxGalHash.load('tmp/hash_test.pck')
    assert ch1 == ch1_load
    printlog('PASS')