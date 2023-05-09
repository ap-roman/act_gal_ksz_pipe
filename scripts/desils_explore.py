from fnl_pipe.util import OutputManager, get_fname, ChunkedMaskedReader
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut
from fnl_pipe.catalog import get_files

from pixell import enplot, enmap

import matplotlib.pyplot as plt

import itertools
import time

import numpy as np

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


# gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'
gal_mask_path = data_path + 'vr_source/desils/intersect_sdss_desi_mask.fits'


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

# reconstructs a boolean galaxy mask from chunked t_hp mc list data
def reconstruct_mask(chunked_reader, gal_pipe, ref_map, thresh=1e-6, downgrade=10):
    accum_map = ref_map.copy()
    accum_map[:,:] = 0.

    dec_inds = gal_pipe.dec_inds
    ra_inds = gal_pipe.ra_inds

    # print(f'nchunk: {chunked_reader.nchunk}')
    while chunked_reader.has_next_chunk:
        # print(f'ichunk {chunked_reader.ichunk}')
        chunk_inds, t_hp = chunked_reader.get_next_chunk()
        this_decs = dec_inds[chunk_inds[0]:chunk_inds[1]]
        this_ras = ra_inds[chunk_inds[0]:chunk_inds[1]]
        accum_map[this_decs, this_ras] += np.abs(t_hp.sum(axis=1))

    accum_map[:,:] = (np.abs(accum_map) >= thresh)
    accum_map = enmap.downgrade(accum_map, downgrade)
    accum_map[:,:] = (accum_map >= thresh)
    accum_map = enmap.upgrade(accum_map, downgrade)

    return accum_map



# A script to plot various aspects of the desils data and implicitly probe the
# chunked writer/reader classes
if __name__ == "__main__":
    om = OutputManager(base_path='output', title='desils-explore', logs=['log',])

    printlog = om.printlog

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

    gal_mask = enmap.read_map(gal_mask_path)

    # zerr_grid = np.linspace(0.025, 0.1, 8)
    # vr_widths = ['0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0']
    # do_cuts = [True, False]

    cut_labels = []
    alphas_opt = []

    zerr_max = 0.1
    do_cut = 1
    vr_width = '1.0'

    cutstring = f'zerr_max {zerr_max:.3f}, do_lrg_cut {do_cut}, vr_width {vr_width}'
    printlog(cutstring)
    
    zerr_cut = ZerrCut(zerr_max)

    north_lrgcut = AndCut([LRGNorthCut(), zerr_cut])
    south_lrgcut = AndCut([LRGSouthCut(), zerr_cut])

    north_nolrgcut = zerr_cut
    south_nolrgcut = zerr_cut

    lrg_cut = [north_lrgcut, south_lrgcut]
    nolrg_cut = [north_nolrgcut, south_nolrgcut]

    gal_pipe_lrg = desils_cat.get_subcat(lrg_cut, ref_map, vr_width)
    gal_pipe_lrg.import_data()
    gal_pipe_lrg.make_vr_list()
    row_mask_lrg = gal_pipe_lrg.cut_inbounds_mask

    br_lrg = ChunkedMaskedReader(mc_list, 1024 * 32, row_mask_lrg, bufname='t_mc')


    gal_pipe_nolrg = desils_cat.get_subcat(nolrg_cut, ref_map, vr_width)
    gal_pipe_nolrg.import_data()
    gal_pipe_nolrg.make_vr_list()
    row_mask_nolrg = gal_pipe_nolrg.cut_inbounds_mask

    br_nolrg = ChunkedMaskedReader(mc_list, 1024 * 32, row_mask_nolrg, bufname='t_mc')

    plt.figure(dpi=300)
    plt.hist(gal_pipe_lrg.get_desils_field('zerr'), bins=100, cumulative=True, density=True,
                  label='lrg')
    plt.hist(gal_pipe_nolrg.get_desils_field('zerr'), bins=100, cumulative=True, density=True,
                  label='all galaxies')
    plt.legend()
    plt.xlabel('zerr')
    om.savefig('zerr_hist.png')
    plt.close()

    # make boolean mask from written chunk reader file
    effective_mask_lrg = reconstruct_mask(br_lrg, gal_pipe_lrg, ref_map)
    effective_mask_nolrg = reconstruct_mask(br_nolrg, gal_pipe_nolrg, ref_map)

    # fig1 = enplot.plot(enmap.downgrade(effective_mask_lrg, 16), ticks=15, colorbar=True)
    fig1 = enplot.plot(enmap.downgrade(effective_mask_lrg, 16), ticks=15, colorbar=True)
    om.savefig('reconstructed_lrg_mask', mode='pixell', fig=fig1)

    fig2 = enplot.plot(enmap.downgrade(effective_mask_nolrg, 16), ticks=15, colorbar=True)
    om.savefig('reconstructed_nolrg_mask', mode='pixell', fig=fig2)

    fig3 = enplot.plot(enmap.downgrade(gal_mask, 16), ticks=15, colorbar=True)
    om.savefig('desils_sdss_intersect_mask', mode='pixell', fig=fig3)