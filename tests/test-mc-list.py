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
nl_path = f'data/nl_desils_pub_{NTRIAL_NL}.npy'
fl_path = f'data/fl_desils_pub_nfl_{NTRIAL_FL}_nave_{NAVE_FL}_niter_{NITER_FL}.npy'


gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'


# gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'
# gal_mask_path = data_path + 'vr_source/desi_ls/intersect_sdss_desi_mask.h5'
desils_v3_north = data_path + 'vr_source/desils/v03_desils_north_cmass.h5'
desils_v3_south = data_path + 'vr_source/desils/v03_desils_south_cmass.h5'

mc_list_out = '/data/aroman/tmp/test_mc.h5'

planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_enmap_path = mask_path + 'planck_foreground.npy'


zerr_max = 0.05
vr_width = '1.0'


if __name__ == "__main__":
    om = OutputManager(base_path='output', title='test-mc-list', logs=['log'])
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

    desils_cat = DESILSCat(cat_north=desils_v3_north, cat_south=desils_v3_south)
    
    zerr_cut = ZerrCut(zerr_max)

    north_cut = AndCut([LRGNorthCut(), zerr_cut])
    south_cut = AndCut([LRGSouthCut(), zerr_cut])

    gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)
    gal_pipe.import_data()
    gal_pipe.make_vr_list()
    row_mask = gal_pipe.cut_inbounds_mask

    size_gb = gal_pipe.ngal_inbounds * NTRIAL_MC * 4 * 1e-9

    print(f'expected buffer size: {size_gb:2f} GB')

    ngal = gal_pipe.ngal_inbounds
    ngal_in = row_mask.sum()
    assert ngal_in == gal_pipe.ngal_in

    # write random data to tmp file
    cw = ChunkedTransposeWriter(mc_list_out, chunk_size=32, nrow=ngal, ncol=NTRIAL_MC, bufname='t_mc')

    # populate random data array (float32) and write to 
    rand_data = np.empty((NTRIAL_MC, ngal), dtype=np.float32)
    for itrial in range(NTRIAL_MC):
        this_col = np.random.rand(ngal)
        cw.add_row(this_col)
        rand_data[itrial] = this_col
        print(f'finished generating itrial {itrial} of {NTRIAL_MC}')

    cw.finalize()
    rand_data = rand_data.T
    assert cw.complete

    # extract the expected masked array and delete original (space)
    masked_chunk = rand_data[row_mask]
    del rand_data

    data_read_direct = None

    with h5py.File(mc_list_out, 'r') as h5file:
        dset = h5file['t_mc']
        data_read_direct = dset[:][row_mask]

    assert data_read_direct is not None

    cr = ChunkedMaskedReader(mc_list_out, 1024 * 32, row_mask, bufname='t_mc')

    print(f'test chunked reader nchunk {cr.nchunk}')

    data_read_hnc = None

    while cr.has_next_chunk:
        print(f'fetching chunk {cr.ichunk}')
        inds, nc = cr.get_next_chunk()
        if data_read_hnc is None:
            data_read_hnc = nc.astype(np.float32)
        else:
            data_read_hnc = np.concatenate((data_read_hnc, nc.astype(np.float32)))

    assert data_read_hnc.shape == data_read_direct.shape

    plt.figure(dpi=300)
    plt.title('T_hp MC list column (galaxy varies)')
    plt.scatter(np.arange(ngal_in), data_read_hnc[:, NTRIAL_MC//2], s=1)
    plt.ylabel('T_hp (mK)')
    plt.xlabel('igal')
    om.savefig('reconstructed_thp_column.png')

    plt.figure(dpi=300)
    plt.title('T_hp MC list row (itrial_mc varies)')
    plt.scatter(np.arange(NTRIAL_MC), data_read_hnc[ngal_in//2], s=1, label='chunked_read')
    plt.scatter(np.arange(NTRIAL_MC), data_read_direct[ngal_in//2], s=1, label='direct_read')
    plt.legend()
    plt.ylabel('T_hp (mK)')
    plt.xlabel('itrial_mc')
    om.savefig('reconstructed_thp_row.png')

    assert data_read_hnc.dtype == np.float32
    assert fequal(data_read_hnc, masked_chunk)


    # cross_pipe = CMBxGalPipe(act_pipe_150, gal_pipe, gal_mask, output_manager=om)
    # cross_pipe.import_nl(nl_path)
    # cross_pipe.import_fl(fl_path)

    # cross_pipe.write_mc_list(ntrial_mc=NTRIAL_MC, outpath=mc_list_out)
