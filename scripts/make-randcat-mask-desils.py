from pixell import enmap, enplot
from fnl_pipe.util import bounds_check, get_fname, OutputManager
import h5py
import numpy as np

data_path = '/home/aroman/data/'

act_path = data_path + 'act/'
map_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits' # private, reference map


desi_base = data_path + 'vr_source/desi_ls/'

cat_base = 'desils_randcat_'
cat_north_path = desi_base + cat_base + 'north.h5'
cat_south_path = desi_base + cat_base + 'south.h5'

mask_out_path = desi_base + 'desi_mask.fits'
intersect_out_path = desi_base + 'intersect_sdss_desi_mask.fits'

sdss_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'

# cats = [🐈, 🐱, 😹, 😻]
cats = [cat_north_path, cat_south_path]

grid_downsample = 10 # resolution downgrade factor for ACT map -> 
plot_downsample = 16 # a further downsample factor for plots

# max_ram = 16 * 1024 * 1024 * 1024 # bytes

plots = True

if __name__ == "__main__":
    # determine gridding and resolution 
    ref_t = enmap.read_map(map_path)[0]

    for dim in ref_t.shape:
        assert dim % grid_downsample == 0
    
    mask = ref_t.downgrade(grid_downsample)
    mask[:,:] = 0.

    sdss_mask = enmap.read_map(sdss_mask_path)
    # print(ref_t.wcs, sdss_mask.wcs, mask.wcs)
    sdss_mask.wcs = ref_t.wcs # force wcs?
    assert sdss_mask.wcs == ref_t.wcs

    for cat in cats:
        with h5py.File(cat) as cat_file:
            print(f'processing random catalog {get_fname(cat)}')
            ras = cat_file['ra_deg'][:] * np.pi / 180.
            decs = cat_file['dec_deg'][:] * np.pi / 180.
            ngal_cat = len(ras)

            sky = np.array((decs, ras))
            coords = mask.sky2pix(sky)

            ms = mask.shape
            in_bounds = (coords[0] < ms[0]) * (coords[0] >= 0.) * (coords[1] < ms[1]) * (coords[1] >= 0.)

            ncoords_out = ngal_cat - in_bounds.sum()
            print(f'galaxies outside of map bounds: {ncoords_out} of {ngal_cat} ({100.*ncoords_out/ngal_cat:.1f}%)')

            coords_in = coords[:, in_bounds]
            assert bounds_check(coords_in, mask.shape)
            coords_in = coords_in.astype(int)
            mask[coords_in[0], coords_in[1]] = 1.


    mask_upscale = enmap.upgrade(mask, grid_downsample)

    intersect = sdss_mask * mask_upscale

    if plots:
        om = OutputManager(base_path='output', title='make_randcat_mask_desils', logs=['log',])

        fig1 = enplot.plot(mask_upscale.downgrade(plot_downsample), ticks=15, colorbar=True)
        om.savefig(f'desils_randcat_mask', mode='pixell', fig=fig1)

        fig2 = enplot.plot(sdss_mask.downgrade(plot_downsample), ticks=15, colorbar=True)
        om.savefig(f'sdss_mask', mode='pixell', fig=fig2)

        fig3 = enplot.plot(intersect.downgrade(plot_downsample), ticks=15, colorbar=True)
        om.savefig(f'desils_x_sdss_mask', mode='pixell', fig=fig3)

    enmap.write_map(mask_out_path, mask_upscale)
    enmap.write_map(intersect_out_path, intersect)