import os
from os import listdir
from os.path import isfile, join

import h5py
from astropy.io import fits

import numpy as np

import pixell
from pixell import enmap


def iround(f_ind):
    sgn = np.sign(f_ind).astype(int)
    ind_abs = (np.abs(f_ind) + 0.5).astype(int)
    return sgn * ind_abs


def bounds_check(ipos, shape):
    return np.logical_and(np.logical_and(ipos[:,0] >= 0, ipos[:,0] < shape[0]), 
                          np.logical_and(ipos[:,1] >=0, ipos[:,1] < shape[1]))


def get_fname(path):
    return '.'.join(path.split(os.sep)[-1].split('.')[:-1])


def get_ext(path):
    return path.split('.')[-1]


def get_files(basedir, ext='h5'):
    paths = os.listdir(basedir)
    return [join(basedir,p) for p in paths if get_ext(p) == ext]


def import_gals(gal_cat, zerr_cut=0.05):
        print(f'importing galaxy catalog {gal_cat}')

        gal_ext = get_ext(gal_cat)

        assert(gal_ext == 'h5')

        # TODO add old fits handler?
        z_mask = None
        # WARN: DESI and SDSS data has different formats!!!
        with h5py.File(gal_cat, 'r') as f:
            # handle legacy v00 files
            if 'vr_smoothed' in f:
                vr_s = f['vr_smoothed'][:]
                vr_u = f['vr_unsmoothed'][:]
                zerrs = f['zerr'][:]
            else:
                vr_s = f['vr'][:]
                vr_u = vr_s.copy()
                zerrs = vr_s.copy()
                zerrs[:] = 0.
            zs = f['z'][:]

            decs =  f['dec_deg'][:] * (np.pi / 180)
            ras =  f['ra_deg'][:] * (np.pi / 180)

            z_mask = zerrs <= zerr_cut

            ngal_in = len(vr_s)
            ngal2d = z_mask.sum()

            gal_pos = np.empty((ngal2d, 2), dtype=float)
            gal_pos[:,0] = decs[z_mask]
            gal_pos[:,1] = ras[z_mask]

            # inds = []
            # for i, (dec, ra, vrs, vru, zerr) in enumerate(zip(decs, ras, vr_s, vr_u, zerrs)):
            #     if zerr <= zerr_cut:
            #         gal_pos.append([ra, dec])
            #         inds.append(i)

            print(f'Keeping {ngal2d} of {ngal_in} galaxies; fraction={float(ngal2d)/ngal_in:.3f}')

        assert z_mask is not None
        print("done")

        return gal_pos, vr_s[z_mask], vr_u[z_mask], zerrs[z_mask], zs[z_mask]


# Using a reference ACT map, keep only galaxies that are within the field
def make_vr_list(ref_map_t, gal_pos, vr_s, vr_u, zerr, zs):
    gal_inds = []
    vr_list = []

    n_gal = len(vr_s)

    ntick_percent = max(1, 0.01 * n_gal)
    
    idecs, iras = iround(ref_map_t.sky2pix((gal_pos[:,0], gal_pos[:,1])))

    in_bounds = bounds_check(np.array((idecs, iras)).T, ref_map_t.shape)

    n_ib = in_bounds.sum()

    gal_inds = np.array((idecs[in_bounds], iras[in_bounds])).T

    n_ob = n_gal - n_ib

    print(f'{n_ib} of {n_gal} galaxies in map field; fraction={1 - float(n_ob) / n_gal:.2f}')

    assert len(gal_inds) == len(vr_s[in_bounds])

    return gal_pos[in_bounds], gal_inds, vr_s[in_bounds], vr_u[in_bounds], zerr[in_bounds], zs[in_bounds]


# generate a fixed-format summary file to streamline the handling of 
# galaxy vr data in the broader pipeline
def make_gal_summaries(ref_map_path, catalog_files, zerr_cut=0.05):

    ref_map_t = enmap.read_map(ref_map_path)[0]

    summaries = {}
    for cat in catalog_files:
        gal_pos, vr_s_raw, vr_u_raw, zerr_raw, zs_raw = import_gals(cat, zerr_cut=0.05)
        gal_pos, gal_inds, vr_s, vr_u, zerr, zs = make_vr_list(ref_map_t, gal_pos, vr_s_raw, vr_u_raw, zerr_raw, zs_raw)
        fname = get_fname(cat)
        summaries[fname] = [gal_pos, gal_inds, vr_s, vr_u, zerr, zs]

    return summaries


# TODO: create a convenient class to contain summary file data
# class GalData:
#     def __init__():


def save_gal_summaries(ref_map_path, catalog_files, gal_out_path):
    summaries = make_gal_summaries(ref_map_path, catalog_files)

    with h5py.File(gal_out_path, 'w-') as gal_file:

        grp = gal_file.create_group('summaries')

        inds_all = None
        n_gal_total = 0
        for fname in summaries.keys():
            gal_grp = grp.create_group(fname)

            print(f'loading summary: {fname}')

            gal_pos, gal_inds, vr_s, vr_u, zerr, zs = summaries[fname]
            n_gal = len(gal_inds)
            n_gal_total += n_gal

            print('loading galaxy positions')
            pos = gal_grp.create_dataset('gal_pos', (n_gal, 2), dtype=float)
            pos[:,:] = gal_pos

            print('loading galaxy indices')
            inds = gal_grp.create_dataset('gal_inds', (n_gal, 2), dtype=int)
            inds[:,:] = gal_inds

            if inds_all is None:
                inds_all = gal_inds.copy()
            else:
                inds_all = np.concatenate((inds_all, gal_inds))

            print('loading smoothed velocities')
            vr_s_ds = gal_grp.create_dataset('vr_s', (n_gal,), dtype=float)
            vr_s_ds[:] = vr_s
            print('loading unsmoothed velocities')
            vr_u_ds = gal_grp.create_dataset('vr_u', (n_gal,), dtype=float)
            vr_u_ds[:] = vr_u
            print('loading redshifts')
            z_ds = gal_grp.create_dataset('z', (n_gal,), dtype=float)
            z_ds[:] = zs
            print('loading redshift errors')
            zerr_ds = gal_grp.create_dataset('zerr', (n_gal,), dtype=float)
            zerr_ds[:] = zerr

        inds_all = np.unique(inds_all, axis=0)
        n_inds_unique = len(inds_all)
        inds_dset = gal_file.create_dataset('unique_inds', (n_inds_unique, 2), dtype=int)
        inds_dset[:,:] = inds_all
    # fin


# cat_base = '/home/aroman/data/vr_source/v01/desils/'
# sdss_base = '/home/aroman/data/vr_source/v01/sdss/'

# v0_base = '/home/aroman/data/vr_source/v00/'
# v0_set = get_files(v0_base)
# north_cmass = cat_base + 'v01_desils_north_cmass.h5'
# south_cmass = cat_base + 'v01_desils_south_cmass.h5'
# north_lowz = cat_base + 'v01_desils_north_lowz.h5'
# south_lowz = cat_base + 'v01_desils_south_lowz.h5'

# v0_north_cmass = v0_base + 'v00_sdss_cmass_north.h5'
# v0_south_cmass = v0_base + 'v00_sdss_cmass_south.h5'
# v0_north_lowz = v0_base + 'v00_sdss_lowz_north.h5'
# v0_south_lowz = v0_base + 'v00_sdss_lowz_south.h5'

# sdss_cmass_north = sdss_base + 'v01_sdss_cmass_north.h5'
# sdss_cmass_south = sdss_base + 'v01_sdss_cmass_south.h5'
# sdss_lowz_north = sdss_base + 'v01_sdss_lowz_north.h5'
# sdss_lowz_south = sdss_base + 'v01_sdss_lowz_south.h5'

# # catalog_sets = [[north_cmass, north_lowz, south_cmass, south_lowz],
# #                 [north_cmass, north_lowz],
# #                 [south_cmass, south_lowz],
# #                 [north_cmass, south_cmass],
# #                 [north_lowz, south_lowz],
# #                 [south_cmass,],
# #                 [north_cmass,],
# #                 v0_set,
# #                 [v0_north_cmass,],
# #                 [v0_south_cmass,],
# #                 [v0_north_lowz,],
# #                 [v0_south_lowz,]]

# catalog_sets = [[sdss_cmass_north, sdss_cmass_south, sdss_lowz_north, sdss_lowz_south],
#                 [sdss_cmass_north, sdss_cmass_south],
#                 [sdss_lowz_north, sdss_lowz_south],
#                 [sdss_cmass_north, sdss_lowz_north],
#                 [sdss_cmass_south, sdss_lowz_south],
#                 [sdss_cmass_north,]]

# set_names = ['sdss_all', 'sdss_cmass', 'sdss_lowz', 'sdss_north', 'sdss_south', 'sdss_cmass_north']

# # set_names = ['all', 'north', 'south', 'cmass', 'lowz', 'south_cmass', 'north_cmass',
# #              'v0_all', 'v0_cmass_north', 'v0_cmass_south', 'v0_lowz_north', 'v0_lowz_south']

# # TODO: move this to a script folder
# def do_gal_summaries():
#     # catalog_files = [cat_base + 'v01_desils_north_cmass.h5',
#     #                  cat_base + 'v01_desils_south_cmass.h5',
#     #                  cat_base + 'v01_desils_north_lowz.h5',
#     #                  cat_base + 'v01_desils_south_lowz.h5']
#     gal_out_base = '/home/aroman/data/vr_summaries/v01_'
#     ref_map_path = '/home/aroman/data/act/act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'
#     # catalog_files = [cat_base + 'v01_desils_north_cmass.h5',]
#     for cat_set, set_name in zip(catalog_sets, set_names):
#         gal_out_path = gal_out_base + set_name + '.h5'
#         save_gal_summaries(ref_map_path, cat_set, gal_out_path)


# def make_map_sims(ntrial, cl_ref, gal_file, out_path):
	
#     # cl_sim = np.fromfile(cl_ref)
#     pass

# if __name__ == "__main__":
#     do_gal_summaries()