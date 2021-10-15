import os
from os.path import isfile

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


def import_gals(gal_cat, zerr_cut=0.05):
        print(f'importing galaxy catalog {gal_cat}')

        gal_ext = get_ext(gal_cat)

        assert(gal_ext == 'h5')

        # TODO add old fits handler?
        z_mask = None
        with h5py.File(gal_cat, 'r') as f:
            vr_s = f['vr_smoothed'][:]
            vr_u = f['vr_unsmoothed'][:]
            zerrs = f['zerr'][:]
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

        return gal_pos, vr_s[z_mask], vr_u[z_mask], zerrs[z_mask]


def make_vr_list(ref_map_t, gal_pos, vr_s, vr_u, zerr):
    gal_inds = []
    vr_list = []


    n_gal = len(vr_s)

    ntick_percent = max(1, 0.01 * n_gal)
    
    idecs, iras = iround(ref_map_t.sky2pix((gal_pos[:,0], gal_pos[:,1])))

    in_bounds = bounds_check(np.array((idecs, iras)).T, ref_map_t.shape)

    n_ib = in_bounds.sum()

    gal_inds = np.array((idecs[in_bounds], iras[in_bounds])).T

    n_ob = n_gal - n_ib

    print(f'Fraction of out-of-bounds galaxies: {float(n_ob) / n_gal:.2f}')

    assert len(gal_inds) == len(vr_s[in_bounds])

    return gal_inds, vr_s[in_bounds], vr_u[in_bounds], zerr[in_bounds]


# generate a fixed-format summary file to streamline the handling of 
# galaxy vr data in the broader pipeline
def make_gal_summaries(ref_map_path, catalog_files, zerr_cut=0.05):

    ref_map_t = enmap.read_map(ref_map_path)[0]

    summaries = {}
    for cat in catalog_files:
        gal_pos, vr_s_raw, vr_u_raw, zerr_raw = import_gals(cat, zerr_cut=0.05)
        gal_inds, vr_s, vr_u, zerr = make_vr_list(ref_map_t, gal_pos, vr_s_raw, vr_u_raw, zerr_raw)
        fname = get_fname(cat)
        summaries[fname] = [gal_inds, vr_s, vr_u, zerr]

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

            gal_inds, vr_s, vr_u, zerr = summaries[fname]
            n_gal = len(gal_inds)
            n_gal_total += n_gal

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
            print('loading redshift errors')
            zerr_ds = gal_grp.create_dataset('zerr', (n_gal,), dtype=float)
            zerr_ds[:] = zerr

        inds_all = np.unique(inds_all, axis=0)
        n_inds_unique = len(inds_all)
        inds_dset = gal_file.create_dataset('unique_inds', (n_inds_unique,2), dtype=int)
        inds_dset[:,:] = inds_all
    # fin

# TODO: move this to a script folder
def do_gal_summaries():
    cat_base = '/home/aroman/data/vr_source/v01/desils/'
    catalog_files = [cat_base + 'v01_desils_north_cmass.h5',
                     cat_base + 'v01_desils_south_cmass.h5',
                     cat_base + 'v01_desils_north_lowz.h5',
                     cat_base + 'v01_desils_south_lowz.h5']
    # catalog_files = [cat_base + 'v01_desils_north_cmass.h5',]
    ref_map_path = '/home/aroman/data/act/act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'
    gal_out_path = '/home/aroman/data/vr_summaries/vr_summaries.h5'
    save_gal_summaries(ref_map_path, catalog_files, gal_out_path)


def make_map_sims(ntrial, cl_ref, gal_file, out_path):
	
    # cl_sim = np.fromfile(cl_ref)
    pass

if __name__ == "__main__":
    do_gal_summaries()