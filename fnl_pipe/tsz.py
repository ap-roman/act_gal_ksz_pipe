import numpy as np
from pixell import enmap

import astropy.io.fits as fits
from astropy import units as u


def get_clusters(tsz_cat_path):
    ret = {}
    infields = ('decDeg', 'RADeg')
    outfields = ('dec', 'ra')
    coeffs = np.array((np.pi/180, np.pi/180))
    
    with fits.open(tsz_cat_path) as f:
        hdr = f[1].header
        cat = f[1].data

        for infield, outfield, c in zip(infields, outfields, coeffs):
            ret[outfield] = c * cat.field(infield)
    
    return ret


def pix_bounds_check(pix, shape):
    a = np.all(pix[:,0] >= 0) and np.all(pix[:,0] < shape[0])
    b = np.all(pix[:,1] >= 0) and np.all(pix[:,1] < shape[1])
    return a and b


def sky2pixint(sky, map_ref):
    # not sure why this workaround is required
    coords = [map_ref.sky2pix(sky_coord) for sky_coord in sky]
    return np.round(coords).astype(int)


def fill_between(map_t, left, right, c=1.):
    ar = np.array((left,right))
    
    l2 = np.min(ar, axis=0)
    r2 = np.max(ar, axis=0)
    
    for l,r in zip(l2, r2):
        map_t[l[0]:r[0], l[1]:r[1]] = c


def make_tsz_mask(tsz_cat_path, map_ref, mask_width):
    mwv = (mask_width * u.arcmin).to(u.radian).value

    ret = enmap.ndmap(np.ones(map_ref.shape), map_ref.wcs)
    
    clusters = get_clusters(tsz_cat_path)
    dec = clusters['dec']
    ra = clusters['ra']
    
    center = np.array((dec, ra)).T
    
    left = center - 0.5 * mwv * np.array([1, 1])[None, :]
    right = center + 0.5 * mwv * np.array([1, 1])[None, :]
    
    pix_l = sky2pixint(left, map_ref)
    pix_r = sky2pixint(right, map_ref)
    
    assert pix_bounds_check(pix_l, map_ref.shape) and pix_bounds_check(pix_r, map_ref.shape)
    
    fill_between(ret, pix_l, pix_r, 0.)
    
    return ret, np.array((dec,ra)).T
