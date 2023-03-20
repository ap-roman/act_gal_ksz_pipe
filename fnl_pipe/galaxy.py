import numpy as np
import h5py

from fnl_pipe.util import in_bounds, iround, get_fname

# A module to house galaxy-specific objects e.g. descriptors and manipulators
# of galaxy catalogs.

# TODO: for now using GalPipe class from pipe.py

# dev notes (dataset key):
_v3_keys = ('Gmag', 'W1mag', 'W2mag', 'dec_deg', 'fracflux_g', 'fracflux_r', 
               'fracflux_z', 'fracin_g', 'fracin_r', 'fracin_z', 'fracmasked_g', 
               'fracmasked_r', 'fracmasked_z', 'gmag', 'maskbits', 'morphology', 
               'nobs_g', 'nobs_r', 'nobs_z', 'ra_deg', 'rfibermag', 'rmag', 
               'vr_smoothed_0.25', 'vr_smoothed_0.5', 'vr_smoothed_0.75', 
               'vr_smoothed_1.0', 'vr_smoothed_1.25', 'vr_smoothed_1.5', 
               'vr_smoothed_1.75', 'vr_smoothed_2.0', 'vr_unsmoothed', 'z', 
               'zerr', 'zfibermag', 'zmag')


class GalCut:
    pass
    # TODO: implement abstract class/abstractmethod?


class NullCut:
    def __call__(self, catfile):
        # pass ngal somehow?
        ngal = catfile['zerr'].size
        return np.ones(ngal, dtype=bool)


class _RangeCut(GalCut):
    def __init__(self, key, vmin, vmax):
        assert key in _v3_keys
        assert vmin <= vmax
        
        self.key = key
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, catfile):
        dset = catfile[self.key]
        return np.logical_and(dset[:] < self.vmax, dset[:] >= self.vmin)


class ZerrCut(_RangeCut):
    def __init__(self, zerr_max=0.05):
        super().__init__('zerr', 0., zerr_max)
        self.zerr_max = zerr_max


class _LRGCut(GalCut):
    def __init__(self, cvec):
        assert len(cvec) == 6
        self.cvec = cvec
        # cvec encodes the constants that define the north/south LRG cuts respectively

    # return the indices of galaxies matching the cut criteria
    def __call__(self, catfile):
        z = catfile['zmag'][:]
        r = catfile['rmag'][:]
        g = catfile['gmag'][:]
        w1 = catfile['W1mag'][:]
        zfiber = catfile['zfibermag'][:]

        # "We remove saturated objects and sources near bright stars, large galaxies, or globular clusters by requiring 
        #  that LS MASKBITS4 1, 5, 6, 7, 11, 12, and 13 are not set"

        ormask = 0b1100011100011101
        bitmask = catfile['maskbits'][:]

        include_mask = np.bitwise_or(bitmask, ormask) == ormask 

        print(f'include_mask includesum: {include_mask.sum()} of {len(bitmask)}')

        c1, c2, c3, c4, c5, c6 = self.cvec

        a = (z - w1) > 0.8 * (r - z) - c1
        b = np.logical_or((g - w1 > c2) * (g - r > c3), r - w1 > c4)
        c = (r - z > (z - c5) * 0.45) * (r - z > (z - c6) * 0.19)
        d = r - z > 0.7
        e = zfiber < 21.5

        return a * b * c * d * e * include_mask


# 2010.11282 eq 1
class LRGSouthCut(_LRGCut):
    def __init__(self):
        super().__init__([0.6, 2.6, 1.4, 1.8, 16.83, 13.8])


# 2010.11282 eq 2
class LRGNorthCut(_LRGCut):
    def __init__(self):
        super().__init__([0.65, 2.67, 1.45, 1.85, 16.69, 13.68])


class _LogicalCut(GalCut):
    def __init__(self, cuts, comparator, initializer):
        if isinstance(cuts, GalCut):
            cuts = [cuts,]

        for cut in cuts:
            assert isinstance(cut, GalCut)

        self.cuts = cuts

        assert comparator == np.logical_and or comparator == np.logical_or
        assert initializer == np.zeros or initializer == np.ones
        self.comparator = comparator
        self.initializer = initializer

    def __call__(self, catfile):
        ngal = catfile['zerr'].size

        ret = self.initializer(ngal, dtype=bool)

        for cut in self.cuts:
            ret = self.comparator(ret, cut(catfile))

        return ret


class OrCut(_LogicalCut):
    def __init__(self, cuts):
        super().__init__(cuts, comparator=np.logical_or, initializer=np.zeros)


class AndCut(_LogicalCut):
    def __init__(self, cuts):
        super().__init__(cuts, comparator=np.logical_and, initializer=np.ones)


# Expects v3 desils files
class GalPipe:
    def __init__(self, cat_path, vr_dset, inds, ref_map):
        self.cat_path = cat_path
        self.cat_name = get_fname(cat_path)
        self.vr_dset = vr_dset
        self.inds = inds.copy() # set of inclusion masks

        self.ref_map = ref_map

        # legacy fields
        self.ngal_tot = len(inds)
        self.init_lists = False

    def import_data(self, inds=None):
        if inds is None:
            inds = self.inds

        print(f'importing gal data from {self.cat_name}')
        with h5py.File(self.cat_path, 'r') as h5file:
            assert self.vr_dset in h5file.keys()
            assert 'dec_deg' in h5file.keys()
            assert 'ra_deg' in h5file.keys()

            decs = h5file['dec_deg'][:][inds] * np.pi / 180.
            ras = h5file['ra_deg'][:][inds] * np.pi / 180.

            dec_inds, ra_inds = iround(self.ref_map.sky2pix((decs, ras)))
            gal_inds = (dec_inds, ra_inds)

            bounds_mask = in_bounds(gal_inds, self.ref_map.shape)
            print(f'in_bounds: {bounds_mask.sum()} of {self.ngal_tot}')

            self.vrs = h5file[self.vr_dset][:][inds][bounds_mask]
            self.vr_list = self.vrs

            self.bounds_mask = bounds_mask

            self.decs = decs[bounds_mask]
            self.ras = ras[bounds_mask]
            self.dec_inds = dec_inds[bounds_mask]
            self.ra_inds = ra_inds[bounds_mask]
            self.gal_inds = [self.dec_inds, self.ra_inds]

            self.ngal_in = bounds_mask.sum()

            self.init_lists = True

    # legacy method that does nothing
    def make_vr_list(self):
        pass

    def get_xz_list(self, map_t):
        assert self.init_lists
        return self.vrs * map_t[self.dec_inds, self.ra_inds]


class MultiPipe(GalPipe):
    def __init__(self, gal_pipes):
        self.gal_pipes = gal_pipes

        self.init_lists = False

    def import_data(self):
        ngals = []
        self.ngal_in = 0
        for gp in self.gal_pipes:
            gp.import_data()

            ngals.append(gp.ngal_in)
            self.ngal_in += gp.ngal_in
        self.ngals = ngals

        decs = np.empty(self.ngal_in)
        ras = np.empty(self.ngal_in)
        dec_inds = np.empty(self.ngal_in, dtype=int)
        ra_inds = np.empty(self.ngal_in, dtype=int)
        vrs = np.empty(self.ngal_in)

        offset = 0
        for gp, ngal in zip(self.gal_pipes, self.ngals):
            decs[offset:offset + ngal] = gp.decs
            dec_inds[offset:offset + ngal] = gp.dec_inds
            ras[offset:offset + ngal] = gp.ras
            ra_inds[offset:offset + ngal] = gp.ra_inds
            vrs[offset:offset + ngal] = gp.vrs
            offset += ngal

        self.decs = decs
        self.ras = ras
        self.gal_inds = [dec_inds, ra_inds]
        self.vrs = vrs
        self.vr_list = vrs


    def make_vr_list(self):
        for gp in self.gal_pipes:
            gp.make_vr_list()

        self.init_lists = True

    def get_xz_list(self, map_t):
        assert self.init_lists

        ret = np.empty(self.ngal_in)

        offset = 0
        for gp, ngal in zip(self.gal_pipes, self.ngals):
            ret[offset:offset + ngal] = gp.get_xz_list(map_t)
            offset += ngal

        return ret


# This class handles the conversion of input data files (specifically v3 desils era files)
# to GalPipe objects. GalPipes are the pipeline's direct interface to the catalogs on disk
class GalCat:
    _valid_widths = ['0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0']

    def __init__(self, cat_path):
        self.cat_path = cat_path

        ngal = None
        with h5py.File(self.cat_path, 'r') as h5file:
            ngal = h5file['zerr'].size
        assert ngal is not None

        self.ngal = ngal

    # returns a GalPipe object subject to the selection criterion (a list of GalCut)
    def get_subcat(self, cut, ref_map, vr_zerr_width='1.0'):
        assert vr_zerr_width in self._valid_widths
        fwidth = float(vr_zerr_width)

        assert isinstance(cut, GalCut)

        with h5py.File(self.cat_path, 'r') as h5file:
            v3_dset = 'vr_smoothed_' + vr_zerr_width
            assert v3_dset in h5file.keys() or (vr_zerr_width == '1.0' and 'vr_smoothed' in h5file.keys())

            if v3_dset in h5file.keys():
                # v3 file
                vr_dset = v3_dset
            else:
                vr_dset = 'vr_smoothed'

            thiscut = cut(h5file)
            print(f'included galaxies: {thiscut.sum()} of {self.ngal}')
            inds = np.arange(self.ngal)[thiscut]

            return GalPipe(self.cat_path, vr_dset, inds, ref_map)
        return None


class DESILSCat:
    def __init__(self, *, cat_north, cat_south):
        self.cats = []
        
        cat_paths = [cat_north, cat_south]
        for cat_path in cat_paths:
            self.cats.append(GalCat(cat_path))

        assert len(self.cats) == 2

    def get_subcat(self, cuts_list, ref_map, vr_zerr_width='1.0'):
        gps = []
        for cat, cut in zip(self.cats, cuts_list):
            assert isinstance(cut, GalCut)
            gps.append(cat.get_subcat(cut, ref_map, vr_zerr_width))

        return MultiPipe(gps)
