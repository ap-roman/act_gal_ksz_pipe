import pixell
from pixell import enmap, enplot, utils, curvedsky
from pixell.curvedsky import map2alm, alm2map, rand_alm, alm2cl, almxfl, rand_map
from pixell.reproject import enmap_from_healpix, enmap_from_healpix_interp, healpix_from_enmap
import healpy as hp
from scipy.special import sph_harm

import astropy
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits

import h5py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import interpolate
from scipy.interpolate import splev, splrep
from scipy.optimize import minimize


data_path = '/home/aroman/data/'
act_path = data_path + 'act/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'

map_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'
ivar_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_ivar.fits'
beam_path = act_path + 'beam_f150_daynight.txt'

cl_cmb_path = data_path + 'spectra/cl_cmb.npy'
cl_ksz_path = data_path + 'spectra/cl_ksz.npy'

planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_enmask_path = mask_path + 'planck_foreground.npy'

# sdss_catalog_path = data_path + 'sdss_radial_velocities/v1/galaxy_DR10v8_CMASS_South_vrad_using_randoms.fits'
# sdss_catalog_path = data_path + 'sdss_radial_velocities/dr12v5/vradial_DR12v5_CMASS_North.h5'
# sdss_catalog_path = data_path + 'sdss_radial_velocities/dr12v5/vradial_DR12v5_LOWZ_South.h5'
# sdss_catalog_path = data_path + 'sdss_radial_velocities/dr12v5/vradial_DR12v5_LOWZ_North.h5'
# sdss_catalog_path = data_path + 'vr_source/v01/desils/v01_desils_south_cmass.h5'
# catalog_path = data_path + 'vr_summaries/vr_summaries.h5'
# catalog_path = data_path + 'vr_summaries/v01_all.h5'
# catalog_path = data_path + 'vr_summaries/v01_south.h5'
# catalog_path = data_path + 'vr_summaries/v01_north.h5'
# catalog_path = data_path + 'vr_summaries/v01_south_cmass.h5'
# catalog_path = data_path + 'vr_summaries/v01_north_cmass.h5'
# catalog_path = data_path + 'vr_summaries/v01_v0_all.h5'
# catalog_path = data_path + 'vr_summaries/v01_v0_cmass_south.h5'
# catalog_path = data_path + 'vr_summaries/v01_v0_cmass_north.h5'
# catalog_path = data_path + 'vr_summaries/v01_v0_lowz_south.h5'
# catalog_path = data_path + 'vr_summaries/v01_v0_lowz_north.h5'
catalog_path = data_path + 'vr_summaries/v01_sdss_all.h5'
# catalog_path = data_path + 'vr_summaries/v01_sdss_lowz.h5'
# catalog_path = data_path + 'vr_summaries/v01_sdss_cmass.h5'

# datpath = '/home/aroman/act_data/act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'

# TODO: simulate to higher lmax than xform lmax
LMAX=12000
LMAX_SIM = 12000
PLANCK_NSIDE=2048
eta2_ref = 0.003**2 #uK^2 per steridian


def fequal(a,b,tol=1e-6):
    return np.all(np.abs(a - b) <= tol)


def fequal_either(a,b,tol=1e-3):
    return np.all(np.logical_or(np.abs(a - b) <= tol, 2 * np.abs(a - b) / np.abs(a + b) <= tol))


def parse_beam(path):
    line = None
    with open(path, 'r') as f:
        lines = f.readlines()

    ret = []
    for line in lines:
        l = int(line[:6])
        amp = float(line[6:])
        ret.append((l,amp))

    return np.array(ret)


class InterpWrapper:
    def __init__(self, x, y):
        self.tck = interpolate.splrep(x, y, s=0)

    def __call__(self, x_eval):
        return interpolate.splev(np.array(x_eval), self.tck, der=0)


def inf_mask(mask):
    ret = np.zeros(mask.shape,dtype=float)
    ret[mask] = np.inf
    return ret


def map2cl(t_map, lmax):
    return alm2cl(map2alm(t_map, lmax=lmax))


def ind_round(ix, iy, shape):
    ixr = int(ix + 0.5)
    iyr = int(iy + 0.5)
    ixr = min(max(0, ixr), shape[0] - 1)
    iyr = min(max(0, iyr), shape[1] - 1)
    return ixr, iyr


# assumes dec, ra order in radians
def angle_tolerance(pos1, pos2, tol):
    dec_tol = np.abs(pos1[0] - pos2[0]) <= tol[0]

    pos1[1] %= 2 * np.pi
    pos2[1] %= 2 * np.pi

    naive_delta = np.abs(pos1[1] - pos2[1])
    ra_tol = min(naive_delta, 2 * np.pi - naive_delta) <= tol[1] 
    return dec_tol and ra_tol


def make_zero_map(ref_map_t):
    return enmap.ndmap(np.zeros(ref_map_t.shape), ref_map_t.wcs)


def make_zero_alm(ref_map_t, lmax):
    return map2alm(make_zero_map(ref_map_t), lmax=lmax)


def iround(f_ind):
    sgn = np.sign(f_ind).astype(int)
    ind_abs = (np.abs(f_ind) + 0.5).astype(int)
    return sgn * ind_abs


def bounds_check(ipos, shape):
    return (ipos[0] >= 0) and (ipos[0] < shape[0]) and (ipos[1] >=0) \
           and (ipos[1] < shape[1])


def get_fname(path):
    return '.'.join(path.split(os.sep)[-1].split('.')[:-1])


def get_ext(path):
    return path.split('.')[-1]


class GalCat:
    def __init__(self, name, gal_inds, vr_s, vr_u, zerrs, zs):
        self.name = name
        self.gal_inds = gal_inds
        self.vr_s = vr_s
        self.vr_u = vr_u
        self.zerrs = zerrs
        self.zs = zs

        self.ngal = self.gal_inds.shape[0]

def make_sky_map(data, filename, title=''):
    plt.figure(dpi=300.)
    plt.title(f'Sky Map {title}')
    plt.imshow(data[::-1,:])

    print(plt.get_xticks())


    plt.savefig(filename)
    plt.close()


# WARN: assumes rigid reference geometry across maps
class GalPipe:
    def __init__(self, catalog_path, act_pipe, import_now=False, diag_plots=True, lmax=LMAX):
        self.lmax = int(lmax)
        self.catalog_path = catalog_path
        self.diag_plots = diag_plots
        self.ngal = 0
        self.ref_map_t = act_pipe.imap_t

        if(import_now):
            self.import_data()

    def import_data(self, map_fkp=None):
        print(f'importing galaxy catalog {self.catalog_path}')

        do_fkp_sum = map_fkp is not None

        gal_ext = get_ext(self.catalog_path)

        assert(gal_ext == 'h5')

        self.gal_summaries = {}
        with h5py.File(self.catalog_path, 'r') as f:
            summaries = f['summaries']
            fnames = list(summaries.keys())

            for fname in fnames:
                grp = summaries[fname]
                gal_cat = GalCat(fname, grp['gal_inds'][:,:],
                                 grp['vr_s'][:], grp['vr_u'][:],
                                 grp['zerr'][:], grp['z'][:])
                self.gal_summaries[fname] = gal_cat
                self.ngal += gal_cat.ngal
                
                if self.diag_plots and do_fkp_sum:
                    inds = grp['gal_inds'][:,:]
                    fkp_sum = map_fkp[inds[:,0], inds[:,1]].sum()
                    print(f'summary of {fname} subset')
                    print(f'\tfkp sum: {fkp_sum:.2f}')
                    print(f'\tfkp sum per gal: {fkp_sum/gal_cat.ngal:.2f}')

                    # print(f'ind shape: {inds.shape}')
                    # hist, xedges, yedges = np.histogram2d(inds[:,1], inds[:,0])

                    # corners = self.ref_map_t.corners()
                    # print(corners)

        print("done")

    def make_vr_list(self):
        self.gal_inds = np.empty((self.ngal,2), dtype=int)
        self.vr_list = np.empty(self.ngal)

        i = 0
        for fname in self.gal_summaries.keys():
            gal_cat = self.gal_summaries[fname]
            ngal = gal_cat.ngal
            self.gal_inds[i:i+ngal] = gal_cat.gal_inds
            self.vr_list[i:i+ngal] = gal_cat.vr_s
            i += ngal

        self.gal_inds = self.gal_inds.T
        self.vr_list = self.vr_list.T
        self.ngal_in = self.ngal # rename ngal_in?
    # def make_vr_list(self, ref_map_t, make_alm=False):
    #     if make_alm:
    #         self.vr_alm = make_zero_alm(ref_map_t, lmax=self.lmax)

    #     self.vr_map = make_zero_map(ref_map_t)

    #     angular_res = ref_map_t.wcs.wcs.cdelt[0] * np.pi / 180.

    #     ngal = len(self.gal_pos)

    #     corners = ref_map_t.corners()

    #     if self.diag_plots:
    #         plt.figure(dpi=300)
    #         plt.scatter(self.gal_pos[:,1] * 180 / np.pi, self.gal_pos[:,0] * 180 / np.pi,
    #                     marker='+', s=1)
    #         plt.axhline(corners[0,0] * 180 / np.pi, color='red')
    #         plt.axhline(corners[1,0] * 180 / np.pi, color='red')
    #         # plt.axvline(corners[1,0] * 180 / np.pi, color='red')
    #         # plt.axvline(corners[1,1] * 180 / np.pi, color='red')
    #         plt.ylabel('Declination (deg)')
    #         plt.xlabel('Right Ascension (deg)')
    #         plt.savefig('plots/galaxy_locs.png')
    #         plt.close()

    #     gal_inds = []
    #     vr_list = []

    #     n_ob = 0

    #     for i, (v_r, pos) in enumerate(zip(self.v_rad, self.gal_pos)):
    #         dec, ra = pos
    #         idec, ira = iround(ref_map_t.sky2pix((dec,ra)))
    #         # idec, ira = self.vr_map.sky2pix((dec,ra))

    #         if not bounds_check((idec,ira), ref_map_t.shape):
    #             # print(dec, ra, idec, ira)
    #             n_ob += 1
    #         else:
    #             # TODO: check galaxy overlap?
    #             gal_inds.append([idec, ira])
    #             vr_list.append(v_r)
    #             self.vr_map[idec, ira] += v_r
    #             # this method is super inefficient
    #             # self.vr_alm[:self.lmax + 1] += v_r * sph_harm(ms, ls, ra, dec)
    #         # print('Processed galaxy {} of {}'.format(i + 1, ngal))
    #     self.gal_inds = np.array(gal_inds).T
    #     self.vr_list = np.array(vr_list)
    #     self.ngal_in = ngal - n_ob
    #     print('Fraction of out-of-bounds galaxies: {:.2f}'.format(float(n_ob + 1) / ngal))
    #     if make_alm:
    #         pixel_area = angular_res**2
    #         self.vr_alm = map2alm(self.vr_map / pixel_area, lmax=self.lmax)

    def get_t_list(self, t_map):
        return t_map[self.gal_inds[0,:], self.gal_inds[1,:]]

    def get_xz_list(self, t_map):
        t_gal_list = t_map[self.gal_inds[0,:], self.gal_inds[1,:]]
        return t_gal_list * self.vr_list

    # def get_xz_map(self, t_map):
    #     # ret_map = make_zero_map(t_map)
    #     # # WARN: this is unsafe
    #     # ret_map[self.gal_inds[0,:], self.gal_inds[1,:]] += self.vr_list
    #     # ret_map *= t_map
    #     return self.vr_map * t_map

    def compute_a_ksz(self, t_map):
        return self.get_xz_list(t_map).sum()

    # TODO: FIX
    def test_vr_map(self, ref_map_t):
        ntrial = 4096
        locs = np.empty((ntrial,2))
        corners = ref_map_t.corners()
        locs[:,0] = corners[0][0] + np.random.rand(ntrial) * (corners[1][0] - corners[0][0])
        locs[:,1] = corners[0][1] + np.random.rand(ntrial) * (corners[1][1] - corners[0][1])


        angular_res = ref_map_t.wcs.wcs.cdelt * np.pi / 180.

        print(f'map corners: [dec,ra]: {ref_map_t.corners()}')

        n_ob = 0
        for dec, ra in locs:
            idec, ira = ref_map_t.sky2pix((dec, ra), corner=True)
            if not bounds_check([idec, ira], ref_map_t.shape):
                if n_ob < 50: # print the first 50 OOB
                    print([dec, idec, ra, ira], ref_map_t.shape)
                n_ob +=1
            else:
                dec_r, ra_r = ref_map_t.pix2sky((idec,ira), corner=True)
                if not angle_tolerance([dec, ra], [dec_r, ra_r], np.abs(10 * angular_res)):
                    print(f'dec, ra, dec_r, ra_r, res_dec, res_ra, idec, ira')
                    print(f'{dec:.5e} {ra:.5e} {dec_r:.5e} {ra_r:.5e} {angular_res[0]:.5e} {angular_res[1]:.5e} {idec} {ira}')
        
        print('Fraction of out-of-bounds galaxies: {:.2f}'.format(float(n_ob + 1) / ntrial))


class ActCMBPipe:
    def __init__(self, map_path, ivar_path, beam_path, fid_ksz_path, fid_cmb_path, planck_mask_path,
                 custom_l_weight=None,
                 fid_cib_path=None, diag_plots=False, l_fkp=3000, l_ksz=3000, lmax=LMAX, lmax_sim=LMAX_SIM):
        self.lmax = int(lmax)
        self.lmax_sim = int(lmax_sim)
        self.lmax_internal = max(lmax, lmax_sim)
        self.ells = np.arange(self.lmax + 1)
        self.ells_sim = np.arange(self.lmax_sim + 1)
        self.l_plot_norm = self.ells * (1 + self.ells) * 2 * np.pi
        self.l_ksz = l_ksz
        self.l_fkp = l_fkp
        self.diag_plots = diag_plots

        self.map_path = map_path
        self.ivar_path = ivar_path
        self.beam_path = beam_path
        self.planck_mask_path = planck_mask_path

        # paths to fiducial power spectra for simulation
        self.fid_cmb_path = fid_cmb_path
        self.fid_cib_path = fid_cib_path
        self.fid_ksz_path = fid_ksz_path
        self.custom_l_weight = custom_l_weight

        self.imap = None
        self.imap_t = None
        self.ivar = None
        self.sim_map = None
        self.l_weight = None

    def import_data(self, r_lwidth=1.):
        print("importing map: {}".format(self.map_path))
        self.imap = enmap.read_map(self.map_path)
        self.imap_t = self.imap[0]

        # based on corners, store galactic plane location in celestial pixel space
        # useful for imshow masking

        print("done")

        print("generating zero_alm array")
        zero_map = self.get_empty_map()
        self.zero_alm = map2alm(zero_map, lmax=self.lmax) # just compute from np.zeros
        print("done")

        print("importing mask: {}".format(self.planck_mask_path))
        #TODO: investigate weird mask normalization
        planck_mask_ar = np.load(self.planck_mask_path)
        print(f'planck mask range: {planck_mask_ar.min()} {planck_mask_ar.max()}')
        self.mask_t = enmap.ndmap(planck_mask_ar, self.imap_t.wcs)
        self.mask_t = np.minimum(np.maximum(self.mask_t, 0.),1.)
        print("done")

        print("importing inverse variance mask: {}".format(self.ivar_path))
        self.ivar = enmap.read_map(self.ivar_path)
        self.ivar_t = self.ivar[0]
        std_t = np.sqrt(1./self.ivar_t)
        plt.figure(dpi=300)
        plt.title('pixel stdev')
        plt.imshow(np.ma.masked_where(np.isnan(std_t), std_t))
        plt.colorbar()
        plt.savefig('plots/istd.png')
        plt.close()

        # NOTE: we set inf variance pixels to zero stdev since they're masked anyway
        self.std_t = np.nan_to_num(std_t)

        # map ivar=0 to "large" std
        # either sim noise before/after 
        print("done")

        # WARN: untracked unit conversion
        self.adelt = self.imap_t.wcs.wcs.cdelt * np.pi / 180.
        self.cdelt = self.imap_t.wcs.wcs.cdelt

        assert fequal(np.abs(self.adelt[0]), np.abs(self.adelt[1]))

        self.angular_res = self.adelt[0] # radians
        self.pixel_area = self.angular_res**2 # steridians
        # self.ivar_weight = self.angular_res**2

        assert self.ivar.shape == self.imap.shape

        self.beam = parse_beam(self.beam_path)

        if self.diag_plots:
            plt.figure(dpi=300)
            plt.plot(self.beam[:,0], self.beam[:,1])
            plt.xlabel('l')
            plt.ylabel('amplitude')
            plt.savefig('plots/beam.png')
            plt.close()

    def get_empty_map(self):
        return enmap.empty(self.imap_t.shape, self.imap_t.wcs)

    def get_zero_map(self):
        return enmap.ndmap(np.zeros(self.imap_t.shape), self.imap_t.wcs)
    
    def get_zero_alm(self):
        return self.zero_alm.copy()

    # compute w_fkp_theta (FKP weight considering inverse var and mask)
    # map form: map_fkp
    # r_fkp is a weight factor that modifies the fk weighting
    def compute_pixel_weight(self, r_fkp=1., suppl_mask=None, l0=None):
        if l0 is None:
            l0 = self.l_fkp
            print(f"No FKP l scale provided, using l_fkp={self.l_fkp}")
        b2_l0 = self.beam[l0,1]**2
        # ctt_l0 = self.cl_cmb[l0] # this comes from CAMB
        ctt_3k_act = 24.8 * 2 * np.pi / 3000 / (3000 + 1) # vs 51.9 for d_l^{TT} coadd

        # TODO: redo mode-mixing analysis

        assert np.all(self.ivar_t >= 0)

        # TODO: verify normalization!!
        eta_n2 = self.ivar_t / self.pixel_area

        # zero masked pixels
        eta_n2 *= self.mask_t

        if suppl_mask is not None:
            print('Applying supplemental mask')
            eta_n2 *= suppl_mask

        self.eta_n2 = eta_n2
        print(f'Eta^-2 min/max: {np.min(eta_n2):.3e} {np.max(eta_n2):.3e}')

        # compute FKP pixel weight
        # self.w_fkp_theta = eta_n2 / (1 + b2_l0 * ctt_l0 * eta_n2)
        self.w_fkp_theta = eta_n2 / (r_fkp/(b2_l0 * ctt_3k_act) + eta_n2)
        print(f'FKP min/max: {np.min(self.w_fkp_theta):.3e} {np.max(self.w_fkp_theta):.3e}')
        self.map_fkp = enmap.ndmap(self.w_fkp_theta, self.imap_t.wcs)

        plt.figure(dpi=300)
        plt.title('fkp weighting')
        plt.imshow(self.map_fkp)
        plt.colorbar()
        plt.savefig('plots/imshow_fkp.png')

        self.t_pseudo_map = self.imap_t * self.map_fkp
        self.t_pseudo_alm = map2alm(self.t_pseudo_map, lmax=self.lmax)
        self.cl_tt_pseudo = map2cl(self.t_pseudo_map, self.lmax)

        plt.figure(dpi=300)
        plt.title('fkp-weighted temperature map (uK)')
        plt.imshow(np.ma.masked_where(self.t_pseudo_map == 0., self.t_pseudo_map), interpolation='none')
        plt.colorbar()
        plt.savefig('plots/fkp_weighted_t_map.png')
        plt.close()

    # estimate whether mode-mixing is a serious concern with w_fkp
    def compare_mode_mixing(self, l_cut=3000):
        # construct a "left" power spectrum without noise (l<1500)
        cl_1 = self.cl_cmb.copy()
        cl_1[l_cut + 1:] = 0.

        ls = np.arange(len(self.cl_cmb))

        # construct a "right" power spectrum with noise (l>1500)
        cl_2 = self.cl_cmb
        cl_2[:l_cut + 1] = 0.
        cl_pseudo_1 = self.get_pseudo_cl(cl_1)
        cl_pseudo_2 = self.get_pseudo_cl(cl_2)

        plt.figure(dpi=300)
        plt.title('Mode Mixing Diagnostic')
        # plt.yscale('log')
        plt.ylabel(u'$\log(\Delta_l)$')
        plt.xlabel(u'$l$')
        plt.plot(cl_pseudo_1 * ls * (ls + 1) / 2 / np.pi, label='<l_cut', linewidth=1)
        plt.plot(cl_pseudo_2 * ls * (ls + 1) / 2 / np.pi, label='>l_cut', linewidth=1)
        plt.axvline(l_cut)
        plt.legend()
        plt.savefig('plots/mode_mixing_comparison.png')

    # return a estimator-weighted t-pseudo beam given a realization of unbeamed noise
    # steps:
    # apply the fkp weight to input map and transform to harmonic space
    # apply supplied l weighting in harmonic space
    # convert to map space and return 
    # @profile
    def process_t_map(self, t_map, l_weight=None):
        if l_weight is None:
            assert self.l_weight is not None
            l_weight =  self.l_weight

        print("starting process_t_map")
        t_alm = map2alm(t_map * self.map_fkp, lmax=self.lmax)
        weighted_alm = almxfl(t_alm, l_weight)

        t_fkp_est = self.get_empty_map()
        t_fkp_est = alm2map(weighted_alm, t_fkp_est)
        print("done")
        return t_fkp_est

    # @profile
    def process_t_alm(self, t_map, l_weight=None):
        if l_weight is None:
            assert self.l_weight is not None
            l_weight =  self.l_weight

        print("starting process_t_alm")
        t_alm = map2alm(t_map * self.map_fkp, lmax=self.lmax)
        weighted_alm = almxfl(t_alm, l_weight)
        print("done")

        return weighted_alm

    def compute_a_ksz_alm(self, t_pseudo_alm):
        comp_sum = (np.conjugate(self.vr_alm) * t_pseudo_alm).sum()
        print(comp_sum)
        a_ksz = np.abs(comp_sum)
        return a_ksz

    def get_noise_map(self):
        return enmap.ndmap(np.random.normal(size=self.imap_t.shape) * self.std_t, self.imap_t.wcs)

    # return a beamed map from our non-noise simulated cmb power spectrum
    def get_cl_tilde_alm(self):
        ret_alm = rand_alm(self.cl_tt_sim[:self.lmax + 1])
        ret_alm = almxfl(ret_alm, self.beam[:self.lmax + 1, 1])
        return ret_alm

    def get_pseudo_cl(self, cl, beam=False, noise=True):
        if beam:
            bl = self.beam[:self.lmax + 1,1]**2
        else:
            bl = np.ones(self.lmax + 1)
        return cl[:self.lmax + 1] * self.fl[:self.lmax + 1] * bl + noise * self.nl_tilde[:self.lmax + 1]

    def compute_sim_spectra(self, ntrial_fl=8, make_plots=False):

        noisemap = self.get_noise_map()
        self.nl = map2cl(noisemap, lmax=self.lmax_sim)

        # WARN: Noise map should not be beamed
        self.nl_tilde = map2cl(self.map_fkp * noisemap, lmax=self.lmax_sim)


        beam_sim2 = self.beam[:self.lmax_sim + 1,1]**2

        # TODO: repeat to reduce stochasticity on cl_tilde
        cl_test = 300. * 2 * np.pi * np.power(1./self.ells_sim, 2)
        cl_test[0] = 0.

        cl_tilde = np.zeros(self.lmax_sim + 1)
        for i in range(ntrial_fl):
            # cl_test = self.cl_cmb.copy()
            t_map = self.get_empty_map()
            t_alm = rand_alm(cl_test)

            t_alm = almxfl(t_alm, self.beam[:self.lmax_sim + 1,1])

            # t_alm = almxfl(t_alm, self.beam[:self.lmax_sim + 1,1])
            t_map = alm2map(t_alm, t_map) # look into pixel second arg
            cl_tilde += map2cl(t_map * self.map_fkp, lmax=self.lmax_sim)
        cl_tilde /= ntrial_fl

        # perform a pseudo/regular nl comparison
        plt.figure()
        ells = np.arange(self.lmax_sim + 1)
        norm = ells * (ells + 1) / 2 / np.pi

        plt.title('Comparison of pseudo and true nl')
        # plt.plot(ells, norm * self.nl, label='nl')
        plt.plot(ells, norm * self.nl_tilde, label='nl_pseudo')
        plt.plot(ells, norm * self.cl_tt_pseudo, label='cl_tt_pseudo')
        plt.yscale('log')
        plt.xlabel('l')
        plt.ylabel(r'$\Delta_l$ (muK^2)')
        plt.legend()
        plt.savefig('plots/compare_nl.png')    

        # self.fl_actual = (self.cl_tt_pseudo[:self.lmax + 1] - self.nl_tilde[:self.lmax + 1])/self.cl_cmb[:self.lmax + 1]
        # self.fl_actual[:50] = 0.

        self.fl = cl_tilde / cl_test
        self.fl[0] = self.fl[2000] # TODO: hack
        # self.fl[:50] = 0.

        self.cl_tt_sim = (self.cl_tt_pseudo[:self.lmax + 1] - self.nl_tilde[:self.lmax + 1]) / self.fl[:self.lmax + 1]
        self.cl_tt_sim[0] = 0.
        # self.cl_tt_sim[:50] = 0.

        # self.cl_cmb_tilde = self.fl * self.cl_cmb + self.nl_tilde
        # self.nl_sim = self.nl_tilde[:self.lmax + 1] / self.fl[:self.lmax + 1]
        # c_norm = self.cl_tt_pseudo[1000] / self.cl_cmb_tilde[1000]

        # check that the generated cl_obs matches the real cl_obs
        cl_xfer = self.get_pseudo_cl(self.cl_tt_sim)
        # cl_xfer = self.cl_tt_sim * self.fl[:self.lmax + 1] + self.nl_tilde[:self.lmax + 1]

        abs_diff = np.abs(cl_xfer[50:] - self.cl_tt_pseudo[50:])
        rel_err = 2 * abs_diff / np.abs(cl_xfer[50:] + self.cl_tt_pseudo[50:])

        plt.figure()
        plt.plot(abs_diff, label='absolute')
        plt.plot(rel_err, label='relative')
        plt.legend()
        plt.yscale('log')
        plt.ylabel('absolute/rel diff')
        plt.xlabel('l - 50')
        plt.savefig('plots/compare_cl_sim_obs.png')
        plt.close()

        assert fequal_either(cl_xfer[50:], self.cl_tt_pseudo[50:], tol=1e-2), f'max diff: {np.max(np.abs(cl_xfer[50:] - self.cl_tt_pseudo[50:]))}'

        if make_plots:
            norm = self.ells * (self.ells + 1) / 2 / np.pi

            plt.figure(dpi=300)
            plt.plot(self.fl / beam_sim2, label='computed transfer function')
            # plt.plot(self.fl_actual / beam_sim2[:self.lmax + 1], label='actual transfer function')
            plt.legend()
            plt.yscale('log')
            plt.grid(axis='both')
            plt.savefig('plots/cl_transfer.png')

            plt.figure(dpi=300)
            plt.title('pseudo noise plot')
            # plt.plot(self.nl[:self.lmax + 1] * norm, label='nl')
            plt.plot(self.nl_tilde[:self.lmax + 1] * norm, label='nl_tilde')
            plt.legend()
            plt.ylabel(r'$\eta^2 l (l + 1)/2\pi\,\mathrm{\mu K}^2$')
            plt.xlabel(r'$l$')
            plt.savefig('plots/nl_tilde.png')

            plt.figure(dpi=300)
            ells_plot = np.arange(500,self.lmax + 1,1)
            nells = len(ells_plot)
            ell0 = ells_plot[:nells//2]
            ell1 = ells_plot[nells//2:]

            cl_plot = self.cl_tt_pseudo.copy()
            cl_plot[:500] = 0.
            plt.grid(which='both')
            # plt.plot(self.cl_cmb[:self.lmax + 1] * beam_sim2[:self.lmax + 1], label='cl_sim')
            plt.plot(ell0, self.cl_tt_pseudo[ell0] * norm[ell0],
                     alpha=1., linewidth=1, label='cl_obs_pseudo')
            plt.plot(ell1, cl_xfer[ell1] * norm[ell1],
                     alpha=1., linewidth=1, label='cl_sim_fl')
            # plt.plot((self.cl_tt_pseudo - self.nl_tilde[:self.lmax + 1]) * norm, label='cl_obs_minus_eta2')
            # plt.plot(ells_plot, self.cl_cmb_tilde[ells_plot] * norm[ells_plot], label='cl_pseudo_sim')
            # plt.plot(c_norm * self.cl_cmb_tilde[:self.lmax + 1] * norm, label='cl_pseudo_sim shifted')
            plt.yscale('log')
            plt.legend()
            plt.savefig('plots/cl_pseudo_compare.png')

        # TODO: remove        
        # self.fl = self.fl_actual.copy()

        # cl_zz_obs = map2cl(self.z_theta, self.lmax)

    def compute_l_weight(self, r_lwidth=1., s=0):
        ells_sim = np.arange(self.lmax_sim + 1)
        cl_ksz_th = np.exp(-(self.ells_sim / r_lwidth / 5500.)**2)

        if self.custom_l_weight:
            cl_ave = np.load(self.custom_l_weight)
            print(cl_ave)
            ells_sparse = cl_ave[0] + 1 # the ells are shifted implicitly
            cl_xz_noisy = cl_ave[1]
            cl_spl = splrep(ells_sparse, cl_xz_noisy, s=s)
            cl_eval = splev(ells_sim, cl_spl)
            cl_eval[0] = 0.
            cl_eval[8000:] = 0.
            cl_eval[:50] = 0.

            plt.figure(dpi=300)
            plt.plot(ells_sparse, cl_xz_noisy, label='noisy cl_xz')
            plt.plot(ells_sim, cl_eval, label='interpolated cl_xz')
            plt.savefig('plots/custom_l_weight.png')
            plt.close()

            self.l_weight = cl_eval
        else:
            # WARN: noise-free
            cl_xx_weight = self.get_pseudo_cl(self.cl_tt_sim, beam=True, noise=True)

            cl_pseudo_ksz = self.get_pseudo_cl(cl_ksz_th, beam=True)

            cl_zz = 1.

            self.l_weight = cl_ksz_th / (cl_xx_weight * cl_zz)

        self.l_weight /= self.l_weight[self.l_ksz]

        # WARN: l_weight cut
        # WARN: incompatible with custom l-weight?
        self.l_weight[:1500] = 0.

    def get_sim_map(self, l_weight=None):
        if l_weight is None:
            assert self.l_weight is not None
            l_weight =  self.l_weight

        t_map = self.get_empty_map()
        
        # generate a (beamed) cmb realization
        t_alm = self.get_cl_tilde_alm()

        # convert to map
        # t_alm = almxfl(t_alm, pipe.beam[:pipe.lmax_sim + 1, 1])
        t_map = alm2map(t_alm, t_map)

        # print('Injecting noise in map space')
        t_map += self.get_noise_map()
        t_pseudo = self.process_t_map(t_map, l_weight)
        return t_pseudo


def reproj_planck_map(map_inpath, mask_inpath, outpath, mode='GAL090', nside=PLANCK_NSIDE):
    imap = enmap.read_map(map_inpath)
    imap_t = imap[0]

    this_mask = np.array(fits.getdata(mask_inpath, ext=0)[mode]).astype(bool)
    this_mask = hp.pixelfunc.reorder(this_mask, n2r=True)
    enmap_mask = enmap_from_healpix_interp(this_mask, imap_t.shape,
                            imap_t.wcs, interpolate=True)
    np.save(outpath, enmap_mask)

    plt.figure(dpi=300)
    # plt.imshow(enmap_mask)
    # plt.colorbar()
    plt.savefig('plots/mask_test.pdf')


# ksz spectrum source: https://arxiv.org/pdf/1301.0776.pdf
# just a single-use throwaway function to generate an approximate kSZ spectrum
def explore_pksz(r):
    # n = 1025
    # l = np.linspace(0,12000,n)
    l = np.arange(LMAX)
    y = 2.25*(1 - np.exp(-r*l))

    plt.figure(dpi=300)
    plt.plot(l,y)
    plt.title('fiducial kSZ power')
    plt.yscale('log')
    plt.xlabel('l')
    plt.axhline(1.)
    plt.axvline(3000)
    ax = plt.gca()
    ax.set_aspect((1./6) * 11000)
    plt.ylabel(r'$\frac{l(l+1)}{2\pi}C_l$')
    plt.savefig('plots/ksz_power.pdf')

    # out = np.empty((n-1,2))
    out = np.empty((LMAX,2))
    out[:,0] = l
    out[:,1] = y # l (l+1)/2pi normalized
    np.save(cl_ksz_path, out)


def one_time_setup(data_path='/data/'):
    reproj_planck_map(map_path, planck_mask_inpath, planck_enmask_path, mode='GAL080')


setup = False


def make_xz_plot(act_pipe, gal_pipe, r_fkp=0.62, r_lweight=1.56, 
                 do_trials=False, ntrial=4, ncl_bins=128, lmax_plot=None):
    pipe = act_pipe

    if lmax_plot is None:
        lmax_plot = pipe.lmax

    assert lmax_plot <= pipe.lmax

    pipe.compute_pixel_weight(r_fkp=r_fkp)
    pipe.compute_sim_spectra(ntrial_fl=1, make_plots=True)

    # filter large scale modes
    ones = np.ones(pipe.lmax+1)
    ones[:100] = 0.

    # apply fkp weighting to ACT map, return in harmonic space
    t_fkp_alm = pipe.process_t_alm(pipe.imap_t, ones)

    # TODO consolidate the vr_list and vr_map stuff into convenience functions
    vr_map = pipe.get_zero_map()
    gal_pipe.make_vr_list()

    gal_inds = gal_pipe.gal_inds
    vr_map[gal_inds[0], gal_inds[1]] += gal_pipe.vr_list
    vr_alm = map2alm(vr_map, lmax=pipe.lmax)

    cl_xz_pseudo = alm2cl(t_fkp_alm, vr_alm)

    # nbin_sum = lmax_plot // ncl_bins
    nbin_sum = pipe.lmax // ncl_bins


    lmax_comp = nbin_sum * ncl_bins


    ells_comp = np.arange(0, lmax_comp, ncl_bins)
    cl_xz_averaged = cl_xz_pseudo[1:lmax_comp + 1].reshape(lmax_comp // ncl_bins, ncl_bins).sum(axis=1) / ncl_bins

    cl_dump = np.empty((2, nbin_sum))
    cl_dump[0] = ells_comp
    cl_dump[1] = cl_xz_averaged
    print(cl_dump)
    np.save(f'cl_xz_averaged_{ncl_bins}.npy', cl_dump)

    if do_trials:
        cl_xz_samples = np.empty((ntrial, nbin_sum))

        print('doing <xz> noise trials')
        for i in range(ntrial):
            sim_map = pipe.get_sim_map(l_weight=ones)
            sim_alm = map2alm(sim_map, lmax=pipe.lmax)
            cl_raw = alm2cl(sim_alm, vr_alm)

            cl_xz_samples[i] = cl_raw[1:lmax_comp + 1].reshape(lmax_comp // ncl_bins, ncl_bins).sum(axis=1) / ncl_bins
            print(f'finished sim map {i + 1} of {ntrial}')
        print('done')

        cl_stds = np.std(cl_xz_samples, axis=0)

    plt.figure(dpi=300.)

    ells = np.arange(0, lmax_plot, ncl_bins)
    imax = np.argmin((np.arange(0, lmax_comp, ncl_bins) - lmax_plot)**2) + 1
    norm = ells * (ells + 1) / 2 / np.pi

    plt.title('ACT x SDSS Cross Spectrum')
    plt.plot(ells, norm * (cl_xz_averaged[:imax]), label=f'Bin $l$-Width: {ncl_bins}', color='blue')

    if do_trials:
        plt.fill_between(ells, norm * ((cl_xz_averaged - cl_stds)[:imax]), norm * ((cl_xz_averaged + cl_stds)[:imax]),
                         alpha=0.5, color='blue')

    plt.axhline(y=0, color='black')
    plt.legend()

    plt.grid(which='both')
    plt.ylabel(u'$\Delta_l^{XZ}$')
    plt.xlabel(u'$l$')
    plt.savefig('plots/cl_xz_averaged.png')
    plt.close()


# @profile
# TODO: extend to multi-frequency case!
def compute_estimator(act_pipes, gal_pipe, r_lwidth=1., do_mc=False, ntrial=64):

    # set l weight

    # TODO: check cl_zz assumption holds

    # placeholder, in the future we will iterate over frequencies
    pipe = act_pipes[0]
    
    # process the galaxy radial velocity data into map index space
    gal_pipe.make_vr_list()

    cl_zz = np.ones(pipe.lmax + 1)
    # TODO: find correct form for low-l cl_zz!
    # pipe.cl_zz[:pipe.l_ksz] = 0.
    # prove this statement
    # Is l >> the correlation field scale ?
    # Investigate power spectrum to see if cl_zz assumption is valid
    # pipe.cl_zz *= np.var(pipe.v_rad) / pipe.ngal2d # TODO: correct ngal2d

    # pipe.z_theta = pipe.get_zero_map()

    # TODO: Choose l-split that gives same SNR for both bins

    # optimal weighting for the alpha estimator

    # substitute cl_masked for pseudo cl with fkp
    # cl_use = pipe.get_pseudo_cl(pipe.cl_cmb)

    ells_sim = np.arange(12000 + 1)
    cl_ksz_th = np.exp(-(ells_sim / r_lwidth / 5500.)**2)

    # WARN: noise-free
    cl_use = pipe.get_pseudo_cl(pipe.cl_tt_sim, beam=True, noise=True)
    cl_use_nobeam = pipe.get_pseudo_cl(pipe.cl_tt_sim, beam=False)

    cl_pseudo_ksz = pipe.get_pseudo_cl(cl_ksz_th, beam=True)
    cl_pseudo_ksz_nobeam = pipe.get_pseudo_cl(cl_ksz_th, beam=False)


    ell_plot = np.arange(500, pipe.lmax + 1,1)
    # plt.figure(dpi=300)
    # plt.title('l weight components')
    # plt.yscale('log')
    # plt.plot(ell_plot, pipe.beam[ell_plot,1], label='beam')
    # plt.plot(ell_plot, pipe.cl_ksz[ell_plot] / pipe.cl_ksz[3000], label='ksz')
    # plt.plot(ell_plot, cl_use[3000] / cl_use[ell_plot], label='cmb')
    # plt.xlabel('l')
    # plt.ylabel('amplitude (arbitrary)')
    # plt.legend()
    # plt.savefig('plots/l_weight_compare.png')
    # plt.close()


    # WARN: replace
    # l_weight = pipe.beam[:pipe.lmax + 1,1] * pipe.cl_ksz[:pipe.lmax + 1,1] / (cl_use * cl_zz)
    # l_weight = cl_ksz_th / (cl_use * cl_zz)
    # l_weight /= l_weight[pipe.l_ksz]

    # # WARN: l_weight cut
    # l_weight[:1500] = 0.

    # l_weight_nobeam = pipe.beam[:pipe.lmax + 1,1] * cl_pseudo_ksz_nobeam[:pipe.lmax + 1] / (cl_use_nobeam * cl_zz)
    # l_weight_nobeam /= l_weight_nobeam[pipe.l_ksz]

    # l_weight_nobeam[:1500] = 0

    # plt.figure(dpi=300)
    # plt.plot(l_weight, label='beam')
    # plt.plot(l_weight_nobeam, label='no beam')
    # plt.xlabel('l')
    # plt.ylabel('amplitude (arbitrary)')
    # plt.legend()
    # plt.savefig('plots/l_weight.png')
    # plt.close()

    pipe.compute_l_weight(r_lwidth=r_lwidth, s=0.)
    l_weight = pipe.l_weight

    t_fkp_est = pipe.process_t_map(pipe.imap_t, l_weight)

    a_ksz = gal_pipe.compute_a_ksz(t_fkp_est)
    # a_ksz_noweight = gal_pipe.compute_a_ksz(t_fkp_noweight)

    a_std_sq = np.sqrt(((gal_pipe.get_xz_list(t_fkp_est))**2).sum())
    # gal_mask = pipe.vr_map != 0.
    # assert gal_mask.sum() == pipe.ngal_in, f'{len(pipe.gal_pos)} {gal_mask.sum()} {pipe.ngal_in}'

    assert gal_pipe.gal_inds.shape[1] == gal_pipe.ngal_in

    var_vr = np.var(gal_pipe.vr_list)
    var_t_obs = np.var(t_fkp_est[gal_pipe.gal_inds[0,:], gal_pipe.gal_inds[1,:]])

    var_t_all = np.var(t_fkp_est)

    # WARN: almost exact
    a_std_ind = np.sqrt(gal_pipe.ngal_in * var_vr * var_t_obs)


    # a_std_noweight = np.sqrt((gal_pipe.get_xz_list(t_fkp_noweight)**2).sum())
    # print(f'a_noweight {a_ksz_noweight / a_std_noweight:.3f}')
    print(f'a_tv2: {a_ksz / a_std_sq:.3f}')
    print(f'a_t2v2: {a_ksz / a_std_ind:.3f}')

    ls = np.arange(pipe.lmax + 1)

    print(f'ksz_unnormalized: {a_ksz:.4e}')
    # # TODO: check cl tilde edge effects

    # # TODO: move xz plot elsewhere!!
    # ells = np.arange(pipe.lmax + 1)
    # ellnorm = ells * (ells + 1) / 2 /np.pi
    # # Compute and plot cl^{xz}

    # cl_zz = alm2cl(gal_pipe.vr_alm)
    # cl_xx = alm2cl(t_fkp_alm)
    # cl_xx[0] = 0.
    # cl_xz = alm2cl(gal_pipe.vr_alm, t_fkp_alm)
    # cl_xz_std = np.sqrt(cl_xx * cl_zz / (2 * ells + 1))

    # plt.figure(dpi=300)
    # plt.plot(ells, ellnorm * cl_xx)
    # plt.savefig('plots/cl_xx.png')
    # plt.close()

    # # cl_xz[0] = 0

    # bin_width = 500
    # cl_xz_bin = cl_xz[:-1].reshape(bin_width, pipe.lmax // bin_width).sum(axis=0)/bin_width
    # cl_xz_std_bin = np.sqrt((cl_xz_std[:-1]**2).reshape(bin_width, pipe.lmax // bin_width).sum(axis=0))/bin_width
    # bin_centers = np.arange(bin_width//2, pipe.lmax, bin_width)

    # plt.figure(dpi=300)
    # plt.plot(ells, cl_xz)
    # plt.fill_between(ells, cl_xz - cl_xz_std, cl_xz + cl_xz_std, alpha=0.5, color='red')
    # plt.savefig('plots/cl_xz.png')
    # plt.close()

    # plt.figure(dpi=300)
    # plt.title(r'$C_l^{XZ}$')
    # # plt.errorbar(ells, cl_xz, yerr = cl_xz_std)
    # # plt.errorbar(ells, cl_xz, yerr = cl_xz * np.sqrt(2/(2 * ells + 1)))
    # plt.plot(bin_centers, cl_xz_bin)
    # # plt.plot(ells, cl_xz_std)
    # # plt.plot(ells, cl_xz * np.sqrt(2 / (2 * ells + 1)))
    # plt.fill_between(bin_centers, cl_xz_bin - cl_xz_std_bin, cl_xz_bin + cl_xz_std_bin, alpha=0.5, color='red')
    # plt.axhline(0.)

    # # plt.yscale('log')
    # # plt.errorbar(ells, cl_xz)
    # plt.xlabel(r'$l$')
    # plt.ylabel(r'$C_l$ (arbitrary)')
    # plt.savefig('plots/cl_xz_binned.png')
    # plt.close()
    if not do_mc:
        return 0.5 * a_ksz * (1/a_std_sq + 1/a_std_ind), 2 * (a_std_sq * a_std_ind) / (a_std_sq + a_std_ind)
    else:
        a_samples = np.zeros(ntrial)

        # TODO: plot cl_sim vs cl_camb
        # low l disagreement is to be expected
        # high l: should see more power than camb
        # compare fkp weighted pseudo-cl of sim to fkp weighted pseudo cl of data
        for i in range(ntrial):
            print("running variance trial {} of {}".format(i + 1, ntrial))
            t_map = pipe.get_empty_map()
            
            # generate a (beamed) cmb realization
            t_alm = pipe.get_cl_tilde_alm()

            # convert to map
            # t_alm = almxfl(t_alm, pipe.beam[:pipe.lmax_sim + 1, 1])
            t_map = alm2map(t_alm, t_map)

            # print('Injecting noise in map space')
            t_map += pipe.get_noise_map()
            t_pseudo = pipe.process_t_map(t_map, l_weight)

            # make diagnostic plots on first trial
            if i == 0:
                t_map_plot = t_map * pipe.map_fkp
                # make diagnostic plot to compare cl
                cl_tilde_sim = map2cl(t_map_plot, lmax=pipe.lmax)
                cl_pseudo_sim = map2cl(t_pseudo, lmax=pipe.lmax)
                plt.figure(dpi=300)
                plt.title('Comparison of noise map vs ACT power spectra')
                plt.yscale('log')
                norm = pipe.ells * (1 + pipe.ells) / 2 / np.pi
                plt.plot(pipe.ells, cl_pseudo_sim * norm, label='cl from weighted map')
                plt.plot(pipe.ells, pipe.cl_tt_pseudo[:pipe.lmax + 1] * norm * l_weight[:pipe.lmax + 1]**2,
                         label='expected cl_tt for weighted map')
                plt.plot(pipe.ells, cl_tilde_sim * norm, label='cl from noise map')
                plt.plot(pipe.ells, pipe.cl_tt_pseudo[:pipe.lmax + 1] * norm, label='cl_tt_pseudo')
                plt.xlabel('l')
                plt.ylabel(u'$\Delta_l$')
                plt.legend()
                plt.savefig('plots/cl_sim_compare.png')
                plt.close()

                plt.figure(dpi=300)
                plt.title('Ratio of ACT pseudo cl to sim map cl')
                cl_ratio = pipe.cl_tt_pseudo[:pipe.lmax + 1] / cl_tilde_sim
                cl_ratio[0] = 1.
                plt.plot(pipe.ells[50:], pipe.cl_tt_pseudo[50:pipe.lmax + 1] / cl_tilde_sim[50:])
                plt.xlabel('l')
                plt.savefig('plots/cl_ratio.png')
                plt.close()

            # eval t_pseudo at galaxy locs
            t_gal_list = gal_pipe.get_t_list(t_pseudo)

            # TODO: update estimator 
            a_std_sq = np.sqrt(((gal_pipe.get_xz_list(t_pseudo))**2).sum())

            # assert gal_mask.sum() == pipe.ngal_in, f'{len(pipe.gal_pos)} {gal_mask.sum()} {pipe.ngal_in}'

            var_t = np.var(t_gal_list)
            var_t_all_sim = np.var(t_pseudo)

            print(f'var(T), var(T_obs) {var_t:.3e} {var_t_obs:.3e}')
            print(f'var(T_all), var(T_all_obs) {var_t_all:.3e} {var_t_all_sim:.3e}')

            a_std_ind = np.sqrt(gal_pipe.ngal_in * var_vr * var_t)

            a_samples[i] = gal_pipe.compute_a_ksz(t_pseudo)

            if i > 2:
                sample_std = np.std(a_samples[:i + 1])
            else:
                sample_std = 1.
            print(f'{a_ksz:.3e} {a_samples[i]:.3e} {a_ksz / sample_std:.3e} {a_ksz / a_std_sq:.3e} {a_ksz / a_std_ind:.3e}')
        
        print(a_samples)
        a_sigma = np.sqrt(np.var(a_samples))
        a_mean = np.mean(a_samples)
        print("a_ksz, a_sigma, <a>, sigma: {:.2e}, {:.2e}, {:.2e}, {:.2e}".format(a_ksz, 
                                                      a_sigma, a_mean, a_ksz/a_sigma))
        return a_ksz/a_sigma, a_sigma


def do_single_eval(act_pipe, gal_pipe, r_fkp, r_lwidth, suppl_mask=None, ntrial=64):
    act_pipe.compute_pixel_weight(r_fkp=r_fkp, suppl_mask=suppl_mask)
    act_pipe.compute_sim_spectra(ntrial_fl=1, make_plots=True)
    act_pipes = [act_pipe,]

    ahat, a_sigma = compute_estimator(act_pipes, gal_pipe, r_lwidth=r_lwidth, do_mc=True, ntrial=ntrial)
    return ahat, a_sigma

def do_parameter_loop(act_pipe, gal_pipe, range_fkp=[0.1, 10], n_fkp=10,
                      range_lwidth=[0.53, 0.53], n_width=1):
    sigma_mat = np.empty((n_fkp, n_width))

    r_fkps = np.linspace(*range_fkp, n_fkp)
    r_lwidths = np.linspace(*range_lwidth, n_width)

    print('starting fkp/lwidth optimization loop')
    for i, r_fkp in zip(range(n_fkp), r_fkps):
        act_pipe.compute_pixel_weight(r_fkp=r_fkp)
        act_pipe.compute_sim_spectra(ntrial_fl=1, make_plots=True)
        act_pipes = [act_pipe,]

        for j, r_lwidth in zip(range(n_width), r_lwidths):
            ahat, a_sigma = compute_estimator(act_pipes, gal_pipe, r_lwidth=r_lwidth, do_mc=False)
            print('====================================================')
            print(f'r_fkp {r_fkp:.2f} r_lwidth {r_lwidth:.2f} a_hat {ahat:.2f}')
            print('====================================================')
            sigma_mat[i,j] = ahat

    print('finished fkp optimization')

    plt.figure(dpi=300)
    plt.imshow(sigma_mat)
    plt.title('kSZ SNR')
    plt.xlabel('l width param')
    plt.ylabel('fkp param')
    ax = plt.gca()

    def pad_label(ar):
        return [''] + ar + ['']

    xticks = ax.get_xticks().astype(int)[1:-1]
    xlabels = pad_label([f'{a:.2f}' for a in r_lwidths[xticks]])
    ax.set_xticklabels(xlabels)

    yticks = ax.get_yticks().astype(int)[1:-1]
    ylabels = pad_label([f'{a:.2f}' for a in r_fkps[yticks]])
    ax.set_yticklabels(ylabels)

    cax = plt.colorbar()
    plt.savefig('plots/optimize_fkp_width.png')
    plt.close()


def import_sdss_footprint(sdss_footprint_path):
    footprint = enmap.read_map(sdss_footprint_path)
    return footprint


def eshow(x,**kwargs):
    plots = enplot.get_plots(x, **kwargs)
    enplot.show(plots)

def do_sdss_field_noise_comparison():
    mymap = import_sdss_footprint('/home/aroman/data/sdss_footprint/pixellized_sdss_north_completeness.fits')
    mymap += import_sdss_footprint('/home/aroman/data/sdss_footprint/pixellized_sdss_south_completeness.fits')

    # eshow(mymap)
    plot = enplot.get_plots(mymap, downgrade=8)[0]
    # WARN: enplot axes are absent?
    enplot.write('plots/sdss_footprint.tiff', plot)


    # hpx = healpix_from_enmap(mymap, lmax=6000, nside=1024)

    # hdu = fits.open('/home/aroman/data/sdss_footprint/pixellized_sdss_south_completeness.fits')[0]

    # wcs = WCS(hdu.header)

    # plt.subplot(projection=wcs)
    # plt.figure(dpi=300)
    # ax = plt.gca()
    # # ax = plt.axes(projection='geo aitoff')
    # # lon = ax.coords[0]
    # # lat = ax.coords[1]

    # @ticker.FuncFormatter
    # def major_formatter_x(x, pos):
    #     dec, ra = enmap.pix2sky(mymap.shape, mymap.wcs, [0., x])
    #     # print(ra, pos)
    #     return f'{ra:.1f}'

    # @ticker.FuncFormatter
    # def major_formatter_y(y, pos):
    #     dec, ra = enmap.pix2sky(mymap.shape, mymap.wcs, [y, 0.])
    #     # print(ra, pos)
    #     return f'{dec:.1f}'

    # ax.xaxis.set_major_formatter(major_formatter_x)
    # ax.yaxis.set_major_formatter(major_formatter_y)
    # # lon.set_major_formatter(major_formatter)
    # plt.imshow(mymap)
    # # ax.imshow_hpx(hpx)
    # ax.grid()

    # # cax = plt.colorbar()
    # plt.savefig('plots/sdss_footprint.png')
    # plt.close()


def check_xz_gaussian(xz_path, beam=None):
    ar = np.load(xz_path)
    ells = ar[0].astype(int)
    cl_ave = ar[1]

    print(cl_ave)

    if beam is not None:
        cl_ave = cl_ave / beam[ells]

    cl_ave = cl_ave / cl_ave[1]

    def expfun(ells, x):
        ret = x[0] * np.exp(-(ells/x[1])**2)
        # print(ret)
        return ret

    def minfun(x):
        ret = np.sum((expfun(ells, x)[1:] - cl_ave[1:])**2)
        # print(ret)
        return ret

    x0 = np.array([1., 700.])

    res = minimize(minfun, x0, method='BFGS')
    print(res)
    xmin = res.x

    plt.figure(dpi=300.)
    plt.plot(ells, expfun(ells, xmin), label='fit gaussain')
    plt.plot(ells[1:], cl_ave[1:], label='cl_xz')
    plt.xlabel('l')
    plt.ylabel('Amplitude (normalized)')
    plt.title(u'Comparison of $C_l^{TZ}$ and Gaussian Fit')
    plt.legend()
    plt.savefig('plots/cl_xz_fit_comparison')
    plt.close()


def plot_zz(act_pipe, gal_pipe):
    gal_pipe.make_vr_list()

    imap = act_pipe.get_zero_map()
    inds = gal_pipe.gal_inds
    
    imap[inds[0,:], inds[1,:]] += gal_pipe.vr_list

    cl_zz = map2cl(imap, lmax=act_pipe.lmax)

    cl_zz[:50] = 0


    ells = np.arange(act_pipe.lmax + 1)
    norm = ells * (ells + 1) / 2 / np.pi
    plt.figure(dpi=300)
    plt.title(u'$C_l^{ZZ}$')
    plt.plot(ells, cl_zz)
    plt.xlabel(u'l')
    plt.ylabel(u'$C_l^{ZZ}$')
    plt.savefig('plots/cl_zz.png')
    plt.close()

# TODO: plot HP filtered map and search for 'striping'
# TODO: cross spectrum is suppressed at small scales
# TODO: estimate cross-power in a few l-bins, instead
# TODO: z-dependent weighting? Divide SDSS into redshift bins and see if alpha changes
if __name__ == "__main__":
    if setup:
        one_time_setup()

    # do_sdss_field_noise_comparison()

    map_freq = ['090', '150']

    # TODO: loop over freqs

    # act_pipe = ActCMBPipe(map_path, ivar_path, beam_path, cl_ksz_path, cl_cmb_path,
    #                       planck_enmask_path,
    #                       custom_l_weight='cl_xz_averaged_256.npy', diag_plots=True, lmax=LMAX)

    act_pipe = ActCMBPipe(map_path, ivar_path, beam_path, cl_ksz_path, cl_cmb_path,    
                          planck_enmask_path,
                          custom_l_weight=None, diag_plots=True, lmax=LMAX)


    act_pipe.import_data()

    gal_pipe = GalPipe(catalog_path, act_pipe, diag_plots=True)
    gal_pipe.import_data()

    # sdss_footprint = import_sdss_footprint('/home/aroman/data/sdss_footprint/pixellized_sdss_north_completeness.fits')
    # sdss_footprint += import_sdss_footprint('/home/aroman/data/sdss_footprint/pixellized_sdss_south_completeness.fits')

    # # ahat_sdss, a_sigma_sdss = do_single_eval(act_pipe, gal_pipe, r_fkp=1.56, r_lwidth=0.62, ntrial=256, suppl_mask=sdss_footprint)
    # # print(f'SDSS footprint constrained ahat: {ahat_sdss:.2f}, a_sigma: {a_sigma_sdss:.3e}')

    # ahat, a_sigma = do_single_eval(act_pipe, gal_pipe, r_fkp=1.56, r_lwidth=0.62, ntrial=256, suppl_mask=None)
    # print(f'SDSS footprint constrained ahat: {ahat_sdss:.2f}, a_sigma: {a_sigma_sdss:.3e}')
    # print(f'SDSS footprint constrained ahat: {ahat:.2f}, a_sigma: {a_sigma:.3e}')
    do_parameter_loop(act_pipe, gal_pipe, range_fkp=[1.56, 1.56], n_fkp=1,
                      range_lwidth=[0.62, 0.62], n_width=1)
    # make_xz_plot(act_pipe, gal_pipe, r_fkp=0.62, ncl_bins=256,
                   # do_trials=False, ntrial=512, lmax_plot=8000)

    # check_xz_gaussian('cl_xz_averaged_256.npy', act_pipe.beam[:,1])
    # check_xz_gaussian('cl_xz_averaged_256.npy')

    # plot_zz(act_pipe, gal_pipe)