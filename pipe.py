import pixell
from pixell import enmap, enplot, utils, curvedsky
from pixell.curvedsky import map2alm, alm2map, rand_alm, alm2cl, almxfl, rand_map
from pixell.reproject import enmap_from_healpix, enmap_from_healpix_interp
import healpy as hp
from scipy.special import sph_harm

import astropy
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.coordinates import SkyCoord

import h5py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate


# from mpi4py import MPI

# class DirTree:
#     def __init__(self, data_path, map_stem):
#         self.data_path = data_path
#         self.map_stem

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
sdss_catalog_path = data_path + 'vr_source/v01/desils/v01_desils_south_cmass.h5'

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


# WARN: assumes rigid reference geometry across maps
class GalPipe:
    def __init__(self, sdss_catalog_path, ref_map_t, import_now=False, diag_plots=True, lmax=LMAX):
        self.lmax = int(lmax)
        self.sdss_catalog_path = sdss_catalog_path
        self.diag_plots = diag_plots

        if(import_now):
            self.import_data()

    def import_data(self):
        print("importing galaxy catalog")

        gal_ext = get_ext(self.sdss_catalog_path)

        assert(gal_ext == 'h5' or gal_ext == 'fits')

        if gal_ext == 'h5':
            with h5py.File(self.sdss_catalog_path, 'r') as f:
                self.v_rad = f['vr_smoothed'][:]
                self.ngal2d = len(self.v_rad)

                self.gal_pos = np.zeros((self.ngal2d, 2))
                self.gal_pos[:,0] = f['dec_deg'][:] * (np.pi / 180)
                self.gal_pos[:,1] = f['ra_deg'][:] * (np.pi / 180)


        elif gal_ext == 'fits':
            with fits.open(self.sdss_catalog_path) as hdulist:
                table = hdulist[1].data
                self.ngal2d = len(table)

                self.v_rad = table['VR']
                self.gal_pos = np.zeros((self.ngal2d, 2))
                self.gal_pos[:,0] = table['DEC_DEG'] * (np.pi / 180)
                self.gal_pos[:,1] = table['RA_DEG'] * (np.pi / 180)

        print("done")

    def make_vr_list(self, ref_map_t, make_alm=False):
        if make_alm:
            self.vr_alm = make_zero_alm(ref_map_t, lmax=self.lmax)

        self.vr_map = make_zero_map(ref_map_t)

        angular_res = ref_map_t.wcs.wcs.cdelt[0] * np.pi / 180.

        ngal = len(self.gal_pos)

        corners = ref_map_t.corners()

        if self.diag_plots:
            plt.figure(dpi=300)
            plt.scatter(self.gal_pos[:,1] * 180 / np.pi, self.gal_pos[:,0] * 180 / np.pi,
                        marker='+', s=1)
            plt.axhline(corners[0,0] * 180 / np.pi, color='red')
            plt.axhline(corners[1,0] * 180 / np.pi, color='red')
            # plt.axvline(corners[1,0] * 180 / np.pi, color='red')
            # plt.axvline(corners[1,1] * 180 / np.pi, color='red')
            plt.ylabel('Declination (deg)')
            plt.xlabel('Right Ascension (deg)')
            plt.savefig('plots/galaxy_locs.png')
            plt.close()

        gal_inds = []
        vr_list = []

        n_ob = 0

        for i, (v_r, pos) in enumerate(zip(self.v_rad, self.gal_pos)):
            dec, ra = pos
            idec, ira = iround(ref_map_t.sky2pix((dec,ra)))
            # idec, ira = self.vr_map.sky2pix((dec,ra))

            if not bounds_check((idec,ira), ref_map_t.shape):
                # print(dec, ra, idec, ira)
                n_ob += 1
            else:
                # TODO: check galaxy overlap?
                gal_inds.append([idec, ira])
                vr_list.append(v_r)
                self.vr_map[idec, ira] += v_r
                # this method is super inefficient
                # self.vr_alm[:self.lmax + 1] += v_r * sph_harm(ms, ls, ra, dec)
            # print('Processed galaxy {} of {}'.format(i + 1, ngal))
        self.gal_inds = np.array(gal_inds).T
        self.vr_list = np.array(vr_list)
        self.ngal_in = ngal - n_ob
        print('Fraction of out-of-bounds galaxies: {:.2f}'.format(float(n_ob + 1) / ngal))
        if make_alm:
            pixel_area = angular_res**2
            self.vr_alm = map2alm(self.vr_map / pixel_area, lmax=self.lmax)

    def get_t_list(self, t_map):
        return t_map[self.gal_inds[0,:], self.gal_inds[1,:]]

    def get_xz_list(self, t_map):
        t_gal_list = t_map[self.gal_inds[0,:], self.gal_inds[1,:]]
        return t_gal_list * self.vr_list

    def get_xz_map(self, t_map):
        # ret_map = make_zero_map(t_map)
        # # WARN: this is unsafe
        # ret_map[self.gal_inds[0,:], self.gal_inds[1,:]] += self.vr_list
        # ret_map *= t_map
        return self.vr_map * t_map

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

        self.imap = None
        self.imap_t = None
        self.ivar = None
        self.sim_map = None

    def import_data(self):
        print("importing map: {}".format(self.map_path))
        self.imap = enmap.read_map(self.map_path)
        self.imap_t = self.imap[0]

        # based on corners, store galactic plane location in celestial pixel space
        # useful for imshow masking

        print("done")

        print("generating zero_alm array")
        zero_map = self.get_zero_map()
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

        # WARN: note this is off by an amplitude factor
        # TODO: verify piecewise ksz
        self.cl_ksz = np.empty(self.lmax_sim + 1)
        self.cl_ksz[1:2000] = 2 * np.pi / 1000. / np.arange(1,2000,1)
        self.cl_ksz[2000:] = 4 * np.pi / (self.ells_sim[2000:]**2)
        self.cl_ksz[0] = 2 * np.pi / 1000.

        norm = self.ells_sim * (1 + self.ells_sim) / 2 / np.pi

        plt.figure(dpi=300)
        plt.title('kSZ Theory Spectrum')
        plt.ylabel(r'$\Delta_l$ (uK$^2$)')
        plt.xlabel(r'$l$')
        plt.plot(self.ells_sim, self.cl_ksz * norm)
        plt.savefig('plots/cl_theory.png')
        plt.close()

        cl_cmb = np.load(self.fid_cmb_path)[:self.lmax_sim + 1,1]
        # l = np.arange(len(cl_cmb))
        print('cmb lmax:', self.lmax_sim)
        self.cl_cmb = cl_cmb.copy()
        self.cl_cmb[0] = 0.

    def get_zero_map(self):
        return enmap.ndmap(np.zeros(self.imap_t.shape), self.imap_t.wcs)
    
    def get_zero_alm(self):
        return self.zero_alm.copy()

    # compute w_fkp_theta (FKP weight considering inverse var and mask)
    # map form: map_fkp
    def compute_pixel_weight(self, l0=None):
        if l0 is None:
            l0 = self.l_fkp
            print(f"No FKP l scale provided, using l_fkp={self.l_fkp}")
        b2_l0 = self.beam[l0,1]**2
        ctt_l0 = self.cl_cmb[l0] # this comes from CAMB
        ctt_3k_act = 24.8 * 2 * np.pi / 3000 / (3000 + 1) # vs 51.9 for d_l^{TT} coadd

        # TODO: redo mode-mixing analysis

        assert np.all(self.ivar_t >= 0)

        # TODO: verify normalization!!
        eta_n2 = self.ivar_t / self.pixel_area

        # zero masked pixels
        eta_n2 *= self.mask_t

        self.eta_n2 = eta_n2
        print(f'Eta^-2 min/max: {np.min(eta_n2):.3e} {np.max(eta_n2):.3e}')

        # compute FKP pixel weight
        # self.w_fkp_theta = eta_n2 / (1 + b2_l0 * ctt_l0 * eta_n2)
        self.w_fkp_theta = eta_n2 / (1/(b2_l0 * ctt_3k_act) + eta_n2)
        print(f'FKP min/max: {np.min(self.w_fkp_theta):.3e} {np.max(self.w_fkp_theta):.3e}')
        self.map_fkp = enmap.ndmap(self.w_fkp_theta, self.imap_t.wcs)

        # box = np.array([[0.1, -6], [-1.1, -7]])
        # plots = enplot.plot(self.map_fkp.submap(box), range=1.)
        # enplot.write('plots/fkp_weight',plots)
        plt.figure(dpi=300)
        plt.title('fkp weighting')
        plt.imshow(self.map_fkp)
        plt.colorbar()
        plt.savefig('plots/imshow_fkp.png')

        self.t_psuedo_map = self.imap_t * self.map_fkp
        self.t_psuedo_alm = map2alm(self.t_psuedo_map, lmax=self.lmax)
        self.cl_tt_psuedo = map2cl(self.t_psuedo_map, self.lmax)

        plt.figure(dpi=300)
        plt.title('fkp-weighted temperature map (uK)')
        plt.imshow(np.ma.masked_where(self.t_psuedo_map == 0., self.t_psuedo_map), interpolation='none')
        plt.colorbar()
        plt.savefig('plots/fkp_weighted_t_map.png')
        plt.close()

        # plt.figure(dpi=300)
        # plt.plot(self.cl_tt_psuedo * self.ells * (1 + self.ells) / 2 / np.pi)
        # plt.savefig('plots/cl_tt_psuedo.png')

    # estimate whether mode-mixing is a serious concern with w_fkp
    def compare_mode_mixing(self, l_cut=3000):
        # construct a "left" power spectrum without noise (l<1500)
        cl_1 = self.cl_cmb.copy()
        cl_1[l_cut + 1:] = 0.

        ls = np.arange(len(self.cl_cmb))

        # construct a "right" power spectrum with noise (l>1500)
        cl_2 = self.cl_cmb
        cl_2[:l_cut + 1] = 0.
        cl_psuedo_1 = self.get_psuedo_cl(cl_1)
        cl_psuedo_2 = self.get_psuedo_cl(cl_2)

        plt.figure(dpi=300)
        plt.title('Mode Mixing Diagnostic')
        # plt.yscale('log')
        plt.ylabel(u'$\log(\Delta_l)$')
        plt.xlabel(u'$l$')
        plt.plot(cl_psuedo_1 * ls * (ls + 1) / 2 / np.pi, label='<l_cut', linewidth=1)
        plt.plot(cl_psuedo_2 * ls * (ls + 1) / 2 / np.pi, label='>l_cut', linewidth=1)
        plt.axvline(l_cut)
        plt.legend()
        plt.savefig('plots/mode_mixing_comparison.png')

    # return a estimator-weighted t-psuedo beam given a realization of unbeamed noise
    # e.g. from CAMB
    # @profile
    def process_t_map(self, t_map, l_weight):
        print("starting process_t_map")
        t_alm = map2alm(t_map * self.map_fkp, lmax=self.lmax)
        weighted_alm = almxfl(t_alm, l_weight)

        t_fkp_est = self.get_zero_map()
        t_fkp_est = alm2map(weighted_alm, t_fkp_est)
        print("done")
        return t_fkp_est

    # @profile
    def process_t_alm(self, t_map, l_weight):
        print("starting process_t_alm")
        t_alm = map2alm(t_map * self.map_fkp, lmax=self.lmax)
        weighted_alm = almxfl(t_alm, l_weight)
        print("done")

        return weighted_alm

    # def compute_a_ksz(self, vr_map, t_psuedo_weighted):
    #     a_ksz = (self.vr_map * t_psuedo_weighted).sum()
    #     return a_ksz

    def compute_a_ksz_alm(self, t_psuedo_alm):
        comp_sum = (np.conjugate(self.vr_alm) * t_psuedo_alm).sum()
        print(comp_sum)
        a_ksz = np.abs(comp_sum)
        return a_ksz

    def get_noise_map(self):
        return enmap.ndmap(np.random.normal(size=self.imap_t.shape) * self.std_t, self.imap_t.wcs)

    # def get_cl_tilde_map(self):
    #     lmax = self.lmax_sim
    #     map1 = rand_map(self.imap_t.shape, self.imap_t.wcs, self.fl[:lmax + 1] * self.cl_cmb[:lmax + 1], lmax=lmax)
    #     map1 += rand_map(self.imap_t.shape, self.imap_t.wcs, self.nl_tilde[:lmax + 1], lmax=lmax)
    #     map1 *= self.map_fkp
    #     return map1

    # return a beamed map from our non-noise simulated cmb power spectrum
    def get_cl_tilde_alm(self):
        ret_alm = rand_alm(self.cl_tt_sim[:self.lmax + 1])
        ret_alm = almxfl(ret_alm, self.beam[:self.lmax + 1, 1])
        return ret_alm

    def get_psuedo_cl(self, cl, beam=False, noise=True):
        if beam:
            bl = self.beam[:self.lmax + 1,1]**2
        else:
            bl = np.ones(self.lmax + 1)
        return cl[:self.lmax + 1] * self.fl[:self.lmax + 1] * bl + noise * self.nl_tilde[:self.lmax + 1]

    def compute_sim_spectra(self, make_plots=False):

        noisemap = self.get_noise_map()
        # self.nl = map2cl(noisemap, lmax=self.lmax_sim)
        self.nl_tilde = map2cl(self.map_fkp * noisemap, lmax=self.lmax_sim)
        beam_sim2 = self.beam[:self.lmax_sim + 1,1]**2

        cl_test = 300. * 2 * np.pi * np.power(1./self.ells_sim, 2)
        cl_test[0] = 0.
        # cl_test = self.cl_cmb.copy()
        t_map = self.get_zero_map()
        t_alm = rand_alm(cl_test)

        # t_alm = almxfl(t_alm, self.beam[:self.lmax_sim + 1,1])
        t_map = alm2map(t_alm, t_map) # look into pixel second arg
        cl_tilde = map2cl(t_map * self.map_fkp, lmax=self.lmax_sim)

        self.fl_actual = (self.cl_tt_psuedo[:self.lmax + 1] - self.nl_tilde[:self.lmax + 1])/self.cl_cmb[:self.lmax + 1]
        self.fl_actual[:50] = 0.

        self.fl = cl_tilde / cl_test
        self.fl[0] = self.fl[2000] # TODO: hack
        # self.fl[:50] = 0.

        self.cl_tt_sim = (self.cl_tt_psuedo[:self.lmax + 1] - self.nl_tilde[:self.lmax + 1]) / self.fl[:self.lmax + 1]
        self.cl_tt_sim[0] = 0.
        # self.cl_tt_sim[:50] = 0.

        self.cl_cmb_tilde = self.fl * self.cl_cmb + self.nl_tilde
        c_norm = self.cl_tt_psuedo[1000] / self.cl_cmb_tilde[1000]

        # check that the generated cl_obs matches the real cl_obs
        cl_xfer = self.get_psuedo_cl(self.cl_tt_sim)
        # cl_xfer = self.cl_tt_sim * self.fl[:self.lmax + 1] + self.nl_tilde[:self.lmax + 1]

        abs_diff = np.abs(cl_xfer[50:] - self.cl_tt_psuedo[50:])
        rel_err = 2 * abs_diff / np.abs(cl_xfer[50:] + self.cl_tt_psuedo[50:])

        plt.figure()
        plt.plot(abs_diff, label='absolute')
        plt.plot(rel_err, label='relative')
        plt.legend()
        plt.yscale('log')
        plt.ylabel('absolute/rel diff')
        plt.xlabel('l - 50')
        plt.savefig('plots/compare_cl_sim_obs.png')
        plt.close()

        assert fequal_either(cl_xfer[50:], self.cl_tt_psuedo[50:], tol=1e-2), f'max diff: {np.max(np.abs(cl_xfer[50:] - self.cl_tt_psuedo[50:]))}'

        if make_plots:
            plt.figure(dpi=300)
            plt.plot(self.fl / beam_sim2, label='computed transfer function')
            plt.plot(self.fl_actual / beam_sim2[:self.lmax + 1], label='actual transfer function')
            plt.legend()
            plt.yscale('log')
            plt.grid(axis='both')
            plt.savefig('plots/cl_transfer.png')

            norm = self.ells * (self.ells + 1) / 2 / np.pi

            plt.figure(dpi=300)
            plt.title('psuedo noise plot')
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

            cl_plot = self.cl_tt_psuedo.copy()
            cl_plot[:500] = 0.
            plt.grid(which='both')
            # plt.plot(self.cl_cmb[:self.lmax + 1] * beam_sim2[:self.lmax + 1], label='cl_sim')
            plt.plot(ell0, self.cl_tt_psuedo[ell0] * norm[ell0],
                     alpha=1., linewidth=1, label='cl_obs_psuedo')
            plt.plot(ell1, cl_xfer[ell1] * norm[ell1],
                     alpha=1., linewidth=1, label='cl_sim_fl')
            # plt.plot((self.cl_tt_psuedo - self.nl_tilde[:self.lmax + 1]) * norm, label='cl_obs_minus_eta2')
            # plt.plot(ells_plot, self.cl_cmb_tilde[ells_plot] * norm[ells_plot], label='cl_psuedo_sim')
            # plt.plot(c_norm * self.cl_cmb_tilde[:self.lmax + 1] * norm, label='cl_psuedo_sim shifted')
            plt.yscale('log')
            plt.legend()
            plt.savefig('plots/cl_psuedo_compare.png')

        # TODO: remove        
        # self.fl = self.fl_actual.copy()

        # cl_zz_obs = map2cl(self.z_theta, self.lmax)


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


def make_pcmb():
    import camb
    from camb import model, initialpower

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(12000, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')


    totCL = powers['total']
    l = np.arange(totCL.shape[0])
    cl_tt = totCL[:,0] * 2 * np.pi / l / (l + 1)
    cl_tt[0] = 0.
    out = np.empty((totCL.shape[0], 2))
    out[:,0] = l
    out[:,1] = cl_tt

    norm = l * (l + 1) / 2 / np.pi

    plt.figure(dpi=300)
    plt.plot(l, cl_tt * norm)
    plt.title('fiducial CMB power')
    plt.xscale('log')
    plt.xlabel('l')
    plt.ylabel(r'$\frac{l(l+1)}{2\pi}C_l$')
    plt.savefig('plots/cmb_power.pdf')

    np.save(cl_cmb_path, out)


def one_time_setup(data_path='/data/', have_camb=False):
    explore_pksz(0.0001)
    if(have_camb):
        make_pcmb()
    reproj_planck_map(map_path, planck_mask_inpath, planck_enmask_path, mode='GAL080')

setup = False


# @profile
# TODO: extend to multi-frequency case!
def compute_estimator(act_pipes, gal_pipe, ntrial=64):

    # set l weight

    # TODO: check cl_zz assumption holds

    # placeholder, in the future we will iterate over frequencies
    pipe = act_pipes[0]
    # process the galaxy radial velocity data into map index space
    gal_pipe.make_vr_list(pipe.imap_t, make_alm=True)

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

    # substitute cl_masked for psuedo cl with fkp
    # cl_use = pipe.get_psuedo_cl(pipe.cl_cmb)

    # WARN: noise-free
    cl_use = pipe.get_psuedo_cl(pipe.cl_tt_sim, beam=True)

    ell_plot = np.arange(500, pipe.lmax + 1,1)
    plt.figure(dpi=300)
    plt.title('l weight components')
    plt.yscale('log')
    plt.plot(ell_plot, pipe.beam[ell_plot,1], label='beam')
    plt.plot(ell_plot, pipe.cl_ksz[ell_plot] / pipe.cl_ksz[3000], label='ksz')
    plt.plot(ell_plot, cl_use[3000] / cl_use[ell_plot], label='cmb')
    plt.xlabel('l')
    plt.ylabel('amplitude (arbitrary)')
    plt.legend()
    plt.savefig('plots/l_weight_compare.png')
    plt.close()

    l_weight = pipe.beam[:pipe.lmax + 1,1] * pipe.cl_ksz[:pipe.lmax + 1] / (cl_use * cl_zz)
    l_weight /= l_weight[pipe.l_ksz]

    l_weight[0] = 0.

    # WARN: l_weight cut
    l_weight[:1500] = 0.

    l_weight_nobeam = pipe.beam[:pipe.lmax + 1,1] * pipe.cl_ksz[:pipe.lmax + 1] / (cl_use * cl_zz)
    l_weight_nobeam /= l_weight[pipe.l_ksz]

    plt.figure(dpi=300)
    plt.plot(l_weight, label='beam')
    plt.plot(l_weight_nobeam, label='no beam')
    plt.xlabel('l')
    plt.ylabel('amplitude (arbitrary)')
    plt.legend()
    plt.savefig('plots/l_weight.png')
    plt.close()


    ones = np.ones(pipe.lmax+1)
    t_fkp_est = pipe.process_t_map(pipe.imap_t, l_weight)
    t_fkp = pipe.process_t_map(pipe.imap_t, ones)

    t_fkp_alm = map2alm(t_fkp, lmax=pipe.lmax)

    a_ksz = gal_pipe.compute_a_ksz(t_fkp_est)

    a_std_sq = np.sqrt(((gal_pipe.get_xz_list(t_fkp_est))**2).sum())
    # gal_mask = pipe.vr_map != 0.
    # assert gal_mask.sum() == pipe.ngal_in, f'{len(pipe.gal_pos)} {gal_mask.sum()} {pipe.ngal_in}'

    assert gal_pipe.gal_inds.shape[1] == gal_pipe.ngal_in

    var_vr = np.var(gal_pipe.vr_list)
    var_t_obs = np.var(t_fkp_est[gal_pipe.gal_inds[0,:], gal_pipe.gal_inds[1,:]])

    var_t_all = np.var(t_fkp_est)

    # WARN: almost exact
    a_std_ind = np.sqrt(gal_pipe.ngal_in * var_vr * var_t_obs)

    print(f'a_tv2: {a_ksz / a_std_sq:.3f}')
    print(f'a_t2v2: {a_ksz / a_std_ind:.3f}')

    ls = np.arange(pipe.lmax + 1)

    print(f'ksz_unnormalized: {a_ksz:.4e}')
    # TODO: check cl tilde edge effects

    # TODO: move xz plot elsewhere!!
    ells = np.arange(pipe.lmax + 1)
    ellnorm = ells * (ells + 1) / 2 /np.pi
    # Compute and plot cl^{xz}

    cl_zz = alm2cl(gal_pipe.vr_alm)
    cl_xx = alm2cl(t_fkp_alm)
    cl_xx[0] = 0.
    cl_xz = alm2cl(gal_pipe.vr_alm, t_fkp_alm)
    cl_xz_std = np.sqrt(cl_xx * cl_zz / (2 * ells + 1))

    plt.figure(dpi=300)
    plt.plot(ells, ellnorm * cl_xx)
    plt.savefig('plots/cl_xx.png')
    plt.close()

    # cl_xz[0] = 0

    bin_width = 500
    cl_xz_bin = cl_xz[:-1].reshape(bin_width, pipe.lmax // bin_width).sum(axis=0)/bin_width
    cl_xz_std_bin = np.sqrt((cl_xz_std[:-1]**2).reshape(bin_width, pipe.lmax // bin_width).sum(axis=0))/bin_width
    bin_centers = np.arange(bin_width//2, pipe.lmax, bin_width)

    plt.figure(dpi=300)
    plt.plot(ells, cl_xz)
    plt.fill_between(ells, cl_xz - cl_xz_std, cl_xz + cl_xz_std, alpha=0.5, color='red')
    plt.savefig('plots/cl_xz.png')
    plt.close()

    plt.figure(dpi=300)
    plt.title(r'$C_l^{XZ}$')
    # plt.errorbar(ells, cl_xz, yerr = cl_xz_std)
    # plt.errorbar(ells, cl_xz, yerr = cl_xz * np.sqrt(2/(2 * ells + 1)))
    plt.plot(bin_centers, cl_xz_bin)
    # plt.plot(ells, cl_xz_std)
    # plt.plot(ells, cl_xz * np.sqrt(2 / (2 * ells + 1)))
    plt.fill_between(bin_centers, cl_xz_bin - cl_xz_std_bin, cl_xz_bin + cl_xz_std_bin, alpha=0.5, color='red')
    plt.axhline(0.)

    # plt.yscale('log')
    # plt.errorbar(ells, cl_xz)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l$ (arbitrary)')
    plt.savefig('plots/cl_xz_binned.png')
    plt.close()
    # for


    a_samples = np.zeros(ntrial)

    # TODO: plot cl_sim vs cl_camb
    # low l disagreement is to be expected
    # high l: should see more power than camb
    # compare fkp weighted psuedo-cl of sim to fkp weighted psuedo cl of data
    for i in range(ntrial):
        print("running variance trial {} of {}".format(i + 1, ntrial))
        t_map = pipe.get_zero_map()
        # random unbeamed map realization with CMB power spectrum
        # t_rand = rand_alm(pipe.cl_obs)
        # t_rand = rand_alm(pipe.cl_cmb)
        t_alm = pipe.get_cl_tilde_alm()

        # apply beam to t_alm (could just modify power spectrum)
        # t_alm = almxfl(t_rand, pipe.beam[:pipe.lmax_sim + 1,1])
        # convert to map

        t_map = alm2map(t_alm, t_map)

        print('Injecting noise in map space')
        t_map += np.random.normal(0, pipe.std_t)

        # make diagnostic plot to compare cl
        cl_psuedo_sim = map2cl(t_map, lmax=pipe.lmax)
        plt.figure(dpi=300)
        plt.yscale('log')
        norm = pipe.ells * (1 + pipe.ells) / 2 / np.pi
        plt.plot(pipe.ells, cl_psuedo_sim * norm)
        plt.plot(pipe.ells, pipe.cl_tt_psuedo[:pipe.lmax + 1] * norm)
        plt.xlabel('l')
        plt.savefig('plots/cl_sim_compare.png')
        plt.close()

        t_psuedo = pipe.process_t_map(t_map, l_weight)
        # eval t_psuedo at galaxy locs
        t_gal_list = gal_pipe.get_t_list(t_psuedo)

        # TODO: update estimator 
        a_std_sq = np.sqrt(((gal_pipe.get_xz_list(t_psuedo))**2).sum())

        # assert gal_mask.sum() == pipe.ngal_in, f'{len(pipe.gal_pos)} {gal_mask.sum()} {pipe.ngal_in}'

        var_t = np.var(t_gal_list)
        var_t_all_sim = np.var(t_psuedo)

        print(f'var(T), var(T_obs) {var_t_obs:.3e} {var_t:.3e}')
        print(f'var(T_all), var(T_all_obs) {var_t_all:.3e} {var_t_all_sim:.3e}')

        a_std_ind = np.sqrt(gal_pipe.ngal_in * var_vr * var_t)

        a_samples[i] = gal_pipe.compute_a_ksz(t_psuedo)

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


# TODO: plot HP filtered map and search for 'striping'
# TODO: vary cl_th
# TODO: kSZ autospectrum is roughly l^2 WARN: wrong
# TODO: cross spectrum is suppressed at small scales
# TODO: estimate cross-power in a few l-bins, instead
# TODO: z-dependent weighting? Divide SDSS into redshift bins and see if alpha changes
if __name__ == "__main__":
    if setup:
        one_time_setup()

    map_freq = ['090', '150']

    # TODO: loop over freqs?

    act_pipe = ActCMBPipe(map_path, ivar_path, beam_path, cl_ksz_path, cl_cmb_path,
                          planck_enmask_path,
                          diag_plots=True, lmax=LMAX)
    
    act_pipe.import_data()
    
    act_pipe.compute_pixel_weight()
    # act_pipe.test_vr_map()
    # act_pipe.compare_power_spectra()

    act_pipe.compute_sim_spectra(make_plots=True)
    
    gal_pipe = GalPipe(sdss_catalog_path, act_pipe)
    gal_pipe.import_data()

    # TODO: redo mode-mixing analysis
    # act_pipe.compare_mode_mixing()
    
    # act_pipe.compare_mask()

    act_pipes = [act_pipe,]

    compute_estimator(act_pipes, gal_pipe, ntrial=3000)
