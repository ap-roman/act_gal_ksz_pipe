import pixell
from pixell import enmap, enplot, utils, curvedsky
from pixell.curvedsky import map2alm, alm2map, rand_alm, alm2cl, almxfl
from pixell.reproject import enmap_from_healpix, enmap_from_healpix_interp
import healpy as hp

import astropy
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate


class DirTree:
	def __init__(self, data_path, map_stem):
		self.data_path = data_path
		self.map_stem


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

sdss_catalog_path = data_path + 'sdss_radial_velocities/v1/galaxy_DR10v8_CMASS_South_vrad_using_randoms.fits'

# datpath = '/home/aroman/act_data/act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'


LMAX=12000
PLANCK_NSIDE=2048
eta2_ref = 0.003**2 #uK^2 per steridian


def iround(f_ind):
	sgn = np.sign(f_ind).astype(int)
	ind_abs = (np.abs(f_ind) + 0.5).astype(int)
	return sgn * ind_abs


def fequal(a,b,tol=1e-6):
	return np.all(np.abs(a - b) <= tol)


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
	return alm2cl(map2alm(t_map,lmax=lmax-1))


def ind_round(ix, iy, shape):
	ixr = int(ix + 0.5)
	iyr = int(iy + 0.5)
	ixr = min(max(0, ixr), shape[0] - 1)
	iyr = min(max(0, iyr), shape[1] - 1)
	return ixr, iyr


def bounds_check(ipos, shape):
	return (ipos[0] >= 0) and (ipos[0] < shape[0]) and (ipos[1] >=0) \
		   and (ipos[1] < shape[1])


class ActCMBPipe:
	def __init__(self, map_path, ivar_path, beam_path, fid_ksz_path, fid_cmb_path, planck_mask_path,
				 sdss_catalog_path, fid_cib_path=None, diag_plots=False, l_fkp=1500, l_ksz=3000, lmax=LMAX):
		self.lmax = int(lmax)
		self.l_ksz = l_ksz
		self.l_fkp = l_fkp
		self.diag_plots = diag_plots

		self.map_path = map_path
		self.ivar_path = ivar_path
		self.beam_path = beam_path
		self.planck_mask_path = planck_mask_path
		self.sdss_catalog_path = sdss_catalog_path

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
		print("done")

		print("importing mask: {}".format(self.planck_mask_path))
		self.mask_t = enmap.ndmap(np.load(self.planck_mask_path) - 254., self.imap_t.wcs)
		print("done")

		print("importing inverse variance mask: {}".format(self.ivar_path))
		self.ivar = enmap.read_map(self.ivar_path)
		self.ivar_t = self.ivar[0]
		print("done")

		print("importing galaxy catalog")
		with fits.open(self.sdss_catalog_path) as hdulist:
			table = hdulist[1].data
			self.ngal2d = len(table)
			self.gal_table = table.copy()
		print("done")

		# TODO: dangerous untracked unit conversion
		self.adelt = self.imap_t.wcs.wcs.cdelt * np.pi / 180.
		self.cdelt = self.imap_t.wcs.wcs.cdelt

		assert fequal(np.abs(self.adelt[0]), np.abs(self.adelt[1]))

		self.angular_res = self.adelt[0]
		self.ivar_weight = self.angular_res**2

		assert self.ivar.shape == self.imap.shape

		self.cl_nomask = map2cl(self.imap_t, self.lmax)
		self.cl_mask = map2cl(self.imap_t * (1 - self.mask_t), self.lmax)

		self.beam = parse_beam(self.beam_path)

		if self.diag_plots:
			plt.figure(dpi=300)
			plt.plot(self.beam[:,0], self.beam[:,1])
			plt.xlabel('l')
			plt.ylabel('amplitude')
			plt.savefig('plots/beam.png')
			plt.close()

		# TODO fiducial CIB power spectrum import

		cl_ksz = np.load(self.fid_ksz_path)
		l = np.arange(self.lmax)
		print('ksz lmax:',len(l))
		# self.cl_ksz = cl_ksz.copy()
		# self.cl_ksz[1:,1] =  self.cl_ksz[1:,1] * 2 * np.pi / (l[1:] * (l[1:] + 1))
		# self.cl_ksz[0,1] = 0.
		# self.cl_ksz_interp = InterpWrapper(cl_ksz[:,0], cl_ksz[:,1])
		self.cl_ksz = 1./l**2
		self.cl_ksz[0] = 1.

		cl_cmb = np.load(self.fid_cmb_path)[:self.lmax]
		l = np.arange(len(cl_cmb))
		print('cmb lmax:',len(l))
		self.cl_cmb = cl_cmb.copy()
		self.cl_cmb[1:,1] =  self.cl_cmb[1:,1] * 2 * np.pi / (l[1:] * (l[1:] + 1))
		self.cl_cmb[0,1] = 0.

	def get_zero_map(self):
		return enmap.ndmap(np.zeros(self.imap_t.shape), self.imap_t.wcs)
	
	def init_sim_map(self):
		self.sim_map = self.get_zero_map()

	# simulate Tcmb in harmonic space
	def add_sims(self):
		print("adding simulated sky")
		if not self.sim_map:
			self.init_sim_map()

		sim_alm = rand_alm(self.cl_ksz[:,1])
		sim_alm += rand_alm(self.cl_cmb[:,1])

		# add to internal map
		sim_map = alm2map(sim_alm, self.sim_map)
		# plt.figure(dpi=300)
		# plt.imshow(self.sim_map)
		# plt.save('plots/sim_map_test.pdf')

	# plot a comparison cl between the masked + weighted vs raw data
	def compare_mask(self):

		# nz_mask = (self.ivar_t*self.ivar_weight > 0.) * (self.ivar_t*self.ivar_weight != np.inf)

		# eta2_data = (1./(self.ivar_t[nz_mask] * self.ivar_weight))
		# print(eta2_data.min(), eta2_data.max())
		# eta2_data = eta2_data[eta2_data < 1e20]
		# print(eta2_data.min(), eta2_data.max())

		# eta2_approx = eta2_data.mean()
		# print(eta2_approx, eta2_ref)
		print('mask fraction: {}'.format(self.mask_t.sum()/np.prod(self.mask_t.shape)))

		# cl_nomask = map2cl(self.imap_t, self.lmax)
		# cl_mask = map2cl(self.imap_t * (1 - self.mask_t), self.lmax)
		# cl_weight = map2cl(self.imap_t * self.map_fkp, self.lmax)
		# print(cl_weight)

		ls = np.arange(self.lmax)

		plt.figure(dpi=300)

		plt.title('Comparison of Processed Power Spectra')
		plt.yscale('log')
		plt.ylabel(u'$\log(\Delta_l)$')
		plt.xlabel(u'$l$')
		# plt.plot(self.cl_nomask * ls * (ls + 1) / 2 / np.pi, label='unmasked')
		# plt.plot(self.cl_mask * ls * (ls + 1) / 2 / np.pi, label='foreground masked')
		plt.plot(self.cl_tt_psuedo * ls * (ls + 1) / 2 / np.pi, label='fkp weighted')

		plt.legend()
		plt.savefig('plots/mask_cl_comparison.png')

	# TODO: add foreground map
	def compute_pixel_weight(self, l0=None):
		if l0 is None:
			l0 = self.l_fkp
			print("No FKP l scale provided, using l_fkp={}".format(self.l_fkp))
		b2_l0 = self.beam[l0,1]**2
		ctt_l0 = self.cl_cmb[l0,1]

		eta_n2 = self.ivar_t * self.ivar_weight
		# print(eta_n2)
		# eta_n2 = np.maximum(eta_n2, inf_mask(self.mask_t.astype(bool)))
		eta_n2 *= (1 - self.mask_t)

		self.eta_n2 = eta_n2

		# compute FKP pixel weight
		self.w_fkp_theta = eta_n2 / (1 + b2_l0 * ctt_l0 * eta_n2)
		self.map_fkp = enmap.ndmap(self.w_fkp_theta, self.imap_t.wcs)
		self.t_psuedo_map = self.imap_t * self.map_fkp
		self.t_psuedo_alm = map2alm(self.t_psuedo_map, lmax=self.lmax)
		self.cl_tt_psuedo = map2cl(self.t_psuedo_map, self.lmax)

	def get_psuedo_cl(self, cl_ref):
		# TODO: add beam to alms
		# TODO: add ivar noise instead of eta^2 term
		# TODO: long term, seek weighting that minimizes mode-mixing
		t_alm = curvedsky.rand_alm(cl_ref)
		t_map = self.get_zero_map()
		t_map = curvedsky.alm2map(t_alm, t_map)
		t_map *= self.map_fkp
		t_tilde_alm = t_alm.copy()
		t_tilde_alm = curvedsky.map2alm(t_map, t_tilde_alm)
		cl_psuedo = curvedsky.alm2cl(t_tilde_alm)
		return cl_psuedo

	# estimate whether mode-mixing is a serious concern with w_fkp
	def compare_mode_mixing(self, l_cut=1500):
		# construct a "left" power spectrum without noise (l<1500)
		cl_1 = self.cl_cmb[:,1].copy()
		cl_1[l_cut:] = 0.

		ls = np.arange(len(self.cl_cmb))

		# construct a "right" power spectrum with noise (l>1500)
		cl_2 = self.cl_cmb[:,1].copy() + eta2_ref
		cl_2[:l_cut] = 0.
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
	def process_t_map(self, t_map, l_weight):
		t_fkp = t_map * self.map_fkp
		t_alm = map2alm(t_fkp, lmax=self.lmax)

		t_fkp_est = self.get_zero_map()
		t_fkp_est = alm2map(almxfl(self.t_psuedo_alm, l_weight), t_fkp_est)
		return t_fkp_est

	def compute_est_kernel(self, v_r, gal_pos, t_psuedo):
		a_ksz = 0.

		n_ob = 0
		for v_r, pos in zip(v_r, gal_pos):
			dec, ra = pos
			idec, ira = iround(t_psuedo.sky2pix((dec,ra)))

			if not bounds_check((idec,ira), t_psuedo.shape):
				n_ob += 1
			else:
				a_ksz += v_r * t_psuedo[idec, ira]

		print(float(n_ob + 1) / len(gal_pos))
		return a_ksz

	def compute_estimator(self, ntrial=64):
		self.v_rad = self.gal_table['VR']
		self.gal_pos = np.zeros((self.ngal2d, 2))
		# TODO: confirm correct position units for sky2pix!!
		self.gal_pos[:,0] = self.gal_table['DEC_DEG'] * (np.pi / 180)
		self.gal_pos[:,1] = self.gal_table['RA_DEG'] * (np.pi / 180)
		# self.gal_pos[:,0] = self.gal_table['DEC_DEG']
		# self.gal_pos[:,1] = self.gal_table['RA_DEG']
		# TODO: check cl_zz assumption holds
		self.cl_zz = np.ones(self.lmax)
		self.cl_zz[:self.l_ksz] = 0.
		self.cl_zz *= np.var(self.v_rad) / self.ngal2d

		self.cl_obs = self.cl_mask
		# self.z_theta = self.get_zero_map()


		# optimal weighting for the alpha estimator
		l_weight = self.beam[:self.lmax,1] * self.cl_ksz / (self.cl_obs * self.cl_zz)
		l_weight[0] = 0.

		t_fkp_est = self.process_t_map(self.imap_t, l_weight)

		a_ksz = self.compute_est_kernel(self.v_rad, self.gal_pos, t_fkp_est)
		a_samples = np.zeros(ntrial)

		for i in range(ntrial):
			print("running variance trial {} of {}".format(i + 1, ntrial))
			t_obs = self.get_zero_map()
			# random beamed map realization with CMB power spectrum
			t_rand = rand_alm(self.cl_cmb)
			t_alm = almxfl(t_rand, self.beam[:,1])
			t_obs = alm2map(t_alm, t_obs)
			t_psuedo = self.process_t_map(t_obs, l_weight)

			a_samples[i] = self.compute_est_kernel(self.v_rad, self.gal_pos, t_psuedo)
		
		a_sigma = np.sqrt(np.var(a_samples))
		a_mean = np.mean(a_samples)
		print("a_ksz, a_sigma, <a>, sigma: {:.2e}, {:.2e}, {:.2e}, {:.2e}".format(a_ksz, 
														a_sigma, a_mean, a_ksz/a_sigma))

			# TODO: ask slack how to map to pixel space more accurately

			# idec, ira = self.imap_t.wcs.world_to_pixel_values(dec, ra)
			# idec, ira = self.imap_t.wcs.world_to_pixel_values(dec, ra)
			# idec, ira = ind_round(idec, ira, self.imap_t.shape)

			# print(idec, ira)
			# self.z_theta[idec, ira] = v_r
			# TODO: unsafe origin assumption in pixel space

		# cl_zz_obs = map2cl(self.z_theta, self.lmax)


	# def add_planck_mask(self, planck_mask_path, mode='GAL099', nside=2048):
	# 	self.planck_mask_path = planck_mask_path

	# 	if planck_mask_path:
	# 		this_mask = np.invert(fits.getdata(planck_mask_path, ext=0)[mode])
	# 		# enmap_mask = enmap_from_healpix(this_mask, self.mask_t.shape,
	# 		# 							    self.mask_t.wcs, lmax=self.lmax)
	# 		enmap_mask = enmap_from_healpix_interp(this_mask, self.mask_t.shape,
	# 									    self.mask_t.wcs, interpolate=True)
	# 		print(enmap_mask.shape)
	# 		print(self.mask_t.shape)
	# 		self.mask_t = np.minimum(enmap_mask, self.mask_t)

	# 		plt.figure(dpi=300)
	# 		plt.imshow(self.mask_t)
	# 		plt.savefig('plots/mask_test.pdf')


def reproj_planck_map(map_inpath, mask_inpath, outpath, mode='GAL099', nside=PLANCK_NSIDE):
	imap = enmap.read_map(map_inpath)
	imap_t = imap[0]

	this_mask = np.invert(fits.getdata(mask_inpath, ext=0)[mode])
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
	cl_tt = totCL[:,0]
	out = np.empty((totCL.shape[0], 2))
	out[:,0] = l
	out[:,1] = cl_tt

	plt.figure(dpi=300)
	plt.plot(l, cl_tt)
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
	reproj_planck_map(map_path, planck_mask_inpath, planck_enmask_path, mode='GAL090')


# if __name__ == "__main__":
# 	imap = enmap.read_map(datpath)
# 	imap_t = imap[0] # select the t-channel... Note: be careful as indices aren't labelled

# 	# print(imap_t.box())

# 	# bbox = np.sort(np.mod(imap_t.box(), 2*np.pi), axis=1)
# 	bbox = imap_t.box()
# 	# print(bbox)
# 	oset = bbox[:,0]
# 	sbox = 0.125 * (bbox - oset[:,None]) + oset[:,None]
# 	# print(sbox)
# 	stamp_t = imap_t.submap(sbox)
# 	# print('bounding box:', 0.25 * (bbox - oset) + oset)
# 	# plt.figure(dpi=300)
# 	# plt.imshow(stamp_t, vmin=-300, vmax=300)
# 	# plt.savefig('plots/stamp_t.pdf')
# 	# plt.close()

# 	plots = enplot.plot(stamp_t, range=300, mask=0)
# 	# enplot.show(plots)
# 	enplot.write('plots/map_t', plots)


setup = False

# TODO: plot HP filtered map and search for 'striping'
# TODO: vary cl_th
# TODO: kSZ autospectrum is roughly l^2
# TODO: cross spectrum is suppressed at small scales
# TODO: estimate cross-power in a few l-bins, instead
# TODO: z-dependent weighting? Divide SDSS into redshift bins and see if alpha changes
if __name__ == "__main__":
	if setup:
		one_time_setup()

	act_pipe = ActCMBPipe(map_path, ivar_path, beam_path, cl_ksz_path, cl_cmb_path,
						  planck_enmask_path, sdss_catalog_path,
						  diag_plots=True, lmax=12000)
	
	act_pipe.import_data()
	# act_pipe.init_sim_map()
	# act_pipe.add_sims()
	act_pipe.compute_pixel_weight()
	# act_pipe.compare_mode_mixing()
	# act_pipe.compare_mask()
	act_pipe.compute_estimator(ntrial=64)
