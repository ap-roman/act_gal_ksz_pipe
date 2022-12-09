import pixell
from pixell import enmap, enplot, utils, curvedsky
from pixell.curvedsky import map2alm, alm2map, rand_alm, alm2cl, almxfl, rand_map

import numpy as np
import time

data_path = '/home/aroman/data/'
act_path = data_path + 'act/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'

map_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'
ivar_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_ivar.fits'
beam_path = act_path + 'beam_f150_daynight.txt'


def masked_inv(ar):
	zeros = ar == 0.

	ret = ar.copy()
	ret[zeros] = 1.
	ret = 1./ret
	ret[zeros] = 0.

	return ret


NTRIAL = 16
LMAX = 12000
R_LWIDTH = 0.62
ells = np.arange(1 + LMAX)
iells = masked_inv(ells)
l_weight = np.exp(-(ells / R_LWIDTH / 5500.)**2)
cl_ref = iells**2


def get_zero_map(ref):
	ret = ref.copy()
	ret[:,:] = 0.
	return ret


if __name__ == "__main__":
	print(f'importing data...')
	map_t = enmap.read_map(map_path)[0]
	dest_map = map_t.copy()
	ivar_t = enmap.read_map(ivar_path)[0]
	var_t = masked_inv(ivar_t)
	std_t = np.sqrt(var_t)

	# perform a sim map benchmark test
	print(f'performing a sim map benchmark...')
	t0 = time.time()
	for itrial in range(NTRIAL):
		print(f'doing {itrial + 1} trial of {NTRIAL}')
		
		ti = time.time()
		dest_map[:,:] = 0.
		ar_alm = rand_alm(cl_ref)
		ar_alm = almxfl(ar_alm, lfilter=l_weight)
		dest_map = alm2map(ar_alm, dest_map)
		dest_map += np.random.normal(size=map_t.shape) * std_t

		dt_single = time.time() - ti
		print(f'this trial took {dt_single:.3e} s')

	dt_cum = time.time() - t0
	time_per = dt_cum / NTRIAL
	print(f'average time per trial: {time_per:.3e} s')