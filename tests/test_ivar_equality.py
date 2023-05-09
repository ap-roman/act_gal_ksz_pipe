from fnl_pipe.pipe import PipePath
import pixell
from pixell import enmap, enplot, utils, curvedsky
import numpy as np
import os

base = '/home/aroman/data/act/'
base_c = '/home/aroman/data/act_pub/'

ivar_a_path = base + 'act_planck_s08_s19_cmb_f090_daynight_srcfree_ivar.fits'
ivar_b_path = base + 'act_planck_s08_s19_cmb_f090_daynight_ivar.fits'
ivar_c_path = base_c + 'act_planck_dr5.01_s08s18_AA_f090_daynight_ivar.fits'


def import_ivar(path):
	fname = path.split(os.sep)[-1]
	print(f'importing {fname}')

	ivar_t = enmap.read_map(path)[0]

	ivar_min = ivar_t.min()
	ivar_max = ivar_t.max()

	print(f'ivar min/max: {ivar_min:.3e}, {ivar_max:.3e}')

	return ivar_t


def ivar_diff(ivar_a, ivar_b, eps=1e-8):
	diff = np.abs(ivar_a - ivar_b)
	# amp = 0.5 * (np.abs(ivar_a) + np.abs(ivar_b))

	# med_amp = np.median(amp)

	valid = diff > eps

	n_invalid = valid.sum()

	print(f'number of diagreements: {n_invalid}')

	return n_invalid == 0


if __name__ == "__main__":
	print("Testing ivar agreement")

	ivar_a = import_ivar(ivar_a_path)
	ivar_b = import_ivar(ivar_b_path)
	# ivar_c = import_ivar(ivar_c_path)

	assert(ivar_diff(ivar_a, ivar_b))
	print("PASSED")