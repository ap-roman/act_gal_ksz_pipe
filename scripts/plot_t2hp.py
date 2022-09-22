import matplotlib.pyplot as plt
import numpy as np
from pixell import enmap
from pixell.curvedsky import map2alm, alm2cl

data_path = '/home/aroman/data/'
act_path = data_path + 'act/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'
pipe_path = data_path + 'pipe/'

t2hp_path = act_path + f'meta/t2_2048.fits'


LMAX=12000

def map2cl(mymap, lmax=LMAX):
	alm = map2alm(mymap, lmax=lmax)
	cl = alm2cl(alm)
	return cl

if __name__ == "__main__":
	t2_map = enmap.read_map(t2hp_path)


	plt.figure(dpi=300)
	plt.title(r'$T_{hp}^2$')
	plt.imshow(t2_map)
	plt.colorbar()
	plt.savefig('plots/t2hp.png')

	ells = np.arange(LMAX + 1)
	cl_norm = ells *(ells + 1) / 2 / np.pi
	cl = map2cl(t2_map)
	plt.figure(dpi=300)
	plt.title('t2_cl')
	plt.plot(ells, cl_norm * cl)
	plt.xlabel('ell')
	plt.ylabel('Delta_l')
	plt.savefig('plots/t2_cl.png')