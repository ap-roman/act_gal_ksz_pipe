import pymangle
from pixell import enmap, curvedsky
import numpy as np
import matplotlib.pyplot as plt

data_path = '/home/aroman/data/'

act_path = data_path + 'act/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'

map_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'
ivar_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_ivar.fits'
beam_path = act_path + 'beam_f150_daynight.txt'

mangle_base = '/data/sdss/DR12v5/masks/'
mangle_path = mangle_base + 'mask_DR12v5_CMASS_North.ply'

mangle_out = '/home/aroman/data/mangle/sdss_dr12v5_mangle.fits'

if __name__=="__main__":
	mangle = pymangle.Mangle(mangle_path)
	ref_map = enmap.read_map(map_path)[0]
	posmap = ref_map.posmap()

	print("checking that the mangle map contains the ACT field...")
	decs, ras = posmap * 180. / np.pi
	ras += 180. # correct for ACT offset

	plt.figure(dpi=300)
	plt.title('decs at constant ra (deg)')
	plt.plot(decs[:,0])
	plt.savefig('plots/act_decs.png')

	plt.figure(dpi=300)
	plt.title('ras at constant dec (deg)')
	plt.plot(ras[0,:])
	plt.savefig('plots/act_ras.png')

	plt.figure(dpi=300)
	plt.imshow(decs)
	plt.colorbar()
	plt.savefig('plots/mangle_decs.png')

	plt.figure(dpi=300)
	plt.imshow(ras)
	plt.colorbar()
	plt.savefig('plots/mangle_ras.png')

	full_ras, full_decs = np.meshgrid(np.linspace(0,360,128), np.linspace(-90,90,128), indexing='ij')
	# assert(np.all(mangle.contains(ras, decs)))
	print(mangle.contains(ras[0,0], decs[0,0]))
	print(mangle.contains(ras[0,-1], decs[0,-1]))
	print(mangle.contains(ras[-1,0], decs[-1,0]))
	print(mangle.contains(ras[-1,-1], decs[-1,-1]))
	print("done")

	print("computing mangle weights")
	pixell_mangle = enmap.ndmap(np.empty(ref_map.shape), ref_map.wcs)
	weights = mangle.weight(ras.flatten(), decs.flatten()).reshape(ref_map.shape)
	print("done")

	plt.figure(dpi=300)
	plt.imshow(weights[::-1,::-1])
	plt.colorbar()
	plt.savefig('plots/mangle_weights.png')

	full_sky_mangle = mangle.weight(full_ras, full_decs).reshape(128,128)
	plt.figure(dpi=300)
	plt.imshow(full_sky_mangle[::-1,:])
	plt.colorbar()
	plt.savefig('plots/mangle_weights_full.png')

	print("copying weights to pixell mangle map")
	pixell_mangle[:,:] = weights
	print("done")

	print(f"writing pixell mangle file: {mangle_out}")
	pixell_mangle.write(mangle_out)
	print("done")

