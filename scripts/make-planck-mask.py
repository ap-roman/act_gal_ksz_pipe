from pixell import enmap
from pixell.reproject import enmap_from_healpix, enmap_from_healpix_interp, healpix_from_enmap
import numpy as np
import astropy
from astropy.io import fits
import healpy as hp

planck_path = '/home/aroman/data/planck/'
act_path = '/home/aroman/data/act_pub/'

map_path = act_path + 'act_dr5.01_s08s18_AA_f090_daynight_map_srcfree.fits'
planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_outpath_base = '/home/aroman/data/mask/planck_foregound_mask_'

PLANCK_NSIDE=2048

modes = ['GAL040',
         'GAL060',]


def reproj_planck_map(map_inpath, mask_inpath, outpath, mode, nside=PLANCK_NSIDE):
    imap = enmap.read_map(map_inpath)
    imap_t = imap[0]

    this_mask = np.array(fits.getdata(mask_inpath, ext=1)[mode]).astype(bool)
    this_mask = hp.pixelfunc.reorder(this_mask, n2r=True)
    enmap_mask = enmap_from_healpix_interp(this_mask, imap_t.shape,
                            imap_t.wcs, interpolate=True)
    np.save(outpath, enmap_mask)


if __name__ == "__main__":
    # config_file = sys.argv[1]
    # printlog('got config file ' + config_file)
    # config_dict = get_yaml_dict(config_file)
    # local_dict = locals()
    # printlog('dumping config')
    # for key, value in config_dict.items():
    #     printlog(key, value)
    #     local_dict[key] = value
    # printlog('################## DONE ##################')

    for mode in modes:
        print(f'reprojecting {planck_mask_inpath} with mode {mode}')
        reproj_planck_map(map_path, planck_mask_inpath,
                          planck_outpath_base + mode + '.npy', mode=mode)