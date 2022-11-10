import numpy as np

import time

from fnl_pipe.pipe import ActPipe, GalPipe
from fnl_pipe.util import OutputManager


data_path = '/home/aroman/data/'
# act_path = data_path + 'act/'
act_path = data_path + 'act_pub/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'
gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'

# map_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'
# ivar_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_ivar.fits'
# map_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_map.fits'
map_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_map_srcfree.fits'
ivar_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_ivar.fits'
beam_path = act_path + 'act_planck_dr5.01_s08s18_f150_daynight_beam.txt'

cl_cmb_path = data_path + 'spectra/cl_cmb.npy'
cl_ksz_path = data_path + 'spectra/cl_ksz.npy'

planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_enmask_path = mask_path + 'planck_foreground.npy'

catalog_path = data_path + 'vr_summaries/v01_sdss_cmass_north.h5'

kszpipe_path = data_path + 'kszpipe/sdss_dr12/cmass_north/'
kszpipe_cosmo_path = kszpipe_path + 'cosmology.pkl'
kszpipe_box_path = kszpipe_path + 'bounding_box.pkl'
kszpipe_d0_path = kszpipe_path + 'delta0_DR12v5_CMASS_North.h5'

# Parameters chosen ahead of time to maximize snr with this particular dataset
R_FKP = 1.56
R_LWIDTH = 0.62
# NTRIAL_FL = 512
NTRIAL_FL = 256
# NTRIAL_NL = 128
NTRIAL_NL = 64
NAVE_FL = 4

if __name__ == "__main__":
    om = OutputManager(base_path='output', title='make_xfer')

    act_pipe = ActPipe(map_path, ivar_path, beam_path, cl_ksz_path, cl_cmb_path,    
                          planck_enmask_path, om,
                          custom_l_weight=None, diag_plots=True, lmax=12000,
                          gal_mask_path=gal_mask_path)

    act_pipe.import_data()
    act_pipe.update_metadata(r_fkp=R_FKP, r_lwidth=R_LWIDTH, gal_path=catalog_path)
    act_pipe.compute_pixel_weight()
    act_pipe.compute_fl_nl(ntrial_nl=NTRIAL_NL, ntrial_fl=NTRIAL_FL, nave_fl=NAVE_FL, 
                                 fl_path='transfer_function_pub.h5')