import numpy as np

import time

from fnl_pipe.pipe import ActPipe, GalPipe


data_path = '/home/aroman/data/'
act_path = data_path + 'act/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'
pipe_path = data_path + 'pipe/'

map_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'
ivar_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_ivar.fits'
beam_path = act_path + 'beam_f150_daynight.txt'

cl_cmb_path = data_path + 'spectra/cl_cmb.npy'
cl_ksz_path = data_path + 'spectra/cl_ksz.npy'

planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_enmask_path = mask_path + 'planck_foreground.npy'

catalog_path = data_path + 'vr_summaries/v01_sdss_cmass_north.h5'

kszpipe_path = data_path + 'kszpipe/sdss_dr12/cmass_north/'
kszpipe_cosmo_path = kszpipe_path + 'cosmology.pkl'
kszpipe_box_path = kszpipe_path + 'bounding_box.pkl'
kszpipe_d0_path = kszpipe_path + 'delta0_DR12v5_CMASS_North.h5'

fl_path = pipe_path + 'transfer_function.h5'

# Parameters chosen ahead of time to maximize snr with this particular dataset
R_FKP = 1.56
R_LWIDTH = 0.62
# NSIMS = 2046

NSIMS = 2048

if __name__ == "__main__":
    act_pipe = ActPipe(map_path, ivar_path, beam_path, cl_ksz_path, cl_cmb_path,    
                          planck_enmask_path,
                          custom_l_weight=None, diag_plots=True, lmax=12000)

    act_pipe.import_data()
    act_pipe.update_metadata(r_fkp=R_FKP, r_lwidth=R_LWIDTH, gal_path=catalog_path)
    act_pipe.compute_pixel_weight()
    act_pipe.import_fl_nl(fl_path)
    act_pipe.compute_sim_spectra(make_plots=True)
    act_pipe.compute_l_weight()

    gal_pipe = GalPipe(catalog_path, act_pipe, diag_plots=True)
    gal_pipe.import_data()
    gal_pipe.make_vr_list()

    gal_pipe.add_sims(act_pipe, NSIMS)