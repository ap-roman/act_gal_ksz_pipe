from fnl_pipe.realspace import Padded3DPipe
from fnl_pipe.pipe import ActPipe, GalPipe, compute_estimator
import kszpipe
from kszpipe.Cosmology import Cosmology 
from kszpipe.Box import Box

import numpy as np

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

catalog_path = data_path + 'vr_summaries/v01_sdss_cmass_north.h5'

kszpipe_path = data_path + 'kszpipe/sdss_dr12/cmass_north/'
kszpipe_cosmo_path = kszpipe_path + 'cosmology.pkl'
kszpipe_box_path = kszpipe_path + 'bounding_box.pkl'
kszpipe_d0_path = kszpipe_path + 'delta0_DR12v5_CMASS_North.h5'

# Parameters chosen ahead of time to maximize snr with this particular dataset
R_FKP = 1.56
R_LWIDTH = 0.62

if __name__ == "__main__":
    act_pipe = ActPipe(map_path, ivar_path, beam_path, cl_ksz_path, cl_cmb_path,    
                          planck_enmask_path,
                          custom_l_weight=None, diag_plots=True, lmax=12000)

    act_pipe.import_data()
    act_pipe.compute_pixel_weight(r_fkp=R_FKP)
    act_pipe.compute_sim_spectra(ntrial_fl=1, make_plots=True)
    act_pipe.compute_l_weight(r_lwidth=R_LWIDTH)

    gal_pipe = GalPipe(catalog_path, act_pipe, diag_plots=True)
    gal_pipe.import_data()
    gal_pipe.make_vr_list()

    cosmology = kszpipe.io_utils.read_pickle(kszpipe_cosmo_path)
    box = kszpipe.io_utils.read_pickle(kszpipe_box_path)

    real_pipe = Padded3DPipe(gal_pipe, cosmology)

    real_pipe.init_from_box(box)

    t_act_hp_list = gal_pipe.get_t_list(act_pipe.get_t_pseudo_hp())

    real_pipe.add_galaxies(t_list=t_act_hp_list)
    real_pipe.plot_3d(real_pipe.ngal, 'plots/ngal_', var_label=r'$n_{gal}$')

    # This is a debug plot! plot comoving distance from origin
    real_pipe.plot_3d(real_pipe.chi_grid, 'plots/chi_grid_', mode='slice',
                      slice_inds=real_pipe.ind0, var_label=r'$\chi$')

    d0_k = box.read_grid(kszpipe_d0_path, fourier=True)
    real_pipe.init_d0_k(d0_k)

    alpha_2d, a_std = compute_estimator([act_pipe,], gal_pipe, r_lwidth=R_LWIDTH)

    print('=============================================')
    print('2D pipeline results:')
    print(f'a_ksz: {alpha_2d:3e}, a_ksz_nonnorm: {alpha_2d * a_std:3e}')
    print('=============================================')

    alpha_3d = real_pipe.do_harmonic_sum()

    print('=============================================')
    print('3D pipeline results:')
    print(f'a_ksz: {alpha_3d / a_std:3e}, a_ksz_nonnorm: {alpha_3d:3e}')
    print('=============================================')