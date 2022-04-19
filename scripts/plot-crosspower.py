from fnl_pipe.realspace import Padded3DPipe, PipeAdjoint
from fnl_pipe.pipe import ActPipe, GalPipe, compute_estimator

import kszpipe
from kszpipe.Cosmology import Cosmology 
from kszpipe.Box import Box

import numpy as np

import time

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
NTRIAL = 2048


def rel_error_inv_std(n):
    err_prop = np.sqrt(2/n)
    return np.array([1./(1 + err_prop), 1./(1 - err_prop)])


if __name__ == "__main__":
    act_pipe = ActPipe(map_path, ivar_path, beam_path, cl_ksz_path, cl_cmb_path,    
                          planck_enmask_path,
                          custom_l_weight=None, diag_plots=True, lmax=12000)

    act_pipe.import_data()
    act_pipe.update_metadata(r_fkp=R_FKP, r_lwidth=R_LWIDTH, gal_path=catalog_path)
    act_pipe.import_fl_nl(fl_path)
    act_pipe.compute_pixel_weight()
    act_pipe.compute_sim_spectra(make_plots=True)
    act_pipe.compute_l_weight()

    gal_pipe = GalPipe(catalog_path, act_pipe, diag_plots=True)
    gal_pipe.import_data()
    gal_pipe.make_vr_list()

    cosmology = kszpipe.io_utils.read_pickle(kszpipe_cosmo_path)
    box = kszpipe.io_utils.read_pickle(kszpipe_box_path)

    real_pipe = Padded3DPipe(gal_pipe, cosmology)

    real_pipe.init_from_box(box)

    pa = PipeAdjoint(real_pipe) # WARN: superfluous adjoint structure... should get integrated into Padded3DPipe
    pa.box = box

    t_act_hp_list = gal_pipe.get_t_list(act_pipe.get_t_pseudo_hp())

    real_pipe.add_galaxies(t_list=t_act_hp_list)
    # real_pipe.plot_3d(real_pipe.ngal, 'plots/ngal_', var_label=r'$n_{gal}$')

    # This is a debug plot! plot comoving distance from origin
    # real_pipe.plot_3d(real_pipe.chi_grid, 'plots/chi_grid_', mode='slice',
    #                   slice_inds=real_pipe.ind0, var_label=r'$\chi$')

    d0_k = box.read_grid(kszpipe_d0_path, fourier=True)
    real_pipe.init_d0_k(d0_k)

    alpha_2d, a_std = compute_estimator([act_pipe,], gal_pipe, r_lwidth=R_LWIDTH)
    # a_std = 1.
    # alpha_2d = np.nan

    print('=============================================')
    print('2D pipeline results:')
    print(f'a_ksz: {alpha_2d:3e}, a_ksz_nonnorm: {alpha_2d * a_std:3e}')
    print('=============================================')

    # for i in range(1):
    real_pipe.add_galaxies(t_list=t_act_hp_list, wipe=True)
    alpha_3d, ignore = real_pipe.do_harmonic_sum_adj(t_act_hp_list, pa)

    print('=============================================')
    print('3D pipeline results:')
    print(f'a_ksz: {alpha_3d / a_std:3e}, a_ksz_nonnorm: {alpha_3d:3e}')
    print(f'2D/3D ratio: {alpha_2d/alpha_3d:3e}')
    print('=============================================')


    # t_sim_map = act_pipe.get_sim_map()
    # t_sim_list = gal_pipe.get_t_list(t_sim_map)
    # res_2d = np.empty((NTRIAL,2))
    res_3d = np.zeros(NTRIAL)
    t0 = time.time()

    res_2d = np.empty(NTRIAL)
    # just resample if NTRIAL > available sims?
    assert gal_pipe.temp_sims is not None
    assert NTRIAL <= gal_pipe.nsims
    for itrial in range(NTRIAL):
        # t_sim_map = act_pipe.get_sim_map()
        t_sim_list = gal_pipe.temp_sims[itrial]
        # a_ksz_2d = gal_pipe.compute_a_ksz(t_sim_map)
        a_ksz_2d = (gal_pipe.vr_list * t_sim_list).sum()
        # real_pipe.add_galaxies(t_list=t_sim_list, wipe=True)
        a_ksz_3d, ignore = real_pipe.do_harmonic_sum_adj(t_sim_list, pa)
        res_3d[itrial] = a_ksz_3d
        res_2d[itrial] = a_ksz_2d

        tnow = time.time()
        time_per_iter = (tnow - t0) / (itrial + 1)

        print('=============================================')
        print(f'time_per_iter: {time_per_iter:.3e}')
        print('trial results:')
        print(f'a_ksz_3d: {alpha_3d / res_3d[:itrial+1].std():3e}')
        print(f'a_ksz_2d: {a_ksz_2d / a_std:3e} a_ksz_3d: {a_ksz_3d / a_std:3e} 2d/3d: {a_ksz_2d / a_ksz_3d:3e}')
        print('=============================================')


    a3d_stdev = res_3d.std()
    a3d_norm_est = alpha_3d / a3d_stdev
    a3d_lo, a3d_hi = a3d_norm_est * rel_error_inv_std(NTRIAL)

    a2d_stdev = res_2d.std()
    a2d_result = alpha_2d / a2d_stdev
    a2d_lo, a2d_hi = a2d_result * rel_error_inv_std(NTRIAL)
    print(f'MC normalized alpha_2d: {a2d_result:.2f} {a2d_lo:.2f} {a2d_hi:.2f}')
    print(f'MC normalized alpha_3d: {a3d_norm_est:.2f} (lo/hi) {a3d_lo:.2f} {a3d_hi:.2f}')