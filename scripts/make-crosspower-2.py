from fnl_pipe.realspace import Padded3DPipe, PipeAdjoint
from fnl_pipe.pipe import ActPipe, GalPipe, compute_estimator
from fnl_pipe.util import OutputManager

import kszpipe
from kszpipe.Cosmology import Cosmology 
from kszpipe.Box import Box

import numpy as np
import matplotlib.pyplot as plt

import time

data_path = '/home/aroman/data/'
# act_path = data_path + 'act/'
act_path = data_path + 'act_pub/'
planck_path = data_path + 'planck/'
mask_path = data_path + 'mask/'
pipe_path = data_path + 'pipe/'

# map_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits' # private
map_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_map_srcfree.fits' # public
# ivar_path = act_path + 'act_planck_s08_s19_cmb_f150_daynight_srcfree_ivar.fits' # private
ivar_path = act_path + 'act_planck_dr5.01_s08s18_AA_f150_daynight_ivar.fits' # public
# beam_path = act_path + 'beam_f150_daynight.txt' # proprietary beam file
beam_path = act_path + 'act_planck_dr5.01_s08s18_f150_daynight_beam.txt' # public beam

gal_mask_path = data_path + 'sdss_footprint/pixellized_sdss_north_completeness.fits'

cl_cmb_path = data_path + 'spectra/cl_cmb.npy'
cl_ksz_path = data_path + 'spectra/cl_ksz.npy'

planck_mask_inpath = planck_path + 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
planck_enmask_path = mask_path + 'planck_foreground.npy'

catalog_path = data_path + 'vr_summaries/v01_sdss_cmass_north.h5'

kszpipe_path = data_path + 'kszpipe/sdss_dr12/cmass_north/'
kszpipe_cosmo_path = kszpipe_path + 'cosmology.pkl'
kszpipe_box_path = kszpipe_path + 'bounding_box.pkl'
kszpipe_d0_path = kszpipe_path + 'delta0_DR12v5_CMASS_North.h5'


fl_path = pipe_path + 'transfer_function_short_pub.h5'
# fl_path = pipe_path + 'transfer_function_pub.h5'


# Parameters chosen ahead of time to maximize snr with this particular dataset
R_FKP = 1.56
R_LWIDTH = 0.62
NTRIAL = 2048
DELTA_K = 0.006


def rel_error_inv_std(n):
    err_prop = np.sqrt(2/n)
    return np.array([1./(1 + err_prop), 1./(1 - err_prop)])


if __name__ == "__main__":
    om = OutputManager(base_path='output', title='make_xpower')

    act_pipe = ActPipe(map_path, ivar_path, beam_path, cl_ksz_path, cl_cmb_path,    
                          planck_enmask_path, om,
                          custom_l_weight=None, diag_plots=True, lmax=12000,
                          gal_mask_path=gal_mask_path)

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

    d0_k = box.read_grid(kszpipe_d0_path, fourier=True)
    real_pipe.init_d0_k(d0_k)

    alpha_2d, a_std = compute_estimator([act_pipe,], gal_pipe, r_lwidth=R_LWIDTH)

    print('=============================================')
    print('2D pipeline results:')
    print(f'a_ksz: {alpha_2d:3e}, a_ksz_nonnorm: {alpha_2d * a_std:3e}')
    print('=============================================')

    # make edges for k-bins
    kx = real_pipe.box.get_k_component(0, one_dimensional=True)
    kmax = np.max(kx)
    kmin = 0.
    nk = int(np.floor((kmax - kmin) / DELTA_K))
    assert nk > 0
    k_edges = np.linspace(kmin, kmax, nk + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
 
    real_pipe.add_galaxies(t_list=t_act_hp_list, wipe=True)
    alpha_3d, y_k = real_pipe.do_harmonic_sum_adj(t_act_hp_list, pa)
    grids = np.array((y_k, real_pipe.d0_k), dtype=complex)

    p_cross, counts = real_pipe.box.estimate_power_spectra(grids, k_edges, faxis=0)
    p_delta_y = p_cross[:,1,0]

    plt.figure(dpi=300.)
    plt.xscale('log')
    plt.yscale('log')
    plt.bar(k_edges[:-1], -p_delta_y, width=DELTA_K, align='edge')
    plt.savefig('plots/p_y_delta.png')


    print('=============================================')
    print('3D pipeline results:')
    print(f'a_ksz: {alpha_3d / a_std:3e}, a_ksz_nonnorm: {alpha_3d:3e}')
    print(f'2D/3D ratio: {alpha_2d/alpha_3d:3e}')
    print('=============================================')


    res_3d = np.zeros(NTRIAL)
    t0 = time.time()

    p_samples = np.empty((NTRIAL, nk, 2, 2))
    # just resample if NTRIAL > available sims?
    assert gal_pipe.temp_sims is not None
    assert NTRIAL <= gal_pipe.nsims
    for itrial in range(NTRIAL):
        t_sim_list = gal_pipe.temp_sims[itrial]
        a_ksz_3d, y_k = real_pipe.do_harmonic_sum_adj(t_sim_list, pa)
        grids = np.array((y_k, real_pipe.d0_k), dtype=complex)
        p_sample, bin_counts = real_pipe.box.estimate_power_spectra(grids, k_edges, faxis=0)
        res_3d[itrial] = a_ksz_3d

        p_samples[itrial] = p_sample

        tnow = time.time()
        time_per_iter = (tnow - t0) / (itrial + 1)

        print('=============================================')
        print(f'time_per_iter: {time_per_iter:.3e}')
        print('trial results:')
        print(f'a_ksz_3d: {alpha_3d / res_3d[:itrial+1].std():3e}')
        # print(f'a_ksz_2d: {a_ksz_2d / a_std:3e} a_ksz_3d: {a_ksz_3d / a_std:3e} 2d/3d: {a_ksz_2d / a_ksz_3d:3e}')
        print('=============================================')

    np.savez('data/crosspower_arrays.npz', k_edges, p_cross, p_samples)

    p_dy_samples = p_samples[:,:,1,0]


    # 95% conf interval
    p_dy_lo = np.quantile(p_dy_samples, 0.025, axis=0)
    p_dy_hi = np.quantile(p_dy_samples, 0.975, axis=0)

    plt.figure(dpi=300)
    plt.title(r'$-P_{\delta_0 Y}(k)$ binned in k')
    plt.bar(k_edges[:-1], -p_delta_y, width=DELTA_K, align='edge', alpha=0.5)
    plt.fill_between(k_centers, -(p_delta_y + p_dy_lo), -(p_delta_y + p_dy_hi), alpha=0.5)
    plt.xlabel(r'$k$ (Mpc$^{-1}$)')
    plt.ylabel(r'$-P_{\delta_0 Y}$, arbitrary')
    plt.savefig('plots/p_y_delta_95_conf.png')

    a3d_stdev = res_3d.std()
    a3d_norm_est = alpha_3d / a3d_stdev
    a3d_lo, a3d_hi = a3d_norm_est * rel_error_inv_std(NTRIAL)

    # a2d_stdev = res_2d.std()
    # a2d_result = alpha_2d / a2d_stdev
    # a2d_lo, a2d_hi = a2d_result * rel_error_inv_std(NTRIAL)
    # print(f'MC normalized alpha_2d: {a2d_result:.2f} {a2d_lo:.2f} {a2d_hi:.2f}')
    print(f'MC normalized alpha_3d: {a3d_norm_est:.2f} (lo/hi) {a3d_lo:.2f} {a3d_hi:.2f}')