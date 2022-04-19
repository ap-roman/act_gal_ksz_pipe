from fnl_pipe.realspace import Padded3DPipe
from fnl_pipe.pipe import ActPipe, GalPipe, compute_estimator
from fnl_pipe.util import Timer

import kszpipe
from kszpipe.Cosmology import Cosmology 
from kszpipe.Box import Box

import numpy as np
import matplotlib.pyplot as plt

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


def do_masked_histogram(ar, base='', std_lim=3, nbins=500):
    std = ar.std()
    mean = ar.mean()
    mask = np.abs(ar - mean) <= std_lim * std

    print(base + f' (mean/rms): {mean:.3e}/{std:.3e}')

    ar_m = ar[mask]

    plt.figure(dpi=300)
    plt.title(base + ' histogram')
    plt.hist(ar_m, bins=1000)
    plt.axvline(mean, color='black')
    plt.xlabel('delta vr (relative error)')
    plt.savefig('plots/' + base + '.png')


if __name__ == "__main__":
    act_pipe = ActPipe(map_path, ivar_path, beam_path, cl_ksz_path, cl_cmb_path,    
                          planck_enmask_path,
                          custom_l_weight=None, diag_plots=True, lmax=12000)

    t0 = time.time()

    ct = Timer()

    act_pipe.import_data()
    act_pipe.update_metadata(r_fkp=R_FKP, r_lwidth=R_LWIDTH, gal_path=catalog_path)
    act_pipe.import_fl_nl(fl_path)
    ct.start()
    act_pipe.compute_pixel_weight()
    act_pipe.compute_sim_spectra(make_plots=False)
    act_pipe.compute_l_weight()
    ct.stop()

    gal_pipe = GalPipe(catalog_path, act_pipe, diag_plots=False)
    gal_pipe.import_data()
    ct.start()
    gal_pipe.make_vr_list()
    vr_2d = gal_pipe.vr_list
    ct.stop()

    cosmology = kszpipe.io_utils.read_pickle(kszpipe_cosmo_path)
    box = kszpipe.io_utils.read_pickle(kszpipe_box_path)

    real_pipe = Padded3DPipe(gal_pipe, cosmology)
    real_pipe.init_from_box(box)

    ct.start()
    d0_k = box.read_grid(kszpipe_d0_path, fourier=True)
    
    tmath = time.time()

    real_pipe.init_d0_k(d0_k)
    real_pipe.make_vr_grid()
    real_pipe.make_vr_list()

    t_hp_list = gal_pipe.get_t_list(act_pipe.get_t_pseudo_hp())

    a_ksz_2d = np.sum(vr_2d * t_hp_list)
    a_ksz_3d = np.sum(real_pipe.vr_list * t_hp_list)
    ct.stop()

    rel_ksz = (a_ksz_3d - a_ksz_2d) / a_ksz_2d

    print(f'ksz estimators (2d/3d): {a_ksz_2d:.4e} {a_ksz_3d:.4e} {rel_ksz:.5f}')

    vr_diff = real_pipe.vr_list - vr_2d
    vr_diff_rel = (real_pipe.vr_list - vr_2d) / vr_2d

    tfinal = time.time()

    print(f'overall run time: {tfinal - t0:.2f} s')
    print(f'compute time: {ct.dt:.2f} s')

    do_masked_histogram(vr_diff, base='vr_error')
    do_masked_histogram(vr_diff_rel, base='vr_relative_error')