from fnl_pipe.realspace import Padded3DPipe, PipeAdjoint, concat_shape
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
NTRIAL = 16
TOL=1e-12


def f_eq(a,b,tol=1e-9):
    return np.all(np.abs(a - b)/np.sqrt(np.abs(a) * np.abs(b)) <= tol)


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


def dot(a,b):
    return np.sum(a * np.conj(b))


def rel_err(a,b):
    return np.abs(a - b)/np.sqrt(np.abs(a)*np.abs(b))


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

    real_pipe = Padded3DPipe(gal_pipe=gal_pipe, cosmology=cosmology)
    
    pa = PipeAdjoint(real_pipe)
    pa.init_all(box, kszpipe_d0_path, init_gal=True)

    # Verify rdot-rdotadj adjointness
    print('Verifying rdot-rdotadj agreement')
    err1 = np.empty(NTRIAL)
    err2 = np.empty(NTRIAL)
    for i in range(NTRIAL):
        v = np.empty(concat_shape(3, gal_pipe.ngal))
        w = np.empty(gal_pipe.ngal)

        for j in range(3):
            proto_vj = box.simulate_white_noise(fourier=False, simulate_dc=False)
            v[j] = box.interpolate(proto_vj, real_pipe.gal_pos_3d.T, periodic=False)

        proto_w = box.simulate_white_noise(fourier=False, simulate_dc=False)
        w = box.interpolate(proto_w, real_pipe.gal_pos_3d.T, periodic=False)
        
        av = pa.three_dot(v)
        wadj = pa.three_dot_adj(w)
        dot1 = (w * av).sum()
        dot2 = (wadj * v).sum()

        # print(dot1, dot2, rel_err(dot1, dot2))
        err1[i] = dot1
        err2[i] = dot2
    assert f_eq(err1, err2, tol=TOL)
    print('PASS')

    print('Verifying B-Badj agreement')
    # Verify B-Badj adjointness
    for i in range(NTRIAL):
        v = np.empty(concat_shape(3, box.rshape))
        proto_w = np.empty(concat_shape(3, box.rshape))
        w = pa.box.interpolate(box.simulate_white_noise(fourier=False, simulate_dc=False), pa.pipe.gal_pos_3d.T,
                                                      periodic=False)
        
        assert w.shape == np.array((pa.pipe.gal_pipe.ngal,))

        for j in range(3):
            v[j] = box.simulate_white_noise(fourier=False, simulate_dc=False)

        av = pa.B(v)
        wadj = pa.B_adj(w)
        # print(av.shape, w.shape, wadj.shape, v.shape)
        dot1 = (w * av).sum()
        # print((wadj * v).shape)
        dot2 = (wadj * v).sum()

        # print(dot1, dot2, rel_err(dot1, dot2))
        err1[i] = dot1
        err2[i] = dot2
    assert f_eq(err1, err2, tol=TOL)
    print('PASS')

    # print('Verifying interp-extirp agreement')
    # err1 = np.zeros(NTRIAL)
    # err2 = np.zeros(NTRIAL)
    # # Verify interp-exterp adjointness
    # for i in range(NTRIAL):
    #     v = np.empty(concat_shape(3, box.rshape))
    #     proto_w = np.empty(concat_shape(3, box.rshape))
    #     w = np.empty(concat_shape(3, gal_pipe.ngal))

    #     for j in range(3):
    #         v[j] = box.simulate_white_noise(fourier=False, simulate_dc=False)
    #         proto_w[j] = box.simulate_white_noise(fourier=False, simulate_dc=False)
    #         w[j] = box.interpolate(proto_w[j], real_pipe.gal_pos_3d.T, periodic=False)

    #     for j in range(3):
    #         av = pa.interp(v[j])
    #         wadj = pa.exterp(w[j])
    #         # print(av.shape, w.shape, wadj.shape, v.shape)
    #         dot1 = np.dot(w[j], av)
    #         dot2 = (wadj * v[j]).sum()

    #         # print(dot1, dot2, rel_err(dot1, dot2))
    #         err1[i] += dot1
    #         err2[i] += dot2
    # assert f_eq(err1, err2, tol=TOL)
    # print('PASS')

    print('Verifying interp-extir adjointness')
    err1 = np.zeros(NTRIAL)
    err2 = np.zeros(NTRIAL)
    # Verify interp-exterp adjointness
    for i in range(NTRIAL):
        v = np.empty(concat_shape(3, box.rshape))
        proto_w = np.empty(concat_shape(3, box.rshape))
        w = np.empty(concat_shape(3, gal_pipe.ngal))

        for j in range(3):
            v[j] = box.simulate_white_noise(fourier=False, simulate_dc=False)
            proto_w[j] = box.simulate_white_noise(fourier=False, simulate_dc=False)
            w[j] = box.interpolate(proto_w[j], real_pipe.gal_pos_3d.T, periodic=False)

        for j in range(3):
            av = pa.interp(v[j])
            wadj = pa.extirp(w[j])
            # print(av.shape, w.shape, wadj.shape, v.shape)
            dot1 = np.dot(w[j], av)
            dot2 = (wadj * v[j]).sum()

            # print(dot1, dot2, rel_err(dot1, dot2))
            err1[i] += dot1
            err2[i] += dot2
    assert f_eq(err1, err2, tol=TOL)
    print('PASS')

    print('Verifying A-Adj agreement')
    err1 = np.zeros(NTRIAL)
    err2 = np.zeros(NTRIAL)
    # Test A adjoint correctness:
    for i in range(NTRIAL):
        v = box.simulate_white_noise(fourier=True, simulate_dc=False)

        w = np.empty(concat_shape(3, box.rshape))
        for j in range(3):
            w[j] = box.simulate_white_noise(fourier=False, simulate_dc=False)

        for j in range(3):
            av = pa.A(v, j)
            dot1 = box.dot(w[j], av)
            wadj = pa.A_adj(w[j], j)
            dot2 = box.dot(wadj, v)

            # print(dot1, dot2, rel_err(dot1, dot2))
            err1[i] += dot1
            err2[i] += dot2
    assert f_eq(err1, err2, tol=TOL)
    print('PASS')


    # # Verify aprime/aprime_adj adjointness
    # for i in range(NTRIAL):
    #     v = box.simulate_white_noise(fourier=True, simulate_dc=False)
    #     w = box.simulate_white_noise(fourier=False, simulate_dc=False)

    #     a_v = pa.Aprime(v)
    #     adj_w = pa.Aprime_adj(w)

    #     dot1 = box.dot(w, a_v)
    #     dot2 = box.dot(adj_w, v)

    #     print(dot1, dot2, rel_err(dot1, dot2))

    print('Verifying FFT/IFFT agreement')
    # Verify fft self-adjointness
    for i in range(NTRIAL):
        v = box.simulate_white_noise(fourier=True, simulate_dc=False)
        w = box.simulate_white_noise(fourier=False, simulate_dc=False)

        a_v = box.fft(v)
        adj_w = box.fft(w)

        dot1 = box.dot(w, a_v)
        dot2 = box.dot(adj_w, v)

        # print(dot1, dot2, rel_err(dot1, dot2))
        err1[i] = dot1
        err2[i] = dot2
    assert f_eq(err1, err2, tol=TOL)
    print('PASS')
