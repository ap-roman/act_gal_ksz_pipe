from fnl_pipe.realspace import Padded3DPipe, PipeAdjoint
from fnl_pipe.pipe import ActPipe, GalPipe, compute_estimator

import kszpipe
from kszpipe.Cosmology import Cosmology 
from kszpipe.Box import Box

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
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


# crosspower_path = 'data/crosspower_arrays.npz'
# crosspower_path = 'data/crosspower_arrays_0.002.npz'
crosspower_path = 'data/crosspower_arrays_old.npz'

# Parameters chosen ahead of time to maximize snr with this particular dataset
R_FKP = 1.56
R_LWIDTH = 0.62
NTRIAL = 2048
# DELTA_K = 0.002


def rel_error_inv_std(n):
    err_prop = np.sqrt(2/n)
    return np.array([1./(1 + err_prop), 1./(1 - err_prop)])


def import_crosspower(path):
    ret = np.load(path)
    k_edges = ret['arr_0']
    p_cross = ret['arr_1']
    p_samples = ret['arr_2']

    return k_edges, p_cross, p_samples


def chi2(x, y, std, ks, kmax=0.1):
    return np.sum(((x-y) / std)**2)


def get_weight(x):
    return x[1] - x[0]


def conf_opt(y, p_lo, p_thresh, p_hi, *, conf=0.95, tol=1e-3):
    mask = y >= p_thresh
    p_sum = y[mask].sum() / y.sum()
    p_max = np.max(y)

    if np.abs(p_sum - (1 - conf)) <= tol: return mask, p_thresh
    elif p_sum > 1 - conf: return conf_opt(y, p_thresh, 0.5 * (p_thresh + p_hi), p_hi, conf=conf, tol=tol)
    else: return conf_opt(y, p_lo, 0.5 * (p_lo + p_thresh), p_thresh, conf=conf, tol=tol)


# Note: does not handle edge effects
def get_diff_2d(mask, sig=(1,0)):
    assert len(mask.shape) == 2
    padded = np.zeros(mask, mask.shape + [2,2], dtype=mask.dtype)
    padded[1:-1, 1:-1] = mask

    return mask != padded[1 + sig[0]:-1 + sig[0], 1 + sig[1]:-1 + sig[1]]


# return a list of x,y points that represent the boundary of a masked region
def get_contour(mask):
    diff_mask = np.zeros(mask.shape, dtype=bool)

    for s0 in (-1,0,1):
        for s1 in (-1,0,1):
            diff_mask = np.logical_or(diff_mask, get_diff_2d(mask, (s0, s1)))

    return diff_mask


def plot_posterior(posterior, ab_grid, confs=(0.68, 0.95)):
    confs_sorted = np.sort(confs)
    levels = [conf_opt(posterior, 0., 0., np.max(posterior), conf=c)[1] for c in confs_sorted]
    levels.append(posterior.max())
    # print(levels)

    plt.figure(dpi=300)
    plt.title(r'Posterior for (A,B) $P_{\delta_0 Y}$ Toy Model')
    ax = plt.gca()
    CS = ax.contour(*ab_grid, posterior, levels=levels)
    
    ldict = {}
    for level, conf in zip(levels, confs_sorted):
        ldict[level] = f'{100*conf:.0f}%'

    ax.clabel(CS, CS.levels, inline=True, fmt=ldict, fontsize=10)
    plt.axhline(0., color='black', ls='--')
    plt.xlabel('A')
    plt.ylabel('B')
    plt.savefig('plots/p_dy_ab_posterior.png')
    plt.close()


if __name__ == "__main__":

    k_edges, p_cross, p_samples = import_crosspower(crosspower_path)
    k_centers = 0.5 * (k_edges[1:] + k_edges[:-1])
    k2 = k_centers**2

    DELTA_K = k_edges[1] - k_edges[0]

    p_delta_y = p_cross[:,1,0]
    p_d0_obs = p_cross[:,1,1]
    p_d0_obs = 1e-3 * p_d0_obs / p_d0_obs[3] / k2[3]
    p_dy_samples = p_samples[:,:,1,0]

    # 95% conf interval
    p_dy_lo = np.quantile(p_dy_samples, 0.025, axis=0)
    p_dy_hi = np.quantile(p_dy_samples, 0.975, axis=0)

    p_dy_std = np.std(p_dy_samples, axis=0)
    p_dy_mean = np.mean(p_dy_samples, axis=0)


    corrs = np.mean(p_dy_samples[:, 1:] * p_dy_samples[:, :-1], axis=0) / (p_dy_std[1:] * p_dy_std[:-1])
    corrs2 = np.mean(p_dy_samples[:, 2:] * p_dy_samples[:, :-2], axis=0) / (p_dy_std[2:] * p_dy_std[:-2])

    cosmology = kszpipe.io_utils.read_pickle(kszpipe_cosmo_path)
    p_d0 = cosmology.Plin_k_z0(k_centers)
    p_d0 = p_d0 / p_d0[0]

    def optfun(x):
        return chi2(p_d0 * x[0], -k2 * p_delta_y, k2 * p_dy_std, k_centers)

    x0 = [1e-3,]
    res = minimize(optfun, x0)
    print('==================================')
    print(res)
    print('==================================')
    A = res.x[0]

    def optfun_fnl(x):
        return chi2(-k2 * p_delta_y, p_d0 * (x[0] + x[1] / k2), k2 * p_dy_std, k_centers)


    def posterior_grid(grid, weight=1):
        a_fnls, fnls = grid

        chi2_sum = np.sum(((p_d0[None, None, :] * (a_fnls[..., None] + fnls[..., None] / k2[None, None, :]) + k2[None, None, :] * p_delta_y[None, None, :]) / np.sqrt(2 * np.pi) / k2[None, None, :] / p_dy_std[None, None, :])**2, axis=-1)
        ret = np.exp(-0.5 * chi2_sum)

        norm = np.prod(1. / np.sqrt(2 * np.pi) / p_dy_std)

        return ret / ret.sum()

        # return ret * norm * weight


    x0 = [1e-3, 0.]
    res_fnl = minimize(optfun_fnl, x0)
    print('==================================')
    print(res_fnl)
    print('==================================')
    A_fnl, B_fnl = res_fnl.x
    n_afnl = 1024
    n_fnl = 1024

    a_sample = np.linspace(0, 2 * A_fnl, n_afnl)
    fnl_sample = np.linspace(-3 * B_fnl, 3 * B_fnl, n_fnl)
    weight = get_weight(a_sample) * get_weight(fnl_sample)

    grid = np.meshgrid(a_sample, fnl_sample, indexing='ij')

    posterior_samples = posterior_grid(grid, weight)

    p95_mask, p_95 = conf_opt(posterior_samples, 0., 0., np.max(posterior_samples))

    print(f'post_sum: {np.sum(posterior_samples):.3f}, {1 - np.sum(posterior_samples[p95_mask]):.3f}')
    plot_posterior(posterior_samples, grid)

    plt.figure(dpi=300)
    plt.title(f'adjacent bin correlation coefficient (spacing {DELTA_K:.3f}' + r' $Mpc^{-1}$)')
    plt.plot(k_edges[:-2], corrs, label='adjacent-bin corr')
    plt.plot(k_edges[:-3], corrs2, label='2-bin corr')
    plt.xlabel(r'$k$ (Mpc$^{-1}$)')
    plt.ylabel('correlation coefficient (bin i corr i+1)')
    plt.savefig('plots/corr_k_bins.png')

    plt.figure(dpi=300)
    plt.title(r'$-k^2P_{\delta_0 Y}(k)$ binned in $k$')
    # plt.bar(k_edges[:-1], -k2 * p_delta_y, width=DELTA_K, align='edge',
    #         alpha=0.5, label=r'$-k^2P_{\delta_0 Y}$')
    # plt.scatter(k_centers, -k2 * p_delta_y, label=r'$-k^2P_{\delta_0 Y}$')
    # plt.fill_between(k_centers, -k2 * (p_delta_y + p_dy_std), -k2 * (p_delta_y - p_dy_std),
    #                  alpha=0.5, label='+/-1 std')
    plt.plot(k_centers, A * p_d0, ls='--', label=r'$P_{\delta_0}$')
    plt.plot(k_centers, p_d0_obs, label=r'$P_{\delta_0}$ observed')
    plt.errorbar(k_centers, -k2 * p_delta_y, k2 * p_dy_std)
    plt.xlabel(r'$k$ (Mpc$^{-1}$)')
    plt.ylabel(r'$-k^2P_{\delta_0 Y}$, arbitrary')
    plt.legend()
    plt.savefig('plots/p_y_delta_95_conf.png')

    plt.figure(dpi=300)
    plt.title(r'bin SNR')
    plt.bar(k_edges[:-1], -p_delta_y / p_dy_std, width=DELTA_K, align='edge',
            alpha=0.5, label=r'bin snr')
    plt.xlabel(r'$k$ (Mpc$^{-1}$)')
    plt.ylabel(r'bin SNR')
    plt.legend()
    plt.savefig('plots/p_y_delta_snr.png')

    plt.figure(dpi=300)
    plt.title(r'$-k^2P_{\delta_0 Y}(k)$ binned in $k$, noise only')
    # plt.bar(k_edges[:-1], -k2 * p_dy_mean, width=DELTA_K, align='edge',
    #         alpha=0.5, label=r'$-k^2P_{\delta_0 Y}$ mean')
    plt.plot(k_centers, -k2 * p_dy_mean, label=r'$-k^2P_{\delta_0 Y}$ mean')
    plt.fill_between(k_centers, -k2 * p_dy_std, k2 * p_dy_std,
                     alpha=0.5, label='+/-1 std')
    plt.xlabel(r'$k$ (Mpc$^{-1}$)')
    plt.ylabel(r'$-k^2P_{\delta_0 Y}$, arbitrary')
    plt.legend()
    plt.savefig('plots/p_y_delta_noise.png')
