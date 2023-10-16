from fnl_pipe.util import OutputManager, get_yaml_dict, average_fl, downgrade_fl, get_fname, parse_act_beam
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut

from pixell import enplot, enmap

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import sys


def eval_in_region(fun, x0, x1, n):
    assert x0.shape[0] == x1.shape[0] == 2
    xy = np.array(np.meshgrid(*np.linspace(x0,x1,n).T, indexing='ij'))

    fvals = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            fvals[i, j] = fun(xy[:,i,j])
    # fvals = fun(xy)
    return fvals


def get_area_weight(map_t):
    cdelt = map_t.wcs.wcs.cdelt * np.pi / 180. # dec, ra
    dOmega = cdelt[0] * cdelt[1]
    pos = map_t.posmap() # map_centers
    decs, ras = pos
    weights = np.abs(dOmega * np.sin(pos[0]))
    return weights


# takes a reduced (averaged), debeamed cl_tz and returns the parameters of an
# exponential fit as well as uncertainties on the parameters
def get_fit_and_uncertainties(ells_coarse, cl_tz_red, cl_tz_std_red, ilmin, ilmax, x0=[1., 5000.], ntrial=1024):

    cl_tz_norm = 1./ cl_tz_red.mean()

    fit_weight = 1./cl_tz_std_red**2 / (1./cl_tz_red**2).sum()

    a0 = cl_tz_red.mean()

    cl_fit = cl_tz_red / cl_tz_red.mean()

    def exp_model(x):
        return x[0] * np.exp(-(ells_coarse/x[1])**2)

    def opt_fun(x):
        return ((exp_model[ilmin:ilmax] - cl_fit[ilmin:ilmax])**2 * fit_weights).sum()


if __name__ == "__main__":
    om = OutputManager(base_path='output', title='plot-cl-tz', logs=['log'])
    printlog = om.printlog

    config_file = sys.argv[1]
    printlog('got config file ' + config_file)
    config_dict = get_yaml_dict(config_file)
    printlog('dumping config')
    for key, value in config_dict.items():
        printlog(f'{key}: {value}')
    globals().update(config_dict)
    printlog('################## DONE ##################')

    cl_tz_mc = np.load(cl_tz_mc_path) * cl_tz_scale
    ntrial_mc = cl_tz_mc.shape[0]
    lmax = cl_tz_mc.shape[1] - 1
    printlog(f'NTRIAL_MC: {ntrial_mc}, lmax: {lmax}')

    map_weight = enmap.read_map(map_weight_path)
    map_frac = map_weight.sum()
    print('map weight sum: ', map_frac)

    assert lmax % NAVE_TZ_L == 0
    cl_tz_ave = np.empty((ntrial_mc, lmax + 1))
    cl_tz_ave_red = np.empty((ntrial_mc, lmax // NAVE_TZ_L + 1))

    for itrial_mc in range(ntrial_mc):
        cl_tz_ave[itrial_mc] = average_fl(cl_tz_mc[itrial_mc], NAVE_TZ_L)
        cl_tz_ave_red[itrial_mc] = downgrade_fl(cl_tz_mc[itrial_mc], NAVE_TZ_L)

    delta_confidence = (1 - confidence) / 2

    bl = parse_act_beam(beam_path)[1][:lmax + 1]
    bl2 = bl**2
    beam_ave = average_fl(bl, NAVE_TZ_L)

    ells_coarse = downgrade_fl(np.arange(lmax + 1), NAVE_TZ_L)
    cl_tz_data = np.load(cl_tz_sig_path) * cl_tz_scale
    cl_tz_signal = average_fl(cl_tz_data, NAVE_TZ_L)
    cl_tz_signal_debeam = downgrade_fl(cl_tz_data * 1./bl, NAVE_TZ_L)

    cl_tz_sig_red = downgrade_fl(cl_tz_data, NAVE_TZ_L) 

    cl_tt_act = average_fl(np.load(cl_tt_act_path), NAVE_TT)[LMIN_PLOT + 1:]
    cl_sim = np.load(cl_tt_sim_path)
    # cl_tt_sim = average_fl(np.load(cl_tt_sim_path), NAVE_TT)[LMIN_PLOT + 1:]
    cl_tt_sim = average_fl(cl_sim['cl_tot'], NAVE_TT)[LMIN_PLOT + 1:]
    cl_tt_cmb = average_fl(cl_sim['cl_cmb'], NAVE_TT)[LMIN_PLOT + 1:]
    cl_tt_fg = average_fl(cl_sim['cl_fg'], NAVE_TT)[LMIN_PLOT + 1:]
    cl_tt_noise = average_fl(cl_sim['cl_noise'], NAVE_TT)[LMIN_PLOT + 1:]

    fl_cmb_model = average_fl(np.load(fl_sim_path), NAVE_TT)[LMIN_PLOT + 1:] * beam_ave[LMIN_PLOT + 1:]**2

    # normal cdf probability of +/- one sigma
    # we have many DoFs in the averaged spectra and expect gaussanity
    p_sigma_p = 0.5 + 0.3413
    p_sigma_m = 0.5 - 0.3413

    cl_tz_p = np.quantile(cl_tz_ave, p_sigma_p, axis=0)
    cl_tz_med = np.quantile(cl_tz_ave, 0.5, axis=0)
    cl_tz_m = np.quantile(cl_tz_ave, p_sigma_m, axis=0)

    cl_tz_p_red = np.quantile(cl_tz_ave_red, p_sigma_p, axis=0)
    cl_tz_med_red = np.quantile(cl_tz_ave_red, 0.5, axis=0)
    cl_tz_m_red = np.quantile(cl_tz_ave_red, p_sigma_m, axis=0)

    cl_tz_sigma = (cl_tz_signal - cl_tz_med) / (cl_tz_m - cl_tz_med)
    cl_tz_significance = np.sqrt(((cl_tz_sigma[2001:6001])**2).sum() / NAVE_TZ_L)
    printlog(f'cl_tz l=2000->l=6000 significance: {cl_tz_significance:.3e}')

    # these should be fairly symmetrical so this operation is valid
    # cl_tz_sigma = (np.abs(cl_tz_p) + np.abs(cl_tz_m)) / 2

    cl_tz_low = np.quantile(cl_tz_ave, delta_confidence, axis=0)
    cl_tz_high = np.quantile(cl_tz_ave, 1 - delta_confidence, axis=0)

    def exp_model(x):
        return x[0] * np.exp(-(ells_coarse / x[1])**2)

    a0 = cl_tz_signal_debeam[2]

    cl_tz_fit = cl_tz_signal_debeam / a0


    cl_tz_std_red = np.abs(cl_tz_m_red - cl_tz_med_red)

    # ivar_cltz = 1./cl_tz_std_red**2

    # wsum = ivar_cltz.sum()
    # weight = ivar_cltz / wsum

    # opt_coeff = 1e2
    # opt_to_loglike = a0**2 * wsum / opt_coeff

    # def opt_fun(x):
    #     return opt_coeff * (((exp_model(x)[ifit_min:ifit_max] - cl_tz_fit[ifit_min:ifit_max])**2) * weight[ifit_min:ifit_max]).sum(axis=0)

    # x0 = np.array([1., 5000.])
    # # dx = np.array([0.001, 5])

    # # opt_fun_xy = eval_in_region(opt_fun, x0 - 0.5 * dx, x0 + 0.5 * dx, 32)
    # # plt.figure(dpi=300, facecolor='w')
    # # plt.title(r'$Cl^{TZ}$ model optimization function')
    # # plt.imshow(opt_fun_xy)
    # # plt.colorbar()
    # # om.savefig(f'exp_opt_eval_{freq_str}')
    # # plt.close()

    # # res = minimize(opt_fun, x0, method='Powell')
    # res = minimize(opt_fun, x0)
    # printlog('===================== OPT RESULT =====================')
    # printlog(res)
    # printlog('=================== END OPT RESULT ===================')
    # x_opt = res.x
    # hess_inv = res.hess_inv / opt_to_loglike
    # printlog('normalized hessian inv')
    # std_alpha = np.sqrt(hess_inv[0,0]) # best-fit standard error on the alpha coefficient
    # printlog(f'fit params C, l0: {a0 * x_opt[0]:.3e}, {x_opt[1]:.3e}')
    # printlog(f'SNR on alpha_ksz: {x_opt[0]/std_alpha:.3e}')


    plt.figure(dpi=300, facecolor='w')
    plt.plot(ells_coarse[ifit_min:], cl_tz_signal_debeam[ifit_min:], label='data')
    plt.plot(ells_coarse[ifit_min:], a0 * exp_model(x_opt)[ifit_min:], label='model')
    # plt.plot(ells_coarse[4:], exp_model(x_opt)[4:], label='one-sigma bar')
    plt.ylabel('cl_tz (uK^2)')
    plt.xlabel('l')
    plt.legend()
    om.savefig(f'cl_tz_model_{freq_str}')

    dof = ifit_max - ifit_min
    std = np.sqrt(2 * dof)
    # dof_short = 8

    # compute model significances
    res_model = ((a0 * exp_model(x_opt) - cl_tz_signal_debeam) / np.abs(cl_tz_m_red - cl_tz_med_red))
    # res_sum_short = (res_model[4:12]**2).sum()
    res_sum = (res_model[ifit_min:ifit_max]**2).sum()
    printlog('fit goodness for l between 2000 and 8000')
    printlog(f'expected chi2: {dof}, actual residual chi2: {res_sum:.3e}, departure (sigma): {(res_sum - dof)/std:.3e}')

    # printlog('fit goodness for l between 2000 and 10000')
    # printlog(f'expected chi2: {dof}, actual residual chi2: {res_sum:.3e}, departure (sigma): {(res_sum/2/dof - 0.5):.3e}')

    plt.figure(dpi=300, facecolor='w')
    plt.title('Model Residuals')
    plt.plot(ells_coarse[ifit_min:ifit_max], res_model[ifit_min:ifit_max], label='data')
    plt.ylabel('model residual (sigma)')
    plt.xlabel('l')
    om.savefig(f'cl_tz_model_residual_{freq_str}')

    ells = np.arange(lmax + 1)
    ells_ave = average_fl(ells, NAVE_TZ_L)[LMIN_PLOT+1:]
    ells_ave_tt = average_fl(ells, NAVE_TT)[LMIN_PLOT + 1:]
    norm = ells_ave * (ells_ave + 1) / 2 / np.pi
    norm_tt = ells_ave_tt * (ells_ave_tt + 1) / 2 / np.pi
    norm_coarse = ells_coarse * (ells_coarse + 1) / 2 / np.pi
    ells_red = downgrade_fl(ells, NAVE_TZ_L)

    cl_tz_corr = np.corrcoef(cl_tz_ave_red, rowvar=False)

    # print(cl_tz_corr.shape)
    # assert np.all(cl_tz_corr.shape == [1 + lmax//NAVE_TZ_L, 1 + lmax//NAVE_TZ_L])

    plt.figure(dpi=300, facecolor='w')
    plt.title('Apparent transfer function')
    plt.plot(ells_ave_tt, fl_cmb_model)
    plt.xlabel(r'$l$')
    plt.ylabel(r'fl_empirical')
    plt.yscale('log')
    om.savefig(f'fl_camb_{freq_str}.png')

    plt.figure(dpi=300, facecolor='w')
    plt.title(r'$C_l^{\tilde{T} \tilde{T}}$ Components')
    plt.plot(ells_ave_tt, norm_tt * cl_tt_sim, label=f'total {freq_str}')
    plt.plot(ells_ave_tt, norm_tt * cl_tt_cmb, label=f'camb cmb {freq_str}')
    plt.plot(ells_ave_tt, norm_tt * cl_tt_fg, label=f'fg {freq_str}')
    plt.plot(ells_ave_tt, norm_tt * cl_tt_noise, label=f'noise {freq_str}')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{\tilde{T}\tilde{T}}l(l+1)/2\pi$')
    plt.yscale('log')
    plt.legend()
    om.savefig(f'cl_tt_components_{freq_str}.png')

    plt.figure(dpi=300, facecolor='w')
    plt.title(r'$C_l^{\tilde{T} \tilde{T}}$ Component Relative Power')
    plt.plot(ells_ave_tt, cl_tt_cmb / cl_tt_sim, label=f'camb cmb {freq_str}')
    plt.plot(ells_ave_tt, cl_tt_fg / cl_tt_sim, label=f'fg {freq_str}')
    plt.plot(ells_ave_tt, cl_tt_noise / cl_tt_sim, label=f'noise {freq_str}')
    plt.xlabel(r'$l$')
    plt.ylabel(r'Relative Power')
    plt.legend()
    om.savefig(f'cl_tt_components_relative_{freq_str}.png')

    plt.figure(dpi=300, facecolor='w')
    plt.title(r'$C_l^{\tilde{T} \tilde{T}}$')
    plt.plot(ells_ave_tt, norm_tt * cl_tt_act, label=f'act {freq_str}')
    plt.plot(ells_ave_tt, norm_tt * cl_tt_sim, label=f'sim {freq_str}')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{\tilde{T}\tilde{T}}l(l+1)/2\pi$')
    plt.yscale('log')
    plt.legend()
    om.savefig(f'cl_tt_{freq_str}.png')

    plt.figure(dpi=300, facecolor='w')
    plt.title('data vs sim comparison')
    plt.plot(ells_ave_tt, cl_tt_act/cl_tt_sim)
    plt.xlabel(r'$l$')
    plt.ylabel(r'cl_data / cl_sim')
    plt.ylim(0.8,1.2)
    plt.xscale('log')
    om.savefig(f'cl_tt_compare_{freq_str}.png')

    plt.figure(dpi=300, facecolor='w')
    plt.title(r'$C_l^{\tilde{T}Z}$ correlation matrix')
    plt.imshow(cl_tz_corr[1:,1:], extent=[1,12000,1,12000])
    plt.colorbar()
    plt.xlabel(r'$l$ bin ' + f'(lwidth={NAVE_TZ_L})')
    plt.ylabel(r'$l$ bin ' + f'(lwidth={NAVE_TZ_L})')
    om.savefig(f'cl_tz_corr_{freq_str}.png')

    plt.figure(dpi=300, facecolor='w')
    plt.title(r'$C_l^{\tilde{T}Z}b_l^{-1}$')
    plt.plot(ells[LMIN_PLOT+1:], norm * (np.abs(cl_tz_signal) / beam_ave)[LMIN_PLOT+1:], label=r'$C_l^{\tilde{T}Z}b_l^{-1}$')
    plt.plot(ells[LMIN_PLOT+1:], norm * (np.abs(cl_tz_low) / beam_ave)[LMIN_PLOT+1:], label=f'low {delta_confidence:.3f}')
    plt.plot(ells[LMIN_PLOT+1:], norm * (np.abs(cl_tz_high) / beam_ave)[LMIN_PLOT+1:], label=f'high {1 - delta_confidence:.3f}')
    plt.legend()
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{\tilde{T}Z}b_l^{-1}l(l+1)$')
    plt.yscale('log')
    plt.grid(which='both')
    plt.axhline(0,color='black')
    om.savefig(f'cl_tz_log_{freq_str}.png')

    ymax = norm[LREF] * np.max([np.abs((cl_tz_signal / beam_ave)[LREF]),
                      np.abs((cl_tz_low / beam_ave)[LREF]),
                      np.abs((cl_tz_high / beam_ave)[LREF])])

    plt.figure(dpi=300, facecolor='w')
    plt.title(r'$C_l^{\tilde{T}Z}b_l^{-1}$')
    plt.plot(ells[LMIN_PLOT+1:], norm * (cl_tz_signal / beam_ave)[LMIN_PLOT+1:], label=r'$C_l^{\tilde{T}Z}b_l^{-1}$')
    plt.plot(ells[LMIN_PLOT+1:], norm * (cl_tz_low / beam_ave)[LMIN_PLOT+1:], label=f'low {delta_confidence:.3f}')
    plt.plot(ells[LMIN_PLOT+1:], norm * (cl_tz_high / beam_ave)[LMIN_PLOT+1:], label=f'high {1 - delta_confidence:.3f}')
    plt.legend()
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{\tilde{T}Z}b_l^{-1}l(l+1)$')
    plt.grid(which='both')
    plt.axhline(0,color='black')
    om.savefig(f'cl_tz_{freq_str}.png')

    plt.figure(dpi=300, facecolor='w')
    plt.title(r'$C_l^{\tilde{T}Z}b_l^{-1}$')
    plt.plot(ells[LMIN_PLOT+1:], norm * (cl_tz_signal / beam_ave)[LMIN_PLOT+1:], label=r'$C_l^{\tilde{T}Z}b_l^{-1}$')
    plt.plot(ells[LMIN_PLOT+1:], norm * (cl_tz_low / beam_ave)[LMIN_PLOT+1:], label=f'low {delta_confidence:.3f}')
    plt.plot(ells[LMIN_PLOT+1:], norm * (cl_tz_high / beam_ave)[LMIN_PLOT+1:], label=f'high {1 - delta_confidence:.3f}')
    plt.plot(ells_coarse[ifit_min:ifit_max], 
            (norm_coarse * a0 * exp_model(x_opt))[ifit_min:ifit_max], label='fit')
    plt.legend()
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{\tilde{T}Z}b_l^{-1}l(l+1)$')
    plt.xlim(0,8000)
    plt.ylim(-ymax, ymax)
    plt.grid(which='both')
    plt.axhline(0,color='black')
    om.savefig(f'cl_tz_zoomed_{freq_str}.png')

    plt.figure(dpi=300, facecolor='w')
    plt.title(r'Per-bin $C_l^{\tilde{T}Z}$ significance')
    plt.plot(ells[LMIN_PLOT+1:], ((cl_tz_signal - cl_tz_med) / (cl_tz_m - cl_tz_med))[LMIN_PLOT+1:], 
             label=r'$C_l^{\tilde{T}Z}b_l^{-1}$')
    plt.xlabel(r'$l$')
    plt.ylabel(r'bin $\sigma$')
    plt.grid(which='both')
    plt.axhline(0,color='black')
    om.savefig(f'cl_tz_sigma_{freq_str}.png')