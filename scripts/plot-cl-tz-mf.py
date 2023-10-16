from fnl_pipe.util import OutputManager, get_yaml_dict, average_fl, downgrade_fl, get_fname, parse_act_beam
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut

from pixell import enplot, enmap

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from scipy.optimize import minimize
from scipy import stats

import sys


CONF_DELTA = 0.3413
P_SIGMA_LO = 0.5 - CONF_DELTA
P_SIGMA_HI = 0.5 + CONF_DELTA


def get_freq_str(freq):
    assert freq == 90 or freq == 150
    if freq == 90: return 'f090'
    if freq == 150: return 'f150'


labels_f090 = ['f090', 'f090 vsub', 'f090 tszmask', 'f090-f150 diff', 'f090 vsub tszmask']
labels_f150 = ['f150', 'f150 vsub', 'f150 tszmask', 'f150 diff', 'f150 vsub tszmask'] # f150 diff is not used
valid_labels = labels_f090 + labels_f150

nlabels = len(valid_labels)

colors_f090 = colormaps['viridis'](np.linspace(0,1,nlabels)[0::2])
colors_f150 = colormaps['viridis'](np.linspace(0,1,nlabels)[1::2])

def update_cdict(labels, colors, cdict):
    for label, color in zip(labels, colors):
        cdict[label] = color

color_map = {}
update_cdict(labels_f090, colors_f090, color_map)
update_cdict(labels_f150, colors_f150, color_map)

# color_map = {'f090': 'red', 'f090 vsub': 'salmon', 'f090 tszmask': 'sienna', 'f090 vsub tszmask': 'darkorange', 
#              'f150': 'blue', 'f150 vsub': 'cornflowerblue', 'f150 tszmask': 'slategray', 'f150 vsub tszmask': 'midnightblue'}

def make_cltz_comparison(om, cltzs, labels, lmin_plot=0, title_suffix='', show_model=True):
    assert len(cltzs) == len(labels)
    
    for label in labels:
        assert label in valid_labels

    ncltz = len(cltzs)

    # colors = ['tomato', 'royalblue', 'forestgreen', 'orange']
    # err_colors = ['lightsalmon', 'steelblue', 'forestgreen', 'orange' ]

    colors = [color_map[label] for label in labels]
    # err_colors = err_colors[:ncltz]
    err_colors = colors.copy()

    plt.figure(dpi=300, facecolor='w')
    plt.title(r'$C_l^{\tilde{T}Z}b_l^{-1}$ ' + title_suffix)
    for cltz, label, color, ecolor in zip(cltzs, labels, colors, err_colors):
        ells = cltz.ells[lmin_plot:]
        ells_red = cltz.ells_red[lmin_plot:]
        norm_red = cltz.norm_red[lmin_plot:]
        norm = ells * (ells + 1) / 2 / np.pi 
        cl_tz_sig_red = cltz.cl_tz_sig_red[lmin_plot:]
        cl_tz_std_red = cltz.cl_tz_std_red[lmin_plot:]

        def exp_model(x):
            return x[0] * np.exp(-(ells / x[1])**2)

        plt.errorbar(ells_red, norm_red * cl_tz_sig_red,
                     yerr=norm_red * cl_tz_std_red, label=label, color=color)
        if show_model:
            plt.plot(cltz.ells_fit, cltz.norm_fit * cltz.best_model, color=color, linestyle='--', label=label + ' gaussian model')
        # plt.plot(ells_red, -norm_red * cl_tz_std_red, label=label + ' noise level', color=ecolor)

        # plt.plot(ells_red, norm * (cl_tz_low / beam_ave)[LMIN_PLOT+1:], label=f'low {delta_confidence:.3f}')
        # plt.plot(ells_red, norm * (cl_tz_high / beam_ave)[LMIN_PLOT+1:], label=f'high {1 - delta_confidence:.3f}')
        # plt.plot(ells_coarse[ifit_min:ifit_max], 
        #         (norm_coarse * a0 * exp_model(x_opt))[ifit_min:ifit_max], label='fit')
    plt.legend()
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{\tilde{T}Z}b_l^{-1}l(l+1)$')
    plt.xlim(1500, 8500)
    plt.ylim(-1e-6, 1e-6)
    plt.grid(which='both')
    plt.axhline(0,color='black')
    om.savefig(f'cl_tz_' + title_suffix+'.png')
    plt.close()


def get_llims_reduced(llims, nave):
    return np.array((np.floor(llims[0]/nave).astype(int), np.ceil(llims[1]/nave).astype(int)))


def get_fit(ar, llims_red):
    return ar[llims_red[0]:llims_red[1]]


def get_red_fit(ar, nave, llims_red):
    return downgrade_fl(ar, nave)[llims_red[0]:llims_red[1]]


# take llims in native format (ells)
def get_red_fit2(ar, nave, llims):
    llims_red = get_llims_reduced(llims, nave)
    return get_red_fit(ar, nave, llims_red)


def chi2_to_sigma(chi2, ndof):

    p_value = 1 - stats.chi2.cdf(chi2, df=ndof)

    norm_sigmas = np.linspace(-30,30,1024)
    norm_ps = 1 - stats.norm.cdf(norm_sigmas)
    best_fit = np.argmin()

    return p_val, sigma 


class FitWrapper:
    def _update_cl_tz(self, cl_tz):
        assert self.lmax == len(cl_tz) - 1
        cl_tz_tmp = get_red_fit(cl_tz, self.nave_tz, self.llims_red)
        
        self.ascale = np.abs(cl_tz_tmp).mean()
        self.cl_tz_fit =  cl_tz_tmp / self.ascale
        self.cl_tz_raw = cl_tz_tmp
        self.of2chi2 = self.ascale**2 * self.ivar_norm / self.ndof

    def update_cl_tz_rand(self):
        assert self.do_rand
        nrand = len(self.ells_fit)
        cl_tz_tmp = np.random.normal(size=nrand) * self.cl_tz_std_fit

        self.ascale = np.abs(cl_tz_tmp).mean()
        self.cl_tz_fit =  cl_tz_tmp / self.ascale
        self.cl_tz_raw = cl_tz_tmp
        self.of2chi2 = self.ascale**2 * self.ivar_norm / self.ndof

    def __init__(self, cl_tz_mc, cl_tz_std_fit, llims, *, cl_tz=None, nave_tz=50, om=None, freq_str=''):
        lmax = cl_tz_mc.shape[1] - 1
        self.ells = np.arange(lmax + 1)

        self.om = om
        self.freq_str = freq_str

        self.do_rand = cl_tz is None

        self.lmax = lmax
        self.nave_tz = nave_tz
        self.llims = llims
        self.llims_red = get_llims_reduced(llims, nave_tz)
        self.nell_fit = np.ptp(self.llims_red)
        self.ndof = np.ptp(self.llims_red) - 3

        self.cl_tz_std_fit = cl_tz_std_fit
        
        ivar = 1. / self.cl_tz_std_fit**2
        self.ivar_norm = ivar.sum()
        self.ivar_weight_fit = ivar / self.ivar_norm

        self.ells_fit = get_red_fit(self.ells, nave_tz, self.llims_red)
        self.norm_fit = get_red_fit(self.ells * (self.ells + 1)/2/np.pi, nave_tz, self.llims_red)

        if self.do_rand:
            self.update_cl_tz_rand()
        else:
            self._update_cl_tz(cl_tz)


    # def _exp_model_fit(self, x):
    #     return x[0] * np.exp(-(self.ells_fit / x[1] / self.lscale)**2)

    def _exp_model_2d(self, alist, llist):
        return alist[:, None, None] * np.exp(-(self.ells_fit[None, None, :] / llist[None, :, None])**2)

    def _exp_model_1d(self, alist, l):
        return alist[:, None] * np.exp(-(self.ells_fit[None, :] / l)**2)

    def exp_model(self, a, l):
        return a * np.exp(-(self.ells_fit / l)**2)

    def _opt_fun_2d(self, alist, llist):
        return self.of2chi2 * (((self._exp_model_2d(alist, llist) - self.cl_tz_fit[None,None,:])**2 * self.ivar_weight_fit[None,None,:]).sum(axis=-1))

    def _opt_fun_1d(self, alist, l):
        return self.of2chi2 * (((self._exp_model_1d(alist, l) - self.cl_tz_fit[None,:])**2 * self.ivar_weight_fit[None,:]).sum(axis=-1))

    def _opt_fun(self, a, l):
        return self.of2chi2 * (((self.exp_model(a, l) - self.cl_tz_fit)**2 * self.ivar_weight_fit).sum(axis=-1))

    def do_fit(self, x0, dxm, dxp, scale, dims, iiter=0, niter=3, suffix=''):
        x0 = np.array(x0)
        dxm = np.array(dxm)
        dxp = np.array(dxp)

        alist = np.linspace(x0[0] + dxm[0], x0[0] + dxp[0], dims[0])
        llist = np.linspace(x0[1] + dxm[1], x0[1] + dxp[1], dims[1])

        of = self._opt_fun_2d(alist, llist)

        da = np.ptp(alist)
        dl = np.ptp(llist)

        # plt.figure(dpi=300, facecolor='w')
        # plt.title(r'$\chi^2$ per DoF ' + f'(DoF={self.ndof})')
        # plt.imshow(of[::-1,:], extent=[np.min(llist),np.max(llist),np.min(alist),np.max(alist)], aspect=dl/da)
        # plt.colorbar()
        # plt.xlabel('l')
        # plt.ylabel('A - arbitrary amplitude')
        # self.om.savefig(f'do_fit_of_{self.freq_str}_iiter_{iiter}{suffix}.png')
        # plt.close()

        imin = np.argmin(of)
        ia, il = np.unravel_index(imin, dims)
        amin = alist[ia]
        lmin = llist[il]

        if iiter < niter:
            return self.do_fit([amin, lmin], scale * dxm, scale * dxp, scale, dims, iiter + 1, niter, suffix)

        self.converged = True
        self.best_x = np.array((amin, lmin))
        # self.sig_chi2 = self.of2chi2 * (self.cl_tz_fit**2 * self.ivar_weight_fit)[self.llims[0]:self.llims[1]].sum() * self.ndof
        self.sig_chi2 = ((self.cl_tz_raw / self.cl_tz_std_fit)**2).sum()
        self.best_chi2 = of[ia, il] * self.ndof
        # self.chi2_goodness = (self.best_chi2 - 1) / np.sqrt(2 / self.ndof) 
        self.chi2_goodness = (self.best_chi2 - self.ndof)
        return self.best_x

    def _find_sigma(self, x0, da, ntrial=1024):
        printlog = self.om.printlog

        chi2_0 = self._opt_fun(*x0)
        a0, l0 = x0
        # sigma = np.sqrt(2 / self.ndof)
        sigma = 1. / self.ndof
        alist = np.linspace(a0 - da, a0 + da, ntrial)
        dc2 = self._opt_fun_1d(alist, l0) - chi2_0

        ia0 = np.argmin(dc2)
        ia_sigma_left = np.argmin((dc2[:ia0] - sigma)**2)
        ia_sigma_right = ia0 + np.argmin((dc2[ia0:] - sigma)**2)

        if ia_sigma_left == 0: printlog('WARNING: sigma left aligns exactly with delta chi2 array boundary')
        if ia_sigma_right == ntrial - 1: printlog('WARNING: sigma right aligns exactly with delta chi2 array boundary')

        return alist[[ia_sigma_left, ia_sigma_right]]

    def expand_chi2_a(self, x0, da, scale=2, ntrial=256, niter=100):
        chi2_0 = self._opt_fun(*x0)
        a0, l0 = x0
        alist = np.linspace(a0 - da, a0 + da, ntrial)
        # sigma = np.sqrt(2 / self.ndof)
        sigma = 1. / self.ndof
        delta_chi2 = self._opt_fun_1d(alist, l0) - chi2_0

        ia0 = np.argmin(delta_chi2)

        if np.ptp(delta_chi2[:ia0]) < sigma or np.ptp(delta_chi2[ia0:]) < sigma:
            if niter == 1:
                self.om.printlog('ERROR: expand_chi2_a failed to converge')
                return None

            return self.expand_chi2_a(x0, da * scale, scale, ntrial, niter-1)
        
        a_sigma = self._find_sigma(x0, da)

        self.om.printlog(f'expand_chi2_a success: converged on iter {niter}')
        # plt.figure(dpi=300, facecolor='w')
        # plt.plot(alist, delta_chi2)
        # plt.title(r'$\Delta \chi^2$ per DoF with ' + f'l0={l0:.3e}, DoF={self.ndof}')
        # plt.axhline(sigma)
        # plt.axvline(a_sigma[0])
        # plt.axvline(a_sigma[1])
        # plt.xlabel('a (arbitrary amplitude)')
        # plt.ylabel(r'$\Delta \chi^2$ per DoF')
        # self.om.savefig('chi2_1d.png')
        # plt.close()

        return a_sigma

    def phack_l0(self, amin, amax, lmin, lmax, na, nl):
        # outer-loop over test l0s and 
        llist = np.linspace(lmin, lmax, nl)
        alist = np.linspace(amin, amax, na)
        da = np.ptp(alist)
        dl = np.ptp(llist)

        chi2 = self._opt_fun_2d(alist, llist)

        iamin = np.argmin(chi2, axis=0)

        plt.figure(dpi=300, facecolor='w')
        plt.title(r'$\chi^2$ per DoF ' + f'(DoF={self.ndof})')
        plt.imshow(chi2[::-1,:], extent=[np.min(llist),np.max(llist),np.min(alist),np.max(alist)], aspect=dl/da)
        plt.plot(llist, alist[iamin], color='black')
        plt.colorbar()
        plt.xlabel('l')
        plt.ylabel('A - arbitrary amplitude')
        self.om.savefig(f'phack_l0_chi2_{self.freq_str}.png')
        plt.close()


        il_valid = np.arange(nl)[np.logical_and(iamin >0, iamin<na)]

        chi2_min = np.min(chi2, axis=0)
        plt.figure(dpi=300, facecolor='w')
        plt.title(r'$\chi^2$ minimum along line of best amplitude')
        plt.plot(llist, chi2_min)
        plt.xlabel('l')
        plt.ylabel('chi2 per dof at minimum amplitude')
        self.om.savefig(f'chi2_min_l{self.freq_str}.png')
        plt.close()

        dc2_amin = chi2 - np.min(chi2, axis=0)[None,:]

        # sigma = np.sqrt(2 / self.ndof)
        sigma = 1 / self.ndof

        ia_sigma_left = []
        ia_sigma_right = []
        snrs = []
        for il in range(nl):
            if il in il_valid:
                ia_sigma_left = np.argmin((dc2_amin[:iamin[il], il] - sigma)**2)
                ia_sigma_right = np.argmin((dc2_amin[iamin[il]:,il] - sigma)**2) + iamin[il]
                sigma_a = np.abs(alist[ia_sigma_left] - alist[ia_sigma_right]) / 2
                snr = alist[iamin[il]] / sigma_a
                snrs.append(snr)

        snrs = np.array(snrs)

        ilmin = np.arange(nl)[il_valid][np.argmin(snrs)]

        return snrs[ilmin], chi2[iamin[ilmin], ilmin], (alist[iamin[ilmin]], llist[ilmin]) 

    # convert a value
    def x_to_s(self, xs):
        return xs * np.array([self.ascale, 1.0])

    def get_plot_model(self, x):
        assert self.converged
        return self.ells_fit, self.ascale * self.exp_model_2d(self.best_x)[0,0]


# single frequency cltz wrapper
class CltzSFWrapper:
    # factory that takes paths
    # llims is in reduced (averaged) l space
    @classmethod
    def make(cls, om, freq, cl_tz_sig_path, cl_tz_mc_path, beam_path, fl_path,
             nave_tz, cl_tz_coeff=1.0, nave_tt=None):
        if nave_tt is None: nave_tt = nave_tz

        cl_tz_sig = np.load(cl_tz_sig_path) * cl_tz_coeff
        cl_tz_mc = np.load(cl_tz_mc_path) * cl_tz_coeff
        lmax = len(cl_tz_sig) - 1

        fl = np.load(fl_path)

        bl = parse_act_beam(beam_path)[1][:lmax + 1]

        return cls(om, freq, cl_tz_sig, cl_tz_mc, bl, fl, nave_tz, nave_tt, cl_tz_sig_path)

    @classmethod
    def make_from_yaml(cls, om, freq, paths, nave_tz, cl_tz_coeff=1.0, nave_tt=None):
        return cls.make(om, freq, paths['cl_tz_sig'], paths['cl_tz_mc'], paths['beam'], paths['fl'], nave_tz,
                        cl_tz_coeff, nave_tt)

    # constructor that expects arrays
    def __init__(self, om, freq, cl_tz_sig, cl_tz_mc, bl, fl, nave_tz, nave_tt, desc_str):
        self.om = om
        self.desc_str = desc_str

        self.bl = bl
        self.bl2 = bl**2

        self.fl = fl

        assert len(cl_tz_sig) == cl_tz_mc.shape[1]
        self.cl_tz_sig = cl_tz_sig / self.bl
        self.cl_tz_mc = cl_tz_mc / self.bl[None, :]

        self.freq_str = get_freq_str(freq)

        self.nmc = self.cl_tz_mc.shape[0]
        self.lmax = self.cl_tz_mc.shape[1] - 1
        self.nave_tz = nave_tz
        self.nave_tt = nave_tt

        self.ells = np.arange(self.lmax + 1)
        self.norm = self.ells * (self.ells + 1) / 2 / np.pi

        # make reduced arrays
        self.ells_red = downgrade_fl(self.ells, nave_tz)
        self.norm_red = downgrade_fl(self.norm, nave_tz)

        self.cl_tz_sig_red = downgrade_fl(self.cl_tz_sig, nave_tz)
        self.cl_tz_mc_red = np.empty((self.nmc, 1 + self.lmax // nave_tz))
        self.cl_tz_std = cl_tz_mc.std(axis=0)

        for itrial in range(self.nmc):
            self.cl_tz_mc_red[itrial] = downgrade_fl(self.cl_tz_mc[itrial], nave_tz)

        self.cl_tz_std_red = self.cl_tz_mc_red.std(axis=0)

    def exp_model_fit(self, x):
        return x[0] * np.exp(-(self.ells_fit / x[1] / self.lscale)**2)

    def opt_fun(self, x):
        nave_sum = 500
        chunked_sum = (self.ivar_weight * (self.exp_model_fit(x) - self.cl_tz_fit)**2).reshape(-1, nave_sum).sum(axis=-1)
        res = chunked_sum.sum()
        return res

    # dimensionalized opt_fun
    def opt_fun_2(self):
        nave_sum = 500
        chunked_sum = (self.ivar_weight * (self.exp_model_fit(x) - self.cl_tz_fit)**2).reshape(-1, nave_sum).sum(axis=-1)
        res = chunked_sum.sum()
        return res

    # def opt_grad(self, x):
    #     modfun = self.exp_model_fit(x)
    #     base = 2 * self.coeff * (modfun - self.cl_tz_fit) * self.ivar_weight
    #     res = np.array([(base * np.exp(-(self.ells_fit/x[1])**2)).sum(), (-2 * base * modfun * self.ells_fit / x[1]**3).sum()])
    #     # print(res)
    #     return res

    def eval_opt_fun_l(self, a, leval):
        n = len(leval)
        ret = np.empty(n)

        for l, i in zip(leval, range(n)):
            ret[i] = self.opt_fun([a, l])
        return ret

    def eval_of_2d(self, alist, llist):
        na = len(alist)
        nl = len(llist)
        ret = np.empty((na, nl))
        for a, ia in zip(alist, range(na)):
            for l, il in zip(llist, range(nl)):
                ret[ia, il] = self.opt_fun([a, l])
        return ret

    def _fit_exp(self, cl_tz, cl_tz_std, x0, llims, verbose=False):
        printlog = self.om.printlog

        # do best fit to signal
        lmin, lmax = llims

        cl_tz_var = cl_tz_std**2

        nells = lmax - lmin

        self.ells_fit = self.ells[lmin:lmax]

        # cl_tz_norm = 1.
        self.cl_tz_norm = 1./(np.abs(cl_tz[lmin:lmax]).mean())
        cl_tz_norm = self.cl_tz_norm
        self.cl_tz_fit = cl_tz_norm * cl_tz[lmin:lmax]

        self.ivar_weight = 1. / cl_tz_std[lmin:lmax]**2
        ivar_norm = self.ivar_weight.sum()
        self.ivar_weight = self.ivar_weight / ivar_norm
        self.ivar_norm = ivar_norm
        chi2_scale = ivar_norm / cl_tz_norm**2


    def _fit_exp_old(self, cl_tz, cl_tz_std, x0, llims, verbose=False):
        printlog = self.om.printlog

        # do best fit to signal
        lmin, lmax = llims

        cl_tz_var = cl_tz_std**2

        nells = lmax - lmin

        self.ells_fit = self.ells[lmin:lmax]

        # cl_tz_norm = 1.
        self.cl_tz_norm = 1./(np.abs(cl_tz[lmin:lmax]).mean())
        cl_tz_norm = self.cl_tz_norm
        self.cl_tz_fit = cl_tz_norm * cl_tz[lmin:lmax]

        self.ivar_weight = 1. / cl_tz_std[lmin:lmax]**2
        ivar_norm = self.ivar_weight.sum()
        self.ivar_weight = self.ivar_weight / ivar_norm
        self.ivar_norm = ivar_norm
        chi2_scale = ivar_norm / cl_tz_norm**2

        # choose a reasonable normalization for the chi2
        # coeff = 1./opt_fun(x0)

        # res = minimize(self.opt_fun, x0, jac=self.opt_grad, method='Powell')
        # res = minimize(self.opt_fun, x0, jac=self.opt_grad, method='Nelder-Mead')
        # res = minimize(self.opt_fun, x0, bounds=((None, None), (100., np.inf)))
        # res = minimize(self.opt_fun, x0, jac=self.opt_grad, bounds=((None, None), (100., np.inf)))
        # res = minimize(opt_fun, x0, method='L-BFGS-B', bounds=((None, None), (100., np.inf)), options={'eps': 0.01})
        res = minimize(self.opt_fun, x0)
        # print(x0)

        hess_inv = res.hess_inv / chi2_scale
        param_std = np.sqrt(np.diag(hess_inv)) * np.array([1/cl_tz_norm, self.lscale])

        self.resx = res.x
        x = [res.x[0] / cl_tz_norm, res.x[1] * self.lscale]

        if verbose:
            printlog('===================== OPT RESULT =====================')
            printlog(res)
            printlog('=================== END OPT RESULT ===================')
            printlog(f'param_std: {param_std[0]:.3e}, {param_std[1]:.3e}')
            printlog(f'param_snr: {x[0] / param_std[0]:.3e}, {x[1] / param_std[1]:.3e}')
            printlog(f'a: {x[0]:.3e}, l: {x[1]:.3e}')

        return x, res.success

    def fit_exp_with_error(self, llims, ntrial=1024):
        printlog = self.om.printlog
        printlog(f'Doing fit on {self.desc_str}')
        lmin, lmax = llims

        x0 = [-0.5, 5000]
        dxm = [-5, -4900]
        dxp = [5, 10000]

        nsearch = np.array([64, 64])

        nave_fit = 50

        cl_tz_mc_red_fit = np.array([get_red_fit2(this_cltz_mc, nave_fit, llims) for this_cltz_mc in self.cl_tz_mc])
        cl_tz_std_fit = cl_tz_mc_red_fit.std(axis=0)

        fw_sig = FitWrapper(self.cl_tz_mc, cl_tz_std_fit, llims, cl_tz=self.cl_tz_sig, nave_tz=nave_fit, om=self.om, freq_str=self.freq_str)

        x_sig = fw_sig.do_fit(x0, dxm, dxp, np.max(2/nsearch), nsearch, niter=3)
        print(x_sig)
        printlog(f'best chi2: {fw_sig.best_chi2:.3e}')
        printlog(f'best chi2 goodness: {fw_sig.chi2_goodness:.3e}')
        printlog(f'sig ch2: {fw_sig.sig_chi2}')
        sigma_a = fw_sig.expand_chi2_a(x_sig, np.abs(x_sig[0] / 100))
        print(x_sig, sigma_a)
        printlog(f'SNR: {x_sig[0]/(sigma_a[1] - x_sig[0]):.3e}')
        printlog(f'SNR2: {x_sig[0]/(sigma_a[0] - x_sig[0]):.3e}')
        # x_sig, success = self._fit_exp(self.cl_tz_sig, self.cl_, x0=x0, llims=llims, verbose=True)

        self.best_x = fw_sig.best_x.copy()
        self.s_sig = fw_sig.x_to_s(x_sig)
        printlog('s_sig: ' + str(self.s_sig))
        self.ells_fit = fw_sig.ells_fit.copy()
        self.best_model = fw_sig.ascale * fw_sig.exp_model(*x_sig)
        self.norm_fit = fw_sig.norm_fit

        x0 = [0., 10000]
        dxm = [-10, -9900]
        dxp = [10, 10000]

        a_samples = []
        successes = []
        fw_mc = FitWrapper(self.cl_tz_mc, cl_tz_std_fit, llims, nave_tz=nave_fit, om=self.om, freq_str=self.freq_str)
        for itrial in range(ntrial):
            printlog(f'{itrial} of ntrial')
            
            fw_mc.update_cl_tz_rand()
            x_rand = fw_mc.do_fit(x0, dxm, dxp, np.max(2/nsearch), nsearch, niter=3, suffix=f'_imc_{itrial}')
            # x_rand, success = self._fit_exp(self.cl_tz_mc[itrial], cl_tz_std, x0=[0., 100000. / self.lscale], llims=llims)

            a_samples.append(x_rand[0])
            # successes.append(success)
            successes.append(True)
        a_mc = np.array(a_samples)
        successes = np.array(successes)

        nsuccess = successes.sum()

        printlog(f'{ntrial - successes.sum()} of {ntrial} trials failed to converge')

        assert nsuccess == ntrial

        a_sig = x_sig[0]
        std_a = a_mc.std()
        sigma_a = a_sig / std_a
        printlog(f'mc snr on alpha: {sigma_a:.3e}')

    def fit_phack_l(self, llims_fit, nave_fit=50):
        cl_tz_mc_red_fit = np.array([get_red_fit2(this_cltz_mc, nave_fit, llims_fit) for this_cltz_mc in self.cl_tz_mc])
        print(cl_tz_mc_red_fit.shape)
        cl_tz_std_fit = cl_tz_mc_red_fit.std(axis=0)
        fw = FitWrapper(self.cl_tz_mc, cl_tz_std_fit, llims_fit, cl_tz=self.cl_tz_sig, nave_tz=50, om=self.om, freq_str=self.freq_str)

        print(fw.phack_l0(-2, 2, 2000, 120000, 1000, 100))


def make_inspection_plots(cltz):
    om = cltz.om
    printlog = om.printlog
    desc_str = get_fname(cltz.desc_str)

    plt.figure(dpi=300, facecolor='w')
    plt.plot(cltz.ells_red, cltz.norm_red * cltz.cl_tz_sig_red)
    om.savefig('cl_tz_raw', subdir=desc_str)


if __name__ == "__main__":
    om = OutputManager(base_path='output', title='plot-cl-tz-mf', logs=['log'], replace=True)
    printlog = om.printlog

    config_file = sys.argv[1]
    printlog('got config file ' + config_file)
    config_dict = get_yaml_dict(config_file)
    printlog('dumping config')
    for key, value in config_dict.items():
        printlog(f'{key}: {value}')
    globals().update(config_dict) # I need to stop doing this
    printlog('################## DONE ##################')

    map_weight_f090 = enmap.read_map(paths_f090['map_weight'])
    map_weight_f090_diff = enmap.read_map(paths_f090_diff['map_weight'])
    map_weight_f090_tszmask = enmap.read_map(paths_f090_tszmask['map_weight'])
    map_weight_f150 = enmap.read_map(paths_f150['map_weight'])
    map_weight_f150_tszmask = enmap.read_map(paths_f150_tszmask['map_weight'])

    # wc = 1e14
    wc = 1.

    weight_f090 = wc / map_weight_f090.mean()
    weight_f090_diff = wc / map_weight_f090_diff.mean()
    weight_f150 = wc / map_weight_f150.mean() 
    weight_f090_tszmask = wc / map_weight_f090_tszmask.mean()
    weight_f150_tszmask = wc / map_weight_f150_tszmask.mean()

    print(f'cl_tz_weights: 90: {weight_f090:.3e}, cl_tz_diff 90: {weight_f090_diff:.3e}, 150: {weight_f150:.3e}')

    cltz_f090 = CltzSFWrapper.make_from_yaml(om, 90, paths_f090, NAVE_TZ, weight_f090, NAVE_TT)
    # cltz_f090_diff = CltzSFWrapper.make_from_yaml(om, 90, paths_f090_diff, NAVE_TZ, weight_f090_diff, NAVE_TT)
    # print(cltz_f090_diff.cl_tz_sig_red)
    print(cltz_f090.cl_tz_sig_red)
    cltz_f090_vsub = CltzSFWrapper.make_from_yaml(om, 90, paths_f090_vsub, NAVE_TZ, weight_f090, NAVE_TT)
    cltz_f090_tszmask = CltzSFWrapper.make_from_yaml(om, 90, paths_f090_tszmask, NAVE_TZ, weight_f090_tszmask, NAVE_TT)
    cltz_f090_vsub_tszmask = CltzSFWrapper.make_from_yaml(om, 90, paths_f090_vsub_tszmask, NAVE_TZ, weight_f090_tszmask, NAVE_TT)
    cltz_f150 = CltzSFWrapper.make_from_yaml(om, 150, paths_f150, NAVE_TZ, weight_f150, NAVE_TT)
    cltz_f150_vsub = CltzSFWrapper.make_from_yaml(om, 150, paths_f150_vsub, NAVE_TZ, weight_f150, NAVE_TT)
    cltz_f150_tszmask = CltzSFWrapper.make_from_yaml(om, 150, paths_f150_tszmask, NAVE_TZ, weight_f150_tszmask, NAVE_TT)
    cltz_f150_vsub_tszmask = CltzSFWrapper.make_from_yaml(om, 150, paths_f150_vsub_tszmask, NAVE_TZ, weight_f150_tszmask, NAVE_TT)

    llim_fit = [2000, 7000]

    # cltzs = [cltz_f090, cltz_f090_vsub, cltz_f090_tszmask, cltz_f150, cltz_f150_vsub, cltz_f150_tszmask]

    cltz_f090.fit_exp_with_error(llims=llim_fit, ntrial=1)
    cltz_f090_vsub.fit_exp_with_error(llims=llim_fit, ntrial=1)
    cltz_f090_tszmask.fit_exp_with_error(llims=llim_fit, ntrial=1)
    cltz_f090_vsub_tszmask.fit_exp_with_error(llims=llim_fit, ntrial=1)
    # cltz_f090_diff.fit_exp_with_error(llims=llim_fit, ntrial=1)
    cltz_f150.fit_exp_with_error(llims=llim_fit, ntrial=1)
    cltz_f150_vsub.fit_exp_with_error(llims=llim_fit, ntrial=1)
    cltz_f150_tszmask.fit_exp_with_error(llims=llim_fit, ntrial=1)
    cltz_f150_vsub_tszmask.fit_exp_with_error(llims=llim_fit, ntrial=1)

    # cltz_f090.fit_phack_l(llims_fit=[2000,7000])
    # cltz_f150.fit_phack_l(llims_fit=[2000,7000])

    make_cltz_comparison(om, [cltz_f090_vsub, cltz_f150_vsub], labels=['f090 vsub', 'f150 vsub'], 
                         lmin_plot=5, title_suffix='90 vsub vs 150 vsub')
    # make_cltz_comparison(om, [cltz_f090, cltz_f090_diff], labels=['f090 vsub', 'f090-f150 diff'], 
    #                      lmin_plot=5, title_suffix='90 vs 90-150 diff', show_model=False)
    make_cltz_comparison(om, [cltz_f090, cltz_f150],  labels=['f090', 'f150'], 
                         lmin_plot=5, title_suffix='90 vs 150')
    make_cltz_comparison(om, [cltz_f090, cltz_f090_vsub],  labels=['f090', 'f090 vsub'], 
                     lmin_plot=5, title_suffix='90 vs 90 vsub')
    make_cltz_comparison(om, [cltz_f150, cltz_f150_vsub],  labels=['f150', 'f150 vsub'], 
                     lmin_plot=5, title_suffix='150 vs 150 vsub')

    # tszmask analysis
    make_cltz_comparison(om, [cltz_f090, cltz_f090_tszmask],  labels=['f090', 'f090 tszmask'], 
                     lmin_plot=5, title_suffix='90 vs 90 tszmask')
    make_cltz_comparison(om, [cltz_f090, cltz_f090_vsub_tszmask],  labels=['f090', 'f090 vsub tszmask'], 
                     lmin_plot=5, title_suffix='90 vs 90 vsub tszmask')
    make_cltz_comparison(om, [cltz_f150, cltz_f150_tszmask],  labels=['f150', 'f150 tszmask'], 
                     lmin_plot=5, title_suffix='150 vs 150 tszmask')
    make_cltz_comparison(om, [cltz_f150, cltz_f150_vsub_tszmask],  labels=['f150', 'f150 vsub tszmask'], 
                     lmin_plot=5, title_suffix='150 vs 150 vsub tszmask')

    make_cltz_comparison(om, [cltz_f150, cltz_f150_vsub, cltz_f150_tszmask, cltz_f150_vsub_tszmask], labels=['f150', 'f150 vsub', 'f150 tszmask', 'f150 vsub tszmask'],
                     lmin_plot=5, title_suffix='150 all combinations', show_model=False)
    make_cltz_comparison(om, [cltz_f090, cltz_f090_vsub, cltz_f090_tszmask, cltz_f090_vsub_tszmask], labels=['f090', 'f090 vsub', 'f090 tszmask', 'f090 vsub tszmask'],
                     lmin_plot=5, title_suffix='90 all combinations', show_model=False)
