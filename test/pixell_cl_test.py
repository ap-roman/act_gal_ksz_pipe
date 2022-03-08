from pixell.curvedsky import alm2map, alm2cl, map2alm, rand_alm, rand_map
from pixell.enmap import read_map
from pixell import enmap
import numpy as np
import matplotlib.pyplot as plt

map_path = '/home/aroman/data/act/act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'
ivar_path = '/home/aroman/data/act/act_planck_s08_s19_cmb_f150_daynight_srcfree_ivar.fits'
FSKY = 0.63

def map2cl(t_map, lmax):
    return alm2cl(map2alm(t_map, lmax=lmax))[:lmax + 1]

def fsky(t_map):
    corners = t_map.corners()
    return np.abs(((corners[1,0] % 2*np.pi) - (corners[1,1] % 2*np.pi)) * (np.cos(corners[0,0]) - np.cos(corners[0,1]))) / 4 / np.pi

def get_noise_map(std_t):
    return enmap.ndmap(np.random.normal(size=std_t.shape) * std_t, std_t.wcs)

def get_fl(cl_in, map_fkp, lmax):
    alm_fl = rand_alm(cl_in)
    map_fl = enmap.empty(map_fkp.shape, map_fkp.wcs)
    map_fl = alm2map(alm_fl, map_fl)
    cl_pseudo = map2cl(map_fkp * map_fl, lmax=lmax)

    return cl_pseudo/cl_in


if __name__ == "__main__":
    print('importing map')
    imap_t = read_map(map_path)[0]
    ivar_t = read_map(ivar_path)[0]

    std_t = np.sqrt(np.nan_to_num(1./ivar_t))

    map_fkp = read_map('../data/map_fkp.fits')
    print('done')

    lmax = 6000
    lmax_comp = lmax
    ells = np.arange(lmax_comp + 1)
    ells[0] = 1.
    cl_gen = 1./ells**2
    cl_gen[0] = 0.
    ells[0] = 0.

    norm = ells * (1 + ells) / 2 / np.pi

    f_sky = fsky(imap_t)

    print(f'fsky: {fsky(imap_t):.3f}')

    cl_tot_map = map2cl(imap_t * map_fkp, lmax=lmax_comp)
    cl_noise_interim = map2cl(get_noise_map(std_t), lmax=lmax_comp)
    # cl_noise_map = map2cl(rand_map(imap_t.shape, imap_t.wcs, cl_noise_interim) * map_fkp, lmax=lmax)
    cl_noise_map = map2cl(get_noise_map(std_t) * map_fkp, lmax=lmax_comp)
    # map_noise = rand_map(imap_t.shape, imap_t.wcs, cl_noise_interim / FSKY)
    # map_noise = rand_map(imap_t.shape, imap_t.wcs, cl_noise_interim / FSKY)
    # cl_noise_map = map2cl(map_noise * map_fkp, lmax=lmax)

    # reconstruct 

    cl_cmb_map = cl_tot_map - cl_noise_map

    cl_f = np.ones(lmax_comp + 1)
    cl_f2 = cl_gen

    fl = get_fl(cl_f, map_fkp, lmax_comp)
    fl_2 = get_fl(cl_f2, map_fkp, lmax_comp)

    plt.figure(dpi=300)
    plt.title('transfer function fl')
    plt.plot(ells[:lmax], fl[:lmax], label='fl flat')
    plt.plot(ells[:lmax], fl_2[:lmax], label='fl l^-2')
    plt.legend()
    plt.ylabel('amplitude')
    plt.xlabel(u'$l$')
    plt.savefig('fl_use.png')
    plt.close()

    cl_cmb = cl_cmb_map / fl
    cl_noise = cl_noise_map / fl
    cl_tot = cl_tot_map / fl

    tot_map = rand_map(imap_t.shape, imap_t.wcs, cl_tot)
    cl_sim_check = map2cl(tot_map * map_fkp, lmax=lmax_comp)

    plt.figure()
    plt.plot(ells[:lmax], (norm * cl_sim_check)[:lmax], label='simulated')
    plt.plot(ells[:lmax], (norm * cl_tot_map)[:lmax], label='act map')
    plt.legend()
    plt.ylabel(u'$\Delta$')
    plt.xlabel(u'$l$')
    plt.savefig('cl_fl_compare')
    plt.close()

    tot_rand_map = enmap.empty(imap_t.shape, imap_t.wcs)
    tot_rand_map = alm2map(rand_alm(cl_cmb), tot_rand_map) + get_noise_map(std_t)
    # tot_rand_map = alm2map(rand_alm(cl_cmb), tot_rand_map)
    # tot_rand_map = rand_map(imap_t.shape, imap_t.wcs, cl_cmb)
    cl_rand_map = map2cl(tot_rand_map * map_fkp, lmax=lmax_comp)

    cl_noise_deficit_map = cl_tot_map - cl_rand_map

    plt.figure(dpi=300)
    plt.title('comparison of act cl components')
    # plt.yscale('log')
    lmin_plot = 1000
    plt.plot(ells[lmin_plot:lmax], (norm * cl_tot_map)[lmin_plot:lmax], label='map total')
    plt.plot(ells[lmin_plot:lmax], (norm * cl_cmb_map)[lmin_plot:lmax], label='map cmb')
    plt.plot(ells[lmin_plot:lmax], (norm * cl_rand_map)[lmin_plot:lmax], label='map sim')
    plt.plot(ells[lmin_plot:lmax], (norm * cl_noise_map)[lmin_plot:lmax], label='map noise')
    plt.plot(ells[lmin_plot:lmax], (norm * cl_noise_deficit_map)[lmin_plot:lmax], label='sim-noise delta')
    plt.legend()
    # plt.ylim([0.01,1000])
    plt.ylabel(u'$\Delta_l$')
    plt.xlabel(u'$l$')
    plt.savefig('cl_act_compare.png')
    plt.close()

    plt.figure(dpi=300)
    plt.title('cl sim / cl actual ratio (both pseudo)')
    plt.plot(ells[50:], (cl_tot_map / cl_rand_map)[50:])
    plt.ylabel('cl sim / cl actual')
    plt.xlabel(u'$l$')
    plt.savefig('cl_ratio.png')
    plt.close()


    randalm = rand_alm(cl_gen, lmax=lmax)
    randmap = rand_map(imap_t.shape, imap_t.wcs, cl_gen, lmax=lmax)
    randmap = rand_map(imap_t.shape, imap_t.wcs, cl_gen, lmax=lmax)

    cl_alm = alm2cl(randalm)
    cl_map = map2cl(alm2map(randalm, imap_t.copy()), lmax=lmax)
    cl_psuedo = cl_map.copy()
    cl_randmap = map2cl(randmap, lmax=lmax)

    cl = [cl_gen, cl_alm, cl_map, cl_randmap]
    for c in cl:
        c[0] = 0.1 * 2 * 2 * np.pi

    # fl by two similar methods
    fl_alm = cl_alm / cl_gen
    fl_map2cl = cl_map / cl_gen


    plt.figure(dpi=300)
    plt.title('rand cl alm/map comparison')
    plt.plot(cl_gen * norm, label='base cl', linewidth=1, alpha=0.5)
    plt.plot(cl_alm * norm, label='cl2alm2cl', linewidth=1, alpha=0.5)
    plt.plot(cl_map * norm, label='cl2alm2map2cl', linewidth=1, alpha=0.5)
    plt.plot(cl_randmap * norm, label='cl2map2cl', linewidth=1, alpha=0.5)
    plt.yscale('log')
    plt.legend()
    plt.xlabel(u'$l$')
    plt.ylabel(u'$\Delta_l$')
    plt.savefig('cl_rand_compare.png')
    plt.close()

    plt.figure(dpi=300)
    plt.title('psuedo xfer function comparison')
    plt.plot(fl_alm, label='fl via alm', linewidth=1, alpha=0.5)
    plt.plot(fl_map2cl, label='fl via map2cl', linewidth=1, alpha=0.5)
    plt.yscale('log')
    plt.legend()
    plt.xlabel(u'$l$')
    plt.ylabel('fl')
    plt.savefig('fl_pseudo_compare.png')
    plt.close()

    # compare pixel averages generated from psuedo/normal cl
    var_t_gen = np.var(rand_map(imap_t.shape, imap_t.wcs, cl_gen, lmax=lmax))
    var_t_psuedo = np.var(rand_map(imap_t.shape, imap_t.wcs, cl_psuedo / FSKY, lmax=lmax))

    print(f'var(T), var(T_pseudo) {var_t_gen:.3e}  {var_t_psuedo:.3e}')