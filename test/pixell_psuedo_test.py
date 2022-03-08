from pixell.curvedsky import alm2map, alm2cl, map2alm, rand_alm, rand_map
from pixell.enmap import read_map
import numpy as np
import matplotlib.pyplot as plt

map_path = '/home/aroman/data/act/act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'
FSKY = 0.65

def map2cl(t_map, lmax):
    return alm2cl(map2alm(t_map, lmax=lmax))

if __name__ == "__main__":
    print('importing map')
    imap = read_map(map_path)
    imap_t = imap[0]
    print('done')


    lmax = 6000
    ells = np.arange(lmax + 1)
    cl = np.zeros(lmax + 1)
    cl[0] = 0.
    cl[1:] = 1./(np.arange(1,lmax + 1, 1)**3)

    test_map = rand_map(imap_t.shape, imap_t.wcs, cl, lmax=lmax)

    cl_measured = map2cl(test_map, lmax)
    fl_measured = cl_measured / cl



    norm = ells * (1 + ells) / 2 / np.pi

    plt.figure(dpi=300)
    plt.title('rand cl alm/map comparison')
    plt.plot(cl * norm, label='base cl', linewidth=1, alpha=0.5)
    plt.plot(cl_measured * norm, label='cl2alm2cl', linewidth=1, alpha=0.5)
    # plt.plot(cl_map * norm / FSKY, label='cl2alm2map2cl', linewidth=1, alpha=0.5)
    # plt.plot(cl_randmap * norm / FSKY, label='cl2map2cl', linewidth=1, alpha=0.5)
    plt.yscale('log')
    plt.legend()
    plt.xlabel(u'$l$')
    plt.ylabel(u'$\Delta_l$')
    plt.savefig('cl_pseudo_compare.png')
    plt.close()

    plt.figure(dpi=300)
    plt.plot(fl_measured)
    plt.yscale('log')
    plt.legend()
    plt.xlabel(u'$l$')
    plt.ylabel('fl')
    plt.savefig('fl_pseudo_compare.png')
    plt.close()