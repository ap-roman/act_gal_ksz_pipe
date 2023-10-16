from fnl_pipe.util import OutputManager, get_fname, get_yaml_dict, fequal, ChunkedMaskedReader
from fnl_pipe.cmb import ACTPipe, ACTMetadata
from fnl_pipe.gal_cmb import CMBxGalPipe
from fnl_pipe.galaxy import DESILSCat, AndCut, NullCut, LRGNorthCut, LRGSouthCut, ZerrCut
from fnl_pipe.catalog import get_files

from pixell import enplot, enmap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import itertools
import sys

import mpi4py


def get_centers(edges):
    return 0.5 * (edges[1:] + edges[:-1])


# angles are in radians
def make_gal_survey_polar_plot(decs, ras, zs, *, nras=150, nzs=20, dec_lims=None, weights=None,
                               theta_min=0, theta_max=180):
    ngal = len(ras)

    if weights is None:
        weights = np.ones(ngal)

    if dec_lims is None:
        dec_lims = (min(decs), max(decs))

    gal_mask = (decs >= dec_lims[0]) * (decs <= dec_lims[1])

    decs = decs[gal_mask]
    ras = ras[gal_mask]
    zs = zs[gal_mask]
    weights = weights[gal_mask]

    sample = np.array((ras, zs)).T

    H_count, count_edges = np.histogramdd(sample, bins=(nras, nzs))
    H, edges = np.histogramdd(sample, bins=(nras, nzs), weights=weights)

    edges = np.array(edges)
    # print(H.shape, H_count.shape, edges.shape)

    hzero = H_count == 0
    H_count[hzero] = 1

    H = H / H_count

    H[hzero] = 0

    # print(f'ncell nan: {(H == np.nan).sum()}')
    # print(f'ncell zero: {(H == 0.).sum()}')
    # print(f'ncell count zero: {(H_count == 0.).sum()} of {np.prod(H_count.shape)}')

    # assert fequal(np.array(count_edges), np.array(edges))

    X, Y = np.meshgrid(edges[0], edges[1])

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, polar=True)
    ax.title.set_text(r'DESI-LS $v_r$ $\delta=0$ Slice')
    # ax.pcolormesh(get_centers(edges[0]), get_centers(edges[1]), H)
    ax.pcolormesh(X.T, Y.T, H, cmap=mpl.colormaps['RdBu'])
    ax.grid(False)
    ax.set_xticklabels([])

    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position-15 ),ax.get_rmax()/2.,r'$z$',
            rotation=label_position,ha='center',va='center')
    # ax.set_thetamin(0)
    # ax.set_thetamax(90)


# def make_enmap_sky_plot(plot_map):


def make_gal_plots(gal_pipe, mask_dict, output_manager):
    ras = gal_pipe.ras
    decs = gal_pipe.decs
    zs = gal_pipe.zs
    vrs = gal_pipe.vrs

    make_gal_survey_polar_plot(decs, ras, zs, weights=vrs, dec_lims=[-np.pi/180, np.pi/180])
    output_manager.savefig(f'galaxy_polar_plot.png', mode='matplotlib')
    plt.close()

    plt.figure(dpi=300)
    plt.title('DESI-LS Sample Redshift Distribution')
    plt.hist(zs, bins=64)
    plt.xlabel(r'Redshift $z$')
    plt.ylabel('Galaxy Count')
    output_manager.savefig(f'galaxy_zs.png', mode='matplotlib')
    plt.close()

    plt.figure(dpi=300)
    plt.title('DESI-LS Sample RA Distribution')
    plt.hist(ras * 180 / np.pi, bins=64)
    plt.xlabel(r'Right Ascension (degrees)')
    plt.ylabel('Galaxy Count')
    output_manager.savefig(f'galaxy_ras', mode='matplotlib')
    plt.close()

    plt.figure(dpi=300)
    plt.title('DESI-LS Sample Declination')
    plt.hist(decs * 180 / np.pi, bins=64)
    plt.xlabel(r'Declination $\delta$ (degrees)')
    plt.ylabel('Galaxy Count')
    output_manager.savefig(f'galaxy_decs.png', mode='matplotlib')
    plt.close()

    for key, val in mask_dict.items():
        fig = enplot.plot(enmap.downgrade(val, 16), ticks=15, colorbar=True)
        output_manager.savefig(f'{key}', mode='pixell', fig=fig)


# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# node_name = MPI.Get_processor_name()

# Goals: get analytic SNR, MC snr, SNR per 100,000 objects
# Generate C_l^{TZ}
# Inspect statistics on sim alphas
# "v-shuffle" maps should have very little signal
# 
# show HP-filtered maps
#   feature size < 5.4 arcmin ~ 2-5 pixels.
if __name__ == "__main__":
    om = OutputManager(base_path='output', title='make-paper', logs=['log',], replace=True)
    printlog = om.printlog

    config_file = sys.argv[1]
    printlog('got config file ' + config_file)
    config_dict = get_yaml_dict(config_file)
    local_dict = locals()
    printlog('dumping config')
    for key, value in config_dict.items():
        printlog(f'{key}: {value}')
        local_dict[key] = value
    printlog('################## DONE ##################')


    assert NTRIAL_MC % NFILE_MC == 0

    mc_lists = [mc_list_base + f'_rank_{ifile}.h5' for ifile in range(NFILE_MC)]

    printlog = om.printlog

    act_md = ACTMetadata(r_fkp=1.56, r_lwidth=0.62)

    act_pipe_150 = ACTPipe(map_path, ivar_path, beam_path, planck_enmap_path,
                           om, freq=150, metadata=act_md, plots=True)
    act_pipe_150.import_data()
    act_pipe_150.init_fkp()
    act_pipe_150.init_lweight()

    ref_map = act_pipe_150.map_t
    printlog(f'importing galaxy mask {get_fname(gal_mask_path)}')
    gal_mask = enmap.read_map(gal_mask_path)
    desi_mask = enmap.read_map(desi_mask_path)
    sdss_mask = enmap.read_map(sdss_mask_path)
    mask_dict = {'interesect_mask': gal_mask, 'desi_mask': desi_mask, 'sdss_mask': sdss_mask}

    desils_cat = DESILSCat(cat_north=desils_v3_north, cat_south=desils_v3_south)

    zerr_cut = ZerrCut(zerr_max)
    if do_cut:
        north_cut = AndCut([LRGNorthCut(), zerr_cut])
        south_cut = AndCut([LRGSouthCut(), zerr_cut])
    else:
        north_cut = zerr_cut
        south_cut = zerr_cut

    gal_pipe = desils_cat.get_subcat([north_cut, south_cut], ref_map, vr_width)
    gal_pipe.import_data()
    gal_pipe.make_vr_list()
    row_mask = gal_pipe.cut_inbounds_mask

    cross_pipe = CMBxGalPipe(act_pipe_150, gal_pipe, gal_mask, output_manager=om)

    printlog('importing nl')
    cross_pipe.import_nl(nl_path)
    printlog('importing fl')
    cross_pipe.import_fl(fl_path)

    # make survey coverage/volume plots

    # make 2D polar plot of CMB and galaxy distributoin
    # decs = gal_pipe.decs
    # ras = gal_pipe.ras
    # zs = gal_pipe.zs
    # vrs = gal_pipe.vrs

    # make_gal_survey_polar_plot(decs, ras, zs, weights=vrs)
    # om.savefig(f'galaxy_polar_plot_desils.png', mode='matplotlib')
    # plt.close()

    make_gal_plots(gal_pipe, mask_dict, om)


    # make t_hp diagnostic plots

    t_hp = cross_pipe.make_t_hp_nomask()

    fig1 = enplot.plot(enmap.downgrade(t_hp, 16), ticks=15, colorbar=True)
    om.savefig(f't_hp_nogalmask_{act_pipe_150.freq}', mode='pixell', fig=fig1)

    fig2 = enplot.plot(enmap.downgrade(t_hp * gal_mask, 16), ticks=15, colorbar=True)
    om.savefig(f't_hp_galmask_{act_pipe_150.freq}', mode='pixell', fig=fig2)

    # Do velocity reshuffle test and estimate
    alphas_reshuffle = np.empty(NTRIAL_RESHUFFLE)
    for itrial in range(NTRIAL_RESHUFFLE):
        cross_pipe.compute_estimator(v_shuffle=True)
        alphas_reshuffle[itrial] = cross_pipe.a_ksz_unnorm

    plt.figure(dpi=300)
    plt.title(f'Velocity reshuffle sample (N={NTRIAL_RESHUFFLE})')
    plt.hist(alphas_reshuffle, bins=32)
    plt.xlabel(r'$\alpha$ ($\sigma$)')
    om.savefig('alphas_reshuffle.png')
    plt.close()

    # Get bootstrap SNR

    ntrial_per_file = NTRIAL_MC // NFILE_MC
    alphas_mc = []
    # alphas_bs = []
    for mc_list in mc_lists:
        br = ChunkedMaskedReader(mc_list, 1024 * 32, row_mask, bufname='t_mc')

        ret = cross_pipe.compute_estimator(ntrial_mc=ntrial_per_file, buffered_reader=br)
        alphas_mc = np.concatenate((alphas_mc, cross_pipe.alphas))

    a_ksz_mc = cross_pipe.a_ksz_unnorm / np.std(alphas_mc)
    a_bs = ret['a_ksz_bootstrap_2']
    printlog(f'kSZ estimator: {a_ksz_mc:.3e} vs bootstrap {a_bs:.3e}')


    plt.figure(dpi=300)
    plt.title(f'Estimator Sample')
    plt.hist(alphas_mc, bins=32)
    plt.xlabel(r'$\alpha$ ($\sigma$)')
    om.savefig('alphas_mc.png')
    plt.close()