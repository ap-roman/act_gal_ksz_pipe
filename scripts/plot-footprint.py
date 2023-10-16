from fnl_pipe.util import OutputManager

from pixell import enmap, enplot
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.utils.data import download_file
from astropy.io import fits  # We use fits to open the actual data file

from astropy.utils import data
data.conf.remote_timeout = 60

# from spectral_cube import SpectralCube

# from astroquery.esasky import ESASky
# from astroquery.utils import TableList
from astropy.wcs import WCS
# from reproject import reproject_interp

def renorm(map_t):
    mi = np.min(map_t)
    ma = np.max(map_t)
    return (map_t - mi) / (ma - mi)

data_base = '/home/aroman/data/'

sdss_north = renorm(enmap.read_map(data_base + 'sdss_footprint/pixellized_sdss_north_completeness.fits'))
sdss_south = renorm(enmap.read_map(data_base + 'sdss_footprint/pixellized_sdss_south_completeness.fits'))
desils_north = renorm(enmap.read_map(data_base + 'vr_source/desils/desi_north_mask.fits'))
desils_south = renorm(enmap.read_map(data_base + 'vr_source/desils/desi_south_mask.fits'))
weight_tot = renorm(enmap.read_map(data_base + 'vr_source/desils/w_tot_kendrick_data_f090.fits'))

labels = ['sdss_north', 'sdss_south', 'desils_north', 'desils_south', 'wfkp_090', 'intersect']

out_path = 'footprints.png'

colors = ['red', 'orange', 'blue', 'royalblue', 'green']
# colors = ['FF6B6B', 'FF6B6B', 'FF6B6B', 'FF6B6B', 'FF6B6B']

def make_footprint_plot(masks, colors, om, alpha=0.2):
    fig = plt.figure(figsize = (18,12))
    ax = fig.add_subplot(111, projection = masks[0].wcs)
    for mask, color in zip(masks, colors):
        # fig = enplot.plot(mask, color=color)
        # enplot.write(out_path, fig)


        # Display the moment map image
        # im = ax.imshow(mask, cmap = 'viridis', alpha=alpha)
        im = ax.imshow(mask, alpha=alpha)
    om.savefig(out_path)

def do_single_plot(mask, label, om):
    fig = plt.figure(figsize = (18,12))
    ax = fig.add_subplot(111, projection = mask.wcs)
    im = ax.imshow(mask)
    fig.colorbar(im)
    om.savefig(label + '.png')

def get_mask_footprint(mask):
    return (mask * mask.pixsizemap()).sum() * (180 / np.pi)**2

def union(a, b):
    # print(a.shape)
    maxab = np.maximum(a,b)
    # print(maxab.shape)
    # print(maxab)
    return maxab

if __name__ == "__main__":
    om = OutputManager(base_path='output', title='plot-footprint', logs=['log',], replace=True,)

    act_mask = (weight_tot > 0.).astype(float)

    intersect = union(sdss_north, sdss_south) * union(desils_north, desils_south) * act_mask

    print(intersect)

    masks = [sdss_north, sdss_south, desils_north, desils_south, act_mask, intersect]

    for mask, label in zip(masks, labels):
        print(f'{label} footprint: {get_mask_footprint(mask):.3e} deg^2')
        do_single_plot(mask, label, om)

    make_footprint_plot(masks, colors, om)