import numpy as np

import time

from fnl_pipe.pipe import ActPipe, GalPipe
from fnl_pipe.realspace import act_unit_r, Padded3DPipe
import kszpipe
from pixell import enmap

import matplotlib.pyplot as plt


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
# NSIMS = 2046

NSIMS = 2048

# some script-specific paths
t2_path = act_path + f'meta/t2_{NSIMS}.fits'
mangle_path = '/home/aroman/data/mangle/sdss_dr12v5_mangle.fits'


class test_f:
    def __init__(self):
        pass

    def __call__(self, phi, theta):
        pass


class test_g:
    def __init__(self):
        pass

    def __call__(self, pos):
        return np.ones(pos.shape)


# outward face normals for our bounding cube
# cube_normals = np.array([[0.,-1,0], [1,0,0], [0,1,0], [-1,0,0], [0,0,1], [0,0,-1]]).T
cube_normals = np.array([[0.,1,0], [1,0,0], [0,1,0], [1,0,0], [0,0,1], [0,0,1]]).T

# a specialized class to handle an integral of the form
# \int d^3x f(theta) * g(x), where
#     f(theta) is a quickly varying function on the unit sphere (e.g. CMB data)
#     g(x) is a slowly varying function of position
#     the integral is evaluated over each coarse 3d pixel
class CMBxRealIntegral:
    def __init__(self, box):
        self.box = box

    # compute the 3d position list, along with a weight array
    #     cmb_pos: the 2d angular positions of the CMB pixels
    #     n_r_upscale: a radial resolution upscaling factor that boosts radial resolution
    #                  beyond the native transverse resolution at the box corner
    def make_grid(self, ref_map, nr_upscale=1):
        # make sure the rays terminate at the box boundary
        self.unit_r = act_unit_r(*ref_map.posmap())
        cube_intersect = np.empty(6)
        pos = self.box.pos
        size = self.box.boxsize
        print(pos, size)
        cube_intersect[0] = pos[1] # y min
        cube_intersect[1] = pos[0] + size[0] # x max
        cube_intersect[2] = pos[1] + size[1] # y max
        cube_intersect[3] = pos[0] # x min
        cube_intersect[4] = pos[2] + size[2] # z max
        cube_intersect[5] = pos[2] # z min
        print(cube_intersect)

        cube_dots = np.sum(cube_normals[...,None,None] * self.unit_r[:,None,...], axis=0)

        ray_test = np.array([[1,0,0], [0,1,0], [0,0,1]]).T
        cube_test = np.sum(cube_normals[...,None] * ray_test[:,None,:], axis=0)
        test_dists = cube_intersect[:,None]/cube_test
        print(test_dists.T)

        max_r = 2 * np.max(size)
        print(max_r)
        raw_dists = cube_intersect[:,None,None]/cube_dots
        # raw_dists[raw_dists < 0] = max_r
        raw_dists = np.nan_to_num(raw_dists, posinf=max_r, neginf=max_r)
        rmin = np.min(raw_dists, axis=0)
        print(rmin)
        print(rmin.shape)

        plt.figure(dpi=300)
        plt.imshow(rmin)
        plt.colorbar()
        plt.savefig('plots/integral_grid.png')
        # print(cube_dots.shape)
        # sgn = np.sign(cube_dots)
        # sgn[sgn!=1] = 0.
        # signcheck = sgn.sum(axis=0)
        # print((signcheck != 3.).sum())
        # print(signcheck)
        # print(self.unit_r.shape)


    # accept the 3d positions, 
    def eval(self, fs, gs):


        return ret



# Make the noise weighting for our Y(x), v(x) estimator. This is necessary for the
# weiner filter we intend to apply
if __name__ == "__main__":
    act_pipe = ActPipe(map_path, ivar_path, beam_path, cl_ksz_path, cl_cmb_path,    
                          planck_enmask_path,
                          custom_l_weight=None, diag_plots=True, lmax=12000)

    act_pipe.import_data()
    act_pipe.update_metadata(r_fkp=R_FKP, r_lwidth=R_LWIDTH, gal_path=catalog_path)
    act_pipe.compute_pixel_weight()
    act_pipe.import_fl_nl(fl_path)
    act_pipe.compute_sim_spectra(make_plots=True)
    act_pipe.compute_l_weight()

    gal_pipe = GalPipe(catalog_path, act_pipe, diag_plots=True)
    gal_pipe.import_data()
    gal_pipe.make_vr_list()

    cosmology = kszpipe.io_utils.read_pickle(kszpipe_cosmo_path)
    box = kszpipe.io_utils.read_pickle(kszpipe_box_path)

    real_pipe = Padded3DPipe(gal_pipe, cosmology)
    real_pipe.init_from_box(box)


    t0 = time.time()
    t2_map = enmap.read_map(t2_path)
    mangle_map = enmap.read_map(mangle_path)

    t2int = CMBxRealIntegral(box)
    t2int.make_grid(t2_map)

