import numpy as np
import matplotlib.pyplot as plt
import camb
import kszpipe
from kszpipe.Box import Box

# # basically a camb wrapper
# class Cosmology:
#     def __init__(self):
#         pars = camb.CAMBparams()
#         pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
#         pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
#         res = camb.get_results(pars)

#         self.pars = pars
#         self.res = res

def unit_r(pos_list):
    return pos_list / np.sqrt(pos_list[0]**2 + pos_list[1]**2 + pos_list[2]**2)[None,:]


# An wrapper for a bare array to transparently map between array space and 
# (padded) comoving space (Mpc)
# TODO: implement
class PaddedArray:
    pass


def plot_2d_function(ar_2d, xlabel, ylabel, path_base, var_label, lims):
    plt.figure(dpi=300)
    plt.title(var_label + f' {xlabel}{ylabel} cross section')
    plt.imshow(ar_2d[::-1,:], extent=lims)
    cbar = plt.colorbar()
    cbar.set_label(var_label + ' (summed over axis)')
    plt.xlabel(ylabel + ' (Mpc)')
    plt.ylabel(xlabel + ' (Mpc)')
    plt.savefig(path_base + '_' + xlabel + ylabel + '.png')
    plt.close()


def ind_reshape(ind, shape):
    ndim = len(shape)
    ret = np.empty(ndim, dtype=int)
    for i in range(ndim - 1, -1, -1):
        n = shape[i]
        ret[i] = ind % n
        ind = ind // n
    return ret


# unit of comoving distance is Mpc
class Padded3DPipe:
    # Requires a reference galaxy pipeline to define the geometry
    def __init__(self, gal_pipe, cosmology):
        self.gal_pipe = gal_pipe
        # self.cosmo = Cosmology()
        self.box = None
        self.cosmology = cosmology
        self.init_real = False
        self.init_d0 = False
    
    def _sky_to_comoving(self, ra, dec, z):
        r_co = self.cosmology.chi_z(z, check=True)
        return r_co * np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])        

    def _init_gal(self):
        pos = self.gal_pipe.gal_pos
        zs = self.gal_pipe.zs
        self.gal_pos_3d = self._sky_to_comoving(pos[1], pos[0], zs)
        self.gal_unit_r = unit_r(self.gal_pos_3d)

    def _init_buffers(self):
        assert self.box is not None
        nx, ny, nz = self.dims
        self.a_j = np.zeros((3,nx,ny,nz))
        self.a_k = None # not initialized yet
        self.ngal = np.zeros((nx,ny,nz))

    def init_from_box(self, box):
        self._init_gal()
        self.box = box
        self.dims = box.npix
        self.boxsize = self.box.boxsize
        self.pos = self.box.pos
        self.ind0 = np.floor(self.pos_to_ind(np.array([0.,0.,0.]), check=False)).astype(int)
        assert len(self.dims) == 3

        coords = [np.arange(ni) for ni in self.dims]
        self.indices = np.array(np.meshgrid(*coords, indexing='ij'))
        self.pixel_centers = self.pos[:, None, None, None] + self.box.pixsize[:, None, None, None] * (self.indices + 0.5)
        
        self.chi_grid = np.sqrt(self.pixel_centers[0]**2 + self.pixel_centers[1]**2 + self.pixel_centers[2]**2) 
        
        # debug the apparent missing zero chi issue
        chi_min = np.min(self.chi_grid)
        min_ind = ind_reshape(np.argmin(self.chi_grid), self.chi_grid.shape)

        print(f'chi_min: {chi_min:3e}, zero_ind {self.ind0}, min_ind {min_ind}')

        # WARN: check fails!!!
        self.z_grid = self.cosmology.z_chi(self.chi_grid, check=False)
        assert np.all(self.chi_grid.shape == self.z_grid.shape)
        assert np.all(self.chi_grid.shape == self.dims)

        self.a_grid = 1./(1 + self.z_grid)

        self.pos_limits = np.array((self.pos, self.pos + self.boxsize)).T

        self._init_buffers()

        print(f'array shape {self.dims}')
        print(f'padded box dimensions {self.box.boxsize} (Mpc)')
        print(f'real space volume {self.box.box_volume:.3e} (Mpc^3)')

        # faHD in real space
        fx = self.cosmology.frsd_z(self.z_grid, check=False)
        hx = self.cosmology.H_z(self.z_grid, check=False)
        dx = self.cosmology.D_z(self.z_grid, check=False)
        self.fahd = self.a_grid * fx * hx * dx

        # harmonic space constants
        k_grid = np.ones(self.box.fshape, dtype=np.complex)
        ki = np.empty([3,] + list(self.box.fshape), dtype=np.complex)
        ki[0] = self.box.get_k_component(0, one_dimensional=True)[:, None, None] * k_grid
        ki[1] = self.box.get_k_component(1, one_dimensional=True)[None, :, None] * k_grid
        ki[2] = self.box.get_k_component(2, one_dimensional=True)[None, None, :] * k_grid
        self.ki = ki
        self.k_pre = 1j * self.ki / self.box.get_k2()[None, ...]

        # WARN: NAN!
        self.k_pre = np.nan_to_num(self.k_pre)

        self.init_real = True

    def init_d0_k(self, d0_k, compute_real=False):
        assert self.init_real
        assert self.box.is_fourier_space(d0_k)
        self.d0_k = d0_k

        if compute_real: self.d0_x = self.box.fft(self.d0_k)
        else: self.d0_x = None

        self.init_d0 = True

    # def init(self, nx, ny, nz, dist_pad):
    #     assert dist_pad >= 0
    #     self._init_gal()

    #     self.dist_pad = dist_pad
    #     self.dims = np.array((nx, ny, nz))
    #     self.pos_min = np.min(self.gal_pos_3d, axis=1)
    #     self.pos_max = np.max(self.gal_pos_3d, axis=1)
    #     self.pos_min_pad = self.pos_min - self.dist_pad
    #     self.pos_max_pad = self.pos_max + self.dist_pad
    #     self.pos_corner = self.pos_min_pad
    #     # self.lim = np.array(self.pos_min, self.pos_max).T
    #     # self.lim_pad = np.array(self.pos_min_pad, self.pos_max_pad).T
    #     self.delta_pos = self.pos_max - self.pos_min
    #     self.delta_pos_pad = self.delta_pos + 2 * self.dist_pad

    #     self.volume_real = np.prod(self.delta_pos)
    #     self.volume_real_pad = np.prod(self.delta_pos_pad)

    #     print(f'real space volume (unpadded/padded) {self.volume_real:.3e} {self.volume_real_pad:.3e}')

    #     self.edge_length = max(self.delta_pos)
    #     self.edge_length_pad = max(self.delta_pos_pad)

    #     self.lim = np.array((self.pos_min, self.pos_min + self.edge_length)).T
    #     self.lim_pad = np.array((self.pos_min_pad, self.pos_min_pad + self.edge_length_pad)).T

    #     # TODO: switch to padded array
    #     self.ar_3d = np.zeros((3,nx,ny,nz))
    #     self.ngal = np.zeros((nx,ny,nz))

    #     print(f'unpadded box dimensions {self.delta_pos}')
    #     print(f'padded box dimensions {self.delta_pos_pad}')

    #     # make a box to manipulate the data
    #     npix = np.array((nx, ny, nz), dtype=int)
    #     pos = self.pos_corner
    #     pixsize = self.delta_pos_pad / npix
    #     boxsize = self.delta_pos_pad

    #     # challenge the consistency check by specifying pixel size and pos
    #     self.box = Box(npix, pixsize=pixsize, boxsize=boxsize, pos=pos)

    #     self.init_real = True

    # add galaxies by comoving coordinates, with an optional CMB temp list
    def add_galaxies(self, *, pos_list=None, t_list=None, wipe=False):
        assert self.init_real

        if pos_list is None:
            pos_list = self.gal_pos_3d
            unit_r_list = self.gal_unit_r
        else:
            unit_r_list = unit_r(pos_list)

        # TODO: check pos_list t_list shape compatibility


        ngal = pos_list.shape[1]
        if t_list is None:
            t_list = np.ones(ngal)

        inds = (self.pos_to_ind(pos_list) + 0.5).astype(int)

        if wipe:
            self._init_buffers()

        # ngal is included for diagnostic purposes
        self.ngal[inds[0], inds[1], inds[2]] += 1
        self.a_j[:, inds[0], inds[1], inds[2]] += t_list[None, :] * unit_r_list
        self.a_j *= -self.fahd[None, :]

        self.a_k = np.empty([3,] + list(self.box.fshape), dtype=np.complex)
        for i in range(3):
            self.a_k[i] = self.box.fft(self.a_j[i])

    # presumes (3, ngal) or (3,) shape
    def pos_to_ind(self, pos3d, *, check=True):
        if check: assert self.init_real
        if len(pos3d.shape) == 1:
            return self.dims * (pos3d - self.pos) / self.box.boxsize
        else:
            return self.dims[:,None] * (pos3d - self.pos[:,None]) / self.box.boxsize[:, None]

    def do_harmonic_sum(self):
        assert self.init_real
        assert self.init_d0

        sum_i = np.empty((3,),dtype=float)
        for i in range(3):
            sum_i[i] = self.box.dot(self.d0_k, self.k_pre[i] * self.a_k[i])
        a_ksz_nonnorm = sum_i.sum()

        return a_ksz_nonnorm

    def plot_3d(self, ar, outpath_stem, *, mode='sum', var_label='', **kwargs):
        assert self.init_real
        assert np.all(ar.shape == self.dims)
        assert mode == 'sum' or mode == 'slice'

        x0, x1 = self.pos_limits[0]
        y0, y1 = self.pos_limits[1]
        z0, z1 = self.pos_limits[2]

        if mode == 'sum':
            plot_2d_function(ar.sum(axis=0), 'y', 'z', outpath_stem, var_label=var_label, lims=[z0, z1, y0, y1])
            plot_2d_function(ar.sum(axis=1), 'x', 'z', outpath_stem, var_label=var_label, lims=[z0, z1, x0, x1])
            plot_2d_function(ar.sum(axis=2), 'x', 'y', outpath_stem, var_label=var_label, lims=[y0, y1, x0, x1])
        elif mode == 'slice':
            if 'slice_inds' in kwargs:
                ix, iy, iz = kwargs['slice_inds']
            else:
                ix, iy, iz = np.array(ar.shape) // 2
            print(f'plotting slice inds {ix, iy, iz}')
            plot_2d_function(ar[ix,:,:], 'y', 'z', outpath_stem, var_label=var_label, lims=[z0, z1, y0, y1])
            plot_2d_function(ar[:,iy,:], 'x', 'z', outpath_stem, var_label=var_label, lims=[z0, z1, x0, x1])
            plot_2d_function(ar[:,:,iz], 'x', 'y', outpath_stem, var_label=var_label, lims=[y0, y1, x0, x1])
        
# def get_bounding_box(, x_corr=None):
# def analyze_galaxy_survey(gal_pipe):