import numpy as np
import matplotlib.pyplot as plt
import camb
import kszpipe
from kszpipe.Box import Box


def unit_r(pos_list):
    return pos_list / np.sqrt(pos_list[0]**2 + pos_list[1]**2 + pos_list[2]**2)[None, :]


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


def ensure_list(intlike):
    return [intlike,]


def concat_shape(shape0, shape1):
    ar = [shape0, shape1]
    llist = []

    for ilike in ar:
        if isinstance(ilike, int): ilike = ensure_list(ilike)
        llist.append(list(ilike))

    return tuple(llist[0] + llist[1])


def step_roll(ar, step=1):    
    assert len(ar.shape) == 2
    n = ar.shape[0]

    if n==1: return ar
    else:
        assert n % 2 == 0
        n2 = n // 2
        return np.concatenate((step_roll(ar[:n2], step), 
                        step_roll(np.roll(ar[n2:], n2 * step, axis=1), step)))


def diag_inds(ar, offset=0, axis=0):
    assert len(ar.shape) == 2
    assert axis == 0 or axis == 1

    ind_oset = np.array((0,0),dtype=int)
    ind_oset[axis] = offset

    return np.diag_indices(ar) + ind_oset

# A very limited single-purpose class to implement and verify each linear step 
# of the 3D and 2D computations and their adjoints
class PipeAdjoint:
    def __init__(self, pipe):
        self.pipe = pipe
        self.init = False

    def init_all(self, box, d0_path, init_gal=False):
        self.pipe.init_from_box(box, init_gal=init_gal)
        self.box = box
        self.pipe.init_d0_k(box.read_grid(d0_path, fourier=True))
        self.init = True

    def interp(self, v, gal_pos=None):
        if gal_pos is None:
            gal_pos = self.pipe.gal_pos_3d

        return self.box.interpolate(v, gal_pos.T, periodic=False)

    def extirp(self, w):
        gal_pos = self.pipe.gal_pos_3d.T
        grid = np.zeros(self.pipe.box.rshape)
        self.box.extirpolate(grid=grid, coords=gal_pos, weights=w, periodic=False)
        return grid

    # maps 3 vec over gal space to scalar over gal space
    def three_dot(self, v, r=None):
        if r is None:
            r = self.pipe.gal_unit_r

        assert v.shape == concat_shape(3, self.pipe.gal_pipe.ngal)
        assert r.shape == v.shape
        return (v * r).sum(axis=0)

    def three_dot_adj(self, w, r=None):
        if r is None:
            r = self.pipe.gal_unit_r

        assert w.shape == (self.pipe.gal_pipe.ngal,)
        assert r.shape == concat_shape(3, self.pipe.gal_pipe.ngal)

        ret = np.repeat(w,3).reshape(concat_shape(self.pipe.gal_pipe.ngal, 3)).T
        return ret * r

    # B maps the three-velocity to interpolated vr
    # three-vel component -> interp -> dot with rhat
    def B(self, v):
        assert v.shape == concat_shape(3, self.box.rshape)
        gal_pos = self.pipe.gal_pos_3d
        ngal = self.pipe.gal_pipe.ngal
        unit_r = self.pipe.gal_unit_r

        v_interp = np.empty((3, ngal))
        for j in range(3):
            v_interp[j] = self.interp(v[j], gal_pos=gal_pos)

        return self.three_dot(v_interp)

    # transpose dot -> extirp -> v
    def B_adj(self, w):
        unit_r = self.pipe.gal_unit_r
        ngal = self.pipe.gal_pipe.ngal

        assert w.shape == (ngal,)

        wadj = self.three_dot_adj(w)
        
        ret = np.empty(concat_shape(3, self.pipe.box.rshape))
        for j in range(3):
            ret[j] = self.extirp(wadj[j])
        return ret

    # A maps the fourier grid to a three-vector field in real space, applying
    # the inverse del operator
    # i.e. A maps d0(k) -> fahD * \del_i\del^{-2}d0(x)
    # A: F(I)^3 \to F(R)^3  
    def A(self, v, l):
        assert self.init # necessary to  ensure the presence of certain arrays
        assert self.box.is_fourier_space(v)


        k2 = self.box.get_k2()
        zero_mask = (k2 == 0)
        k2[zero_mask] = 1.

        ar_k = -1j * self.pipe.ki[l] * v / k2
        ar_k[np.broadcast_to(zero_mask, zero_mask.shape)] = 0.
        
        return -self.box.fft(ar_k) * self.pipe.fahd

    # the adjoint of A
    # A^{T}: R^3 \to I^3 
    def A_adj(self, w, l):
        k2 = self.box.get_k2()
        zero_mask = (k2 == 0)
        k2[zero_mask] = 1.

        ar = 1j * self.box.fft(-w * self.pipe.fahd) * self.pipe.ki[l] / k2
        ar[np.broadcast_to(zero_mask, zero_mask.shape)] = 0.

        return ar

    # A simpler version of A (returns the inv del of d0; easy code check since 
    # the laplacian is self-adjoint)
    def Aprime(self, v):
        assert self.init # necessary to  ensure the presence of certain arrays
        assert self.box.is_fourier_space(v)

        ar_k = - v / self.box.get_k2()
        ar_k = np.nan_to_num(ar_k)

        return self.pipe.fahd * self.box.fft(ar_k)

    def Aprime_adj(self, w):
        assert self.box.is_real_space(w)

        v = self.box.fft(self.pipe.fahd * w)
        v = - v / self.box.get_k2()
        v = np.nan_to_num(v)

        return v


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

        # TODO: separate adjoint debug structure?
        self.init_vr_grid = False
        self.init_vr_list = False
    
    def _sky_to_comoving(self, *, ra, dec, z):
        r_co = self.cosmology.chi_z(z, check=True)
        return r_co * np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])        

    def _init_gal(self):
        pos = self.gal_pipe.gal_pos
        zs = self.gal_pipe.zs
        self.gal_pos_3d = self._sky_to_comoving(ra=pos[1], dec=pos[0], z=zs)
        self.gal_unit_r = unit_r(self.gal_pos_3d)

    def _init_buffers(self):
        assert self.box is not None
        nx, ny, nz = self.dims
        self.a_j = np.zeros((3,nx,ny,nz))
        self.a_k = None # not initialized yet
        self.ngal = np.zeros((nx,ny,nz))

    def init_from_box(self, box, init_gal=True):
        if init_gal: self._init_gal()

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

        chi_min = np.min(self.chi_grid)
        min_ind = ind_reshape(np.argmin(self.chi_grid), self.chi_grid.shape)

        print(f'chi_min: {chi_min:3e}, zero_ind {self.ind0}, min_ind {min_ind}')

        # WARN: check fails!!!
        # TODO: address failed checks
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
        print(fx.shape, hx.shape, dx.shape, self.fahd.shape)

        # harmonic space constants
        k_grid = np.ones(self.box.fshape, dtype=np.complex)
        ki = np.empty(concat_shape(3, self.box.fshape), dtype=np.complex)
        ki[0] = self.box.get_k_component(0, one_dimensional=True)[:, None, None] * k_grid
        ki[1] = self.box.get_k_component(1, one_dimensional=True)[None, :, None] * k_grid
        ki[2] = self.box.get_k_component(2, one_dimensional=True)[None, None, :] * k_grid

        # for iax, axis in zip(range(3), ['x', 'y', 'z'])
        plt.figure(dpi=300)
        plt.imshow(np.abs(ki[0, :, :, 0]))
        plt.colorbar()
        plt.savefig('plots/kx.png')

        plt.figure(dpi=300)
        plt.imshow(np.abs(ki[1, :, :, 0]))
        plt.colorbar()
        plt.savefig('plots/ky.png')

        plt.figure(dpi=300)
        plt.imshow(np.abs(ki[2, 0, :, :]))
        plt.colorbar()
        plt.savefig('plots/kz.png')


        self.ki = ki

        k2 = self.box.get_k2()
        zero_mask = (k2 == 0)
        k2[zero_mask] = 1.
        self.k_pre = 1j * self.ki / k2[None, ...] # WARN: should be negative??
        self.k_pre[np.broadcast_to(zero_mask, concat_shape(3, zero_mask.shape))] = 0. # avoid div by 0, k=0 signal is missing anyway
        # self.k_pre = np.nan_to_num(self.k_pre)

        self.init_real = True

    def init_d0_k(self, d0_k, compute_real=False):
        assert self.init_real
        assert self.box.is_fourier_space(d0_k)
        self.d0_k = d0_k

        if compute_real: self.d0_x = self.box.fft(self.d0_k)
        else: self.d0_x = None

        self.init_d0 = True

    def make_vr_grid(self):
        assert self.init_real
        assert self.init_d0

        del_inv_d0_k = -self.d0_k / self.box.get_k2()
        del_inv_d0_k = np.nan_to_num(del_inv_d0_k)
        di_d2_d0_k = 1j * self.ki * del_inv_d0_k[None,...]

        di_d2_d0 = np.empty(concat_shape(3, self.box.rshape))
        for i in range(3):
            di_d2_d0[i] = self.box.fft(di_d2_d0_k[i])

        self.vi_grid = -self.fahd[None, ...] * di_d2_d0

        self.init_vr_grid = True

    def make_vr_list(self, pos_list=None):
        assert self.init_real
        assert self.init_vr_grid

        if pos_list is None:
            pos_list = self.gal_pos_3d
            unit_r_list = self.gal_unit_r
        else:
            unit_r_list = unit_r(pos_list)

        self.vi_list = np.empty(unit_r_list.shape)

        for i in range(3):
            self.vi_list[i] = self.box.interpolate(self.vi_grid[i], self.gal_pos_3d.T,
                                                   periodic=True)
        self.vr_list = np.sum(self.vi_list * unit_r_list, axis=0)

        self.init_vr_list = True

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
            print('WARN: no t_list provided, using all ones')

        inds = (self.pos_to_ind(pos_list) + 0.5).astype(int)

        if wipe:
            self._init_buffers()

        # ngal is included for diagnostic purposes
        self.ngal[inds[0], inds[1], inds[2]] += 1
        fahd_interp = self.box.interpolate(self.fahd, inds.T, periodic=False)
        self.a_j[:, inds[0], inds[1], inds[2]] += t_list[None, :] * unit_r_list * fahd_interp[None, :]
        self.a_j *= -self.fahd[None, ...] # WARN: why was this commented?

        self.a_k = np.empty([3,] + list(self.box.fshape), dtype=np.complex)
        for i in range(3):
            self.a_k[i] = self.box.fft(self.a_j[i])

    # presumes (3, ngal) or (3,) shape
    def pos_to_ind(self, pos3d, *, check=True):
        if check: assert self.init_real
        if len(pos3d.shape) == 1:
            return self.dims * (pos3d - self.pos) / self.box.boxsize
        else:
            return self.dims[:, None] * (pos3d - self.pos[:, None]) / self.box.boxsize[:, None]

    def do_harmonic_sum(self):
        assert self.init_real
        assert self.init_d0

        sum_i = np.empty((3,),dtype=float)
        for i in range(3):
            sum_i[i] = self.box.dot(self.d0_k, self.k_pre[i] * self.a_k[i], normalize=True)
        a_ksz_nonnorm = sum_i.sum()

        return a_ksz_nonnorm

    def do_harmonic_sum_adj(self, t_hp, pa):
        assert self.init_real
        assert self.init_d0

        interim_x = pa.B_adj(t_hp)

        vg_k = np.zeros(self.box.fshape, dtype=complex)

        # TODO: do a trial where this sum is done incorrectly
        # expect suppressed SNR
        for l in range(3):
            vg_k += pa.A_adj(interim_x[l], l)

        a_ksz_nonnorm = self.box.dot(self.d0_k, vg_k, normalize=True)
        # return self.d0_k

        return a_ksz_nonnorm, vg_k

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
