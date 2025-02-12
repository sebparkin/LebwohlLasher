cimport cython

import math
import numpy as np
from libc.math cimport sqrt
from libc.math cimport cos as ccos
from libc.math cimport pow as cpow
from libc.math cimport exp as cexp
cimport numpy as cnp
from cython.parallel import prange

@cython.boundscheck(False)
cdef double one_energy(double[:,:] arr, int ix, int iy, int nmax) noexcept nogil:
    cdef float en = 0.0
    cdef int ixp = (ix+1)%nmax # These are the coordinates
    cdef int ixm = (ix-1)%nmax # of the neighbours
    cdef int iyp = (iy+1)%nmax # with wraparound
    cdef int iym = (iy-1)%nmax #
    #
    # Add together the 4 neighbour contributions
    # to the energy
    #
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*cpow(ccos(ang),2))
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*cpow(ccos(ang),2))
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*cpow(ccos(ang),2))
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*cpow(ccos(ang),2))
    return en
@cython.boundscheck(False)    
cdef double all_energy(double[:,:] arr, int nmax, int[:,:] grid):
    cdef float enall = 0.0
    cdef int size = len(grid)
    cdef int i, j, k

    for k in prange(size, nogil=True):
        i = grid[k][0]
        j = grid[k][1]
        enall += one_energy(arr,i,j,nmax)


        '''
        for i in prange(nmax, nogil=True, num_threads=1):
            for j in range(nmax):
                enall += one_energy(arr,i,j,nmax)

        '''
    return enall

cpdef get_order(double[:,:] arr, int nmax, int[:,:] grid):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    grid3 = np.array(np.mgrid[0:3,0:3].reshape(2,-1).T, dtype=np.int32)
    cdef int[:,:] grid3_view = grid3

    Qab = np.zeros((3,3), dtype=np.double)
    delta = np.eye(3,3, dtype=np.double)
    cdef double[:,:] Qab_view = Qab
    cdef double[:,:] delta_view = delta

    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr)), dtype = np.double).reshape(3,nmax,nmax)
    cdef double[:,:,:] lab_view = lab
    cdef int a, b, c, i, j, k

    for c in prange(9, nogil=True, num_threads=1):
        a = grid3_view[c][0]
        b = grid3_view[c][1]
        for k in range(nmax*nmax):
            i = grid[k][0]
            j = grid[k][1]
            Qab_view[a,b] += 3*lab_view[a,i,j]*lab_view[b,i,j] - delta_view[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()


cdef MC_step(double[:,:] arr,float Ts, int nmax, int[:,:] grid):

    cdef float scale=0.1+Ts
    

    xran = np.random.randint(0,high=nmax, size=(nmax*nmax), dtype = np.int32)
    yran = np.random.randint(0,high=nmax, size=(nmax*nmax), dtype = np.int32)
    aran = np.random.normal(scale=scale, size=(nmax*nmax))
    cdef int[::1] xran_view = xran
    cdef int[::1] yran_view = yran
    cdef double[::1] aran_view = aran

    uniran = np.random.uniform(0.0, 1.0, size = nmax*nmax)
    cdef double[::1] uniran_view = uniran

    cdef int ix = 0, iy = 0, f, i, j
    cdef double ang, en0, en1, boltz
    cdef float accept = 0.0

    for f in prange(nmax*nmax, nogil=True, num_threads=1):
        accept += 0.0
        ix = xran_view[f]
        iy = yran_view[f]
        ang = aran_view[f]
        en0 = one_energy(arr,ix,iy,nmax)
        arr[ix,iy] += ang
        en1 = one_energy(arr,ix,iy,nmax)
        if en1<=en0:
            accept += 1.0
        else:
            accept += 0.0
        # Now apply the Monte Carlo test - compare
        # exp( -(E_new - E_old) / T* ) >= rand(0,1)
            boltz = cexp( -(en1 - en0) / Ts )

            if boltz >= uniran_view[f]:
                accept += 1.0
            else:
                arr[ix,iy] -= ang
                accept += 0.0
    return accept/(nmax*nmax)


def main():
    cdef int nmax = 5

    array = np.arange(nmax*nmax, dtype = np.double).reshape(nmax, nmax)
    grid = np.array(np.mgrid[0:nmax,0:nmax].reshape(2,-1).T, dtype=np.int32)

    print(array)

    cdef double[:,:] arr_view = array
    cdef int[:,:] grid_view = grid

    value = all_energy(arr_view, nmax, grid_view)
    value2 = get_order(arr_view, nmax, grid_view)
    value3 = MC_step(arr_view, 0.5, nmax, grid_view)

    print(value, value2, value3)
    print(array)
    return None
