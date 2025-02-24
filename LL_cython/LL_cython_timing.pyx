import numpy as np
import cython
from libc.math cimport cos as ccos
from libc.math cimport pow as cpow
from libc.math cimport exp as cexp
from cython.parallel import prange

import time
import numpy as np
import pandas as pd

cdef int cores = 4

def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================

@cython.boundscheck(False) 
cdef double one_energy(double[:,:] arr, int ix ,int iy, int nmax) noexcept nogil:
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
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
#=======================================================================
@cython.boundscheck(False) 
cdef float all_energy(double[:,:] arr, int nmax, int[:,:] grid) noexcept nogil:
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    cdef float enall = 0.0
    cdef int i, j, k

    for k in prange(nmax*nmax, num_threads=cores):
        i = grid[k][0]
        j = grid[k][1]
        enall += one_energy(arr,i,j,nmax)
    return enall
#=======================================================================
@cython.boundscheck(False) 
def get_order(double[:,:] arr, int nmax, int[:,:] grid):
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

    for c in prange(9, nogil=True, num_threads=cores):
        a = grid3_view[c][0]
        b = grid3_view[c][1]
        for k in range(nmax*nmax):
            i = grid[k][0]
            j = grid[k][1]
            Qab_view[a,b] += 3*lab_view[a,i,j]*lab_view[b,i,j] - delta_view[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()
#=======================================================================
@cython.boundscheck(False)
def MC_step(double[:,:] arr,float Ts, int nmax, int[:,:] grid):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.

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

    for f in prange(nmax*nmax, nogil=True, num_threads=cores):
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
#=======================================================================
def main(int nsteps, int nmax, float temp):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    #creates xy grid, 2 x nmax squared array for iterating through all x and y values
    grid = np.array(np.mgrid[0:nmax,0:nmax].reshape(2,-1).T, dtype=np.int32)
    cdef int[:,:] grid_view = grid

    # Create and initialise lattice
    lattice = initdat(nmax)
    cdef double[:,:] lattice_view = lattice
    # Plot initial frame of lattice
    # Create arrays to store energy, acceptance ratio and order parameter
    '''
    energy = np.zeros(nsteps+1,dtype=np.dtype)
    ratio = np.zeros(nsteps+1,dtype=np.dtype)
    order = np.zeros(nsteps+1,dtype=np.dtype)
    '''
    energy = np.zeros(nsteps+1, dtype=np.double)
    ratio = np.zeros(nsteps+1, dtype=np.double)
    order = np.zeros(nsteps+1, dtype=np.double)
    cdef double[::1] energy_view = energy
    cdef double[::1] ratio_view = ratio
    cdef double[::1] order_view = order
    # Set initial values in arrays
    energy_view[0] = all_energy(lattice_view,nmax, grid_view)
    ratio_view[0] = 0.5 # ideal value
    order_view[0] = get_order(lattice_view,nmax, grid_view)

    # Begin doing and timing some MC steps.
    cdef double initial = time.time()
    for it in range(1,nsteps+1):
        ratio_view[it] = MC_step(lattice_view,temp,nmax, grid_view)
        energy_view[it] = all_energy(lattice_view,nmax, grid_view)
        order_view[it] = get_order(lattice_view,nmax, grid_view)
    cdef double final = time.time()
    cdef double runtime = final-initial

    return runtime

def timing():

    lat_size = np.arange(10, 101, 10)
    iter_size = np.arange(100, 1001, 100)

    lat_time = np.zeros(10)
    iter_time = np.zeros(10)
    for i in range(10):
        lat_time[i] = main(200, lat_size[i], 0.5)
        iter_time[i] = main(iter_size[i], 50, 0.5)

    lat_dataframe = pd.DataFrame(index = lat_size, columns = [f'Cython: {cores} core'], data = lat_time)
    iter_dataframe = pd.DataFrame(index = iter_size, columns = [f'Cython: {cores} core'], data = iter_time)

    lat_dataframe.to_csv(f'./times/lat_cython_{cores}_core.csv')
    iter_dataframe.to_csv(f'./times/iter_cython_{cores}_core.csv')