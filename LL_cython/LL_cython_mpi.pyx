"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""
#cython: language_level=3

import numpy as np
import cython
from libc.math cimport cos as ccos
from libc.math cimport pow as cpow
from libc.math cimport exp as cexp
cimport numpy as cnp
cnp.import_array()
from cython.parallel import prange
from mpi4py import MPI

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#=======================================================================
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
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
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
cdef float all_energy(double[:,:] arr, int nmax, int[:,:] grid, comm, int grid_len):
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
    cdef int rank = comm.Get_rank()

    with nogil:
        for k in range(grid_len):
            i = grid[k][0]
            j = grid[k][1]
            enall += one_energy(arr,i,j,nmax)
    enall2 = comm.reduce(enall, op=MPI.SUM, root = 0)
    if rank == 0:
        return enall2
    else:
        return 0.0
#=======================================================================
@cython.boundscheck(False) 
def get_order(double[:,:] arr, int nmax, int[:,:] grid, comm, int grid_len):
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
    #assign rank
    cdef int rank = comm.Get_rank()

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

    with nogil:
        for c in range(9):
            a = grid3_view[c][0]
            b = grid3_view[c][1]
            for k in range(grid_len):
                i = grid[k][0]
                j = grid[k][1]
                Qab_view[a,b] += 3*lab_view[a,i,j]*lab_view[b,i,j] - delta_view[a,b]
    
    Qab = comm.reduce(Qab, op=MPI.SUM, root = 0)
    if rank == 0:
        Qab = Qab/(2*nmax*nmax)
    
        eigenvalues,eigenvectors = np.linalg.eig(Qab)
        return eigenvalues.max()
    else:
        return 0.0
#=======================================================================
@cython.boundscheck(False)
def MC_step(double[:,:] arr,float Ts, int nmax, int[:,:] grid, comm, int grid_len):
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

    #assign rank
    cdef int rank = comm.Get_rank()
    cdef int size = comm.Get_size()
    #randomness has to be removed to allow it to be multi cored
    #otherwise two cores could work on the same node
    #its also very difficult to combine

    cdef float scale=0.1+Ts
    
    #scatters a section of the random number array to each core
    aran = np.random.normal(scale=scale, size=(nmax*nmax))
    split_aran = np.array_split(aran, size)
    local_aran = comm.scatter(split_aran, root=0)
    cdef double[::1] aran_view = local_aran

    #same for the uniform random number array
    uniran = np.random.uniform(0.0, 1.0, size = nmax*nmax)
    split_uniran = np.array_split(uniran, size)
    local_uniran = comm.scatter(split_uniran, root=0)
    cdef double[::1] uniran_view = local_uniran

    cdef int ix = 0, iy = 0, f
    cdef double ang, en0, en1, boltz
    cdef float accept = 0.0

    with nogil:
        for f in range(grid_len):
            accept += 0.0
            ix = grid[f][0]
            iy = grid[f][1]
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

    #gather all of the local accepts into one
    comm.Barrier()
    total_accept = comm.reduce(accept, op=MPI.SUM, root = 0)
    if rank == 0:
      return total_accept/(nmax*nmax)
    else:
        return 0.0
#=======================================================================
def main(str program, int nsteps, int nmax, float temp, int pflag):
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
    #initialise MPI
    comm = MPI.COMM_WORLD
    cdef int size = comm.Get_size()
    cdef int rank = comm.Get_rank()

    #creates xy grid, 2 x nmax squared array for iterating through all x and y values
    grid = np.array(np.mgrid[0:nmax,0:nmax].reshape(2,-1).T, dtype=np.int32)

    #splits the grid by the number of cores
    split_grid = np.array_split(grid, size)

    #scatters local grid to each of the cores
    local_grid = comm.scatter(split_grid, root=0)
    cdef int grid_len = len(local_grid)
    cdef int[:,:] local_grid_view = local_grid

    # Create and initialise lattice
    lattice = initdat(nmax)
    cdef double[:,:] lattice_view = lattice

    # Plot initial frame of lattice
    if rank == 0:
        plotdat(lattice,pflag,nmax)
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
    energy_view[0] = all_energy(lattice_view,nmax, local_grid_view, comm, grid_len)
    ratio_view[0] = 0.5 # ideal value
    order_view[0] = get_order(lattice_view,nmax, local_grid_view, comm, grid_len)

    # Begin doing and timing some MC steps.
    if rank == 0:
        initial = time.time()
    for it in range(1,nsteps+1):
        ratio_view[it] = MC_step(lattice_view,temp,nmax, local_grid_view, comm, grid_len)

        energy_view[it] = all_energy(lattice_view,nmax, local_grid_view, comm, grid_len)

        order_view[it] = get_order(lattice_view,nmax, local_grid_view, comm, grid_len)
        #gather all of the local arrs into one
        #done by splitting the array by the section they have worked on
        comm.Barrier()
        split_arr = np.array_split(lattice.reshape(nmax*nmax), size)
        gathered_arr = comm.gather(split_arr[rank], root=0)
        if rank == 0:
            lattice = np.concatenate(gathered_arr).reshape(nmax, nmax)
            lattice_view = lattice

    if rank == 0:
        final = time.time()

        runtime = final-initial
        
        # Final outputs
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
        # Plot final frame of lattice and generate output file
        savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
        plotdat(lattice,pflag,nmax)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
