import time
import numpy as np
from mpi4py import MPI
import pandas as pd

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

def one_energy(arr,ix,iy,nmax):
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
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
#=======================================================================
def all_energy(arr,nmax, grid):
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


    #print(f'Process: all_energy, Core number is: {rank}, lattice loc is: {grid[0]}')

    enall = 0.0
    for i, j in grid:
        enall += one_energy(arr,i,j,nmax)

    return enall
#=======================================================================
def get_order(arr,nmax, grid):
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


    #print(f'Process: get_order, Core number is: {rank}, lattice loc is: {grid[0]}')


    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i, j in grid:
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]

    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)

    return eigenvalues.max()
#=======================================================================
def MC_step(arr,Ts,nmax, grid):
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
    scale=0.1+Ts
    accept = 0

    #randomness has to be removed to allow it to be multi cored
    #otherwise two cores could work on the same node
    #its also very difficult to combine
    '''
    xran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    yran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    '''
    aran = np.random.normal(scale=scale, size=(nmax,nmax))
    for i, j in grid:
        ang = aran[i,j]
        en0 = one_energy(arr,i,j,nmax)
        arr[i,j] += ang
        en1 = one_energy(arr,i,j,nmax)
        if en1<=en0:
            accept += 1
        else:
        # Now apply the Monte Carlo test - compare
        # exp( -(E_new - E_old) / T* ) >= rand(0,1)
            boltz = np.exp( -(en1 - en0) / Ts )

            if boltz >= np.random.uniform(0.0,1.0):
                accept += 1
            else:
                arr[i,j] -= ang

    #gather all of the local accepts into one
    return accept/(nmax*nmax)

#=======================================================================
def main(nsteps, nmax, temp, comm):
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
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        initial = MPI.Wtime()
    
    #creates xy grid, 2 x nmax squared array for iterating through all x and y values
    grid = np.mgrid[0:nmax,0:nmax].reshape(2,-1).T

    #splits the grid by the number of cores
    split_grid = np.array_split(grid, size)

    #scatters local grid to each of the cores
    local_grid = comm.scatter(split_grid, root=0)

    # Create and initialise lattice
    lattice = initdat(nmax)

    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1)
    ratio = np.zeros(nsteps+1)
    order = np.zeros(nsteps+1)
    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax, local_grid)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax, local_grid)

    # Begin doing and timing some MC steps.
    if rank == 0:
        initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax, local_grid)
        energy[it] = all_energy(lattice, nmax, local_grid)
        order[it] = get_order(lattice, nmax, local_grid)
        #gather all of the local arrs into one
        #done by splitting the array by the section they have worked on
        comm.Barrier()
        split_arr = np.array_split(lattice.reshape(nmax*nmax), size)
        gathered_arr = comm.gather(split_arr[rank], root=0)
        if rank == 0:
            lattice = np.concatenate(gathered_arr).reshape(nmax, nmax)
        lattice = comm.bcast(lattice, root = 0)

    comm.Barrier()
    energy = comm.reduce(energy, op=MPI.SUM, root=0)
    order = comm.reduce(order, op=MPI.SUM, root=0)
    ratio = comm.reduce(ratio, op=MPI.SUM, root=0)

    if rank == 0:
        final = time.time()
        runtime = final-initial
        return runtime


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

lat_size = np.arange(10, 101, 10)
iter_size = np.arange(100, 1001, 100)

lat_time = np.zeros(10)
iter_time = np.zeros(10)
for i in range(10):
    lat_time[i] = main(20, lat_size[i], 0.5, comm)
    iter_time[i] = main(iter_size[i], 20, 0.5, comm)

if rank == 0:
    lat_dataframe = pd.DataFrame(index = lat_size, columns = [f'MPI MCstep: {size} core'], data = lat_time)
    iter_dataframe = pd.DataFrame(index = iter_size, columns = [f'MPI MCstep: {size} core'], data = iter_time)

    lat_dataframe.to_csv(f'./times/lat_mpi_MCstep_{size}_core.csv')
    iter_dataframe.to_csv(f'./times/iter_mpi_MCstep_{size}_core.csv')