
import time
import numpy as np
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
    enall = np.sum(one_energy(arr, grid[:,0], grid[:,1], nmax))
    return enall
#=======================================================================
def get_order(arr,nmax):
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
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            Qab[a,b] = np.sum(3*lab[a,:,:]*lab[b,:,:] - delta[a,b])
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
    grid (array) = array containing all indexes of the lattice
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
    aran = np.random.normal(scale=scale, size=(nmax,nmax))


    #to work with numpy array vectoring all angles cannot be changed at once,
    #as energy depends on the adjacent crystals.
    #instead it is changed in a checkerboard pattern, so no adjacent crystal
    #is ever changed at the same time.
    
    #creates a checkerboard of 1s and 0s
    checkerboard = np.indices([nmax, nmax]).sum(axis=0) % 2

    for i in range(2):
        
        arr_copy = arr.copy()

        en0 = one_energy(arr, grid[:,0], grid[:,1], nmax)
        #only adds the random angle to half the lattice
        arr[checkerboard == i] += aran[checkerboard == i]
        en1 = one_energy(arr, grid[:,0], grid[:,1], nmax)
       

        #creates new energies for the checkerboard
        en0_checker = en0.reshape(nmax, nmax)[checkerboard == i]
        en1_checker = en1.reshape(nmax, nmax)[checkerboard == i]
        #adds one for each energy below the previous energy
        accept += np.sum(en1_checker <= en0_checker)
        
        #calculates boltz and compares it to a random number
        boltz = np.exp(-(en1_checker[en1_checker>en0_checker] - en0_checker[en1_checker>en0_checker]) / Ts)
        boltz_rand = np.random.uniform(0, 1.0, size=np.shape(boltz))
        accept += np.sum(boltz >= boltz_rand)

        #creates the index for values that need to be changed back, goes back three array changes
        index = np.where(checkerboard.reshape(nmax*nmax) == i)[0][np.where(en1_checker>en0_checker)[0][np.where(boltz<boltz_rand)[0]]]
    
        #changes values back, and changes all values that arent part of the half checkerboard
        arr.reshape(nmax**2)[index] -= aran.reshape(nmax**2)[index]
        arr[checkerboard == (i+1)%2] = arr_copy[checkerboard == (i+1)%2]

    return accept/(nmax*nmax)
#=======================================================================
def main(nsteps, nmax, temp):
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
    # Create and initialise lattice
    lattice = initdat(nmax)
    # Plot initial frame of lattice
    # Create arrays to store energy, acceptance ratio and order parameter
    '''
    energy = np.zeros(nsteps+1,dtype=np.dtype)
    ratio = np.zeros(nsteps+1,dtype=np.dtype)
    order = np.zeros(nsteps+1,dtype=np.dtype)
    '''
    #create a grid of all possible lattice indexes
    grid = np.mgrid[0:nmax,0:nmax].reshape(2,-1).T

    energy = np.zeros(nsteps+1)
    ratio = np.zeros(nsteps+1)
    order = np.zeros(nsteps+1)
    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax, grid)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax)

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax, grid)
        energy[it] = all_energy(lattice,nmax,grid)
        order[it] = get_order(lattice,nmax)
    final = time.time()
    runtime = final-initial

    return runtime

lat_size = np.arange(10, 101, 10)
iter_size = np.arange(100, 1001, 100)

lat_time = np.zeros(10)
iter_time = np.zeros(10)
for i in range(10):
    lat_time[i] = main(200, lat_size[i], 0.5)
    iter_time[i] = main(iter_size[i], 50, 0.5)

lat_dataframe = pd.DataFrame(index = lat_size, columns = ['NumPy'], data = lat_time)
iter_dataframe = pd.DataFrame(index = iter_size, columns = ['NumPy'], data = iter_time)

lat_dataframe.to_csv('./times/lat_numpy.csv')
iter_dataframe.to_csv('./times/iter_numpy.csv')