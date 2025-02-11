from mpi4py import MPI
import numpy as np
import sys
import time

def one_energy(arr, ix, iy, nmax):

        en = 0.0
        ixp = (ix+1)%nmax # These are the coordinates
        ixm = (ix-1)%nmax # of the neighbours
        iyp = (iy+1)%nmax # with wraparound
        iym = (iy-1)%nmax #

        return np.sum([arr[ix, iy], arr[ix, iyp], arr[ix, iym], arr[ixp, iy], arr[ixm, iy]])/1000


def main(program, iterations, nmax):
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        initial = MPI.Wtime()
    
    grid = np.mgrid[0:nmax,0:nmax].reshape(2,-1).T
    #lattice = np.random.randint(0, 10, size = [nmax, nmax])
    lattice = np.ones([nmax, nmax])
    chunk_size = nmax*nmax//size
    start = rank*chunk_size
    end = (rank+1) * chunk_size if rank != size-1 else nmax*nmax
    #print(start, end)

    '''
    for i in range(iterations):
        for x, y in grid[start:end]:
            if one_energy(lattice, x, y, nmax)%2 != 0:
                lattice[x, y] = (lattice[x, y] + 1)%10
        comm.Barrier()
        lattice.reshape(1,-1)[start:end] = comm.bcast(lattice.reshape(1,-1)[start:end], root = rank)
    local_lattice = lattice.reshape(1,-1)[start:end]
'''
    for i in range(iterations):
        for x, y in grid[start:end]:
            lattice[x, y] += one_energy(lattice, x, y, nmax)
            
        comm.Barrier()
        local_lattice = np.zeros(nmax * nmax)
        local_lattice[start:end] = lattice.reshape(1,-1)[0, start:end]
        new_lattice = comm.reduce(local_lattice.reshape(nmax, nmax), op=MPI.SUM, root=0)
        if rank==0:
            #print(new_lattice)
            lattice = comm.bcast(new_lattice, root=0)
        comm.Barrier()
        #print(lattice, rank, i)
        
        '''
        if rank == 0:
            lattice = np.concatenate(gathered).reshape(nmax,nmax)
            #print(lattice)
            '''
    

    local_lattice = lattice.reshape(1,-1)[0, start:end]

    gathered_results = comm.gather(local_lattice, root=0)
    
    if rank == 0:
        final_lattice = np.concatenate(gathered_results).reshape(nmax,nmax)
        print(final_lattice)
        final = MPI.Wtime()
        print(f'Time Taken: {final - initial}')
        

if __name__ == '__main__':
    if int(len(sys.argv)) == 3:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        main(PROGNAME, ITERATIONS, SIZE)
        
    else:
        print("Usage: python {} <ITERATIONS> <SIZE>".format(sys.argv[0]))
