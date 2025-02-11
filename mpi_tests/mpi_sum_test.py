from mpi4py import MPI
import numpy as np
import sys
import time

def main(program, iterations, nmax):
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        initial = MPI.Wtime()
    
    grid = np.mgrid[0:nmax,0:nmax].reshape(2,-1).T
    lattice = grid.T[1].reshape(nmax, nmax)
    #lattice = np.ones([nmax, nmax])
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
    local_sum = 0
    lattice_split = np.array_split(lattice.reshape(nmax * nmax), size) if rank == 0 else None
    local_lattice = comm.scatter(lattice_split, root=0)
    for i in range(iterations):
        for x, y in grid[start:end]:
            local_sum += lattice[x, y]
            
        comm.Barrier()
        total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
        
        '''
        if rank == 0:
            lattice = np.concatenate(gathered).reshape(nmax,nmax)
            #print(lattice)
            '''
    
    
    if rank == 0:
        print(lattice)
        print(total_sum)
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
