from mpi4py import MPI
import numpy as np
import sys
import time

def main(program, iterations, nmax):
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    grid = np.arange(nmax*nmax).reshape(nmax, nmax)
    select = np.mgrid[0:nmax, 0:nmax].reshape(2, -1).T

    
    split_select = np.array_split(select, size)
    
    scatter_select = comm.scatter(split_select, root = 0)

    for i, j in scatter_select:
        grid[i, j] += 1

    print(grid)

    split_grid = np.array_split(grid.reshape(nmax*nmax), size)
    print(split_grid)

    gather_grid = comm.gather(split_grid[rank], root = 0)


    if rank == 0:
        print(np.concatenate(gather_grid).reshape(nmax, nmax))

if __name__ == '__main__':
    if int(len(sys.argv)) == 3:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        main(PROGNAME, ITERATIONS, SIZE)
        
    else:
        print("Usage: python {} <ITERATIONS> <SIZE>".format(sys.argv[0]))