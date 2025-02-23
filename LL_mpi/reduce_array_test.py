#test to see if reduction can be done at the end of the program for all_energy and get_order

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nmax = 20
iters = 10

lattice = np.ones([nmax, nmax])
grid = np.mgrid[0:nmax,0:nmax].reshape(2,-1).T
local_grid = comm.scatter(np.array_split(grid, size), root = 0)


total = np.zeros(iters)

def calc_sum(lattice, grid):
    total = 0
    for x, y in grid:
        total += lattice[x, y] 
    return total

for i in range(iters):
    total[i] += calc_sum(lattice, local_grid)

total = comm.reduce(total, op=MPI.SUM, root=0)

if rank == 0:
    print(total)