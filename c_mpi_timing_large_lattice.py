import numpy as np
import pandas as pd
from LL_cython import LL_cython_mpi_timing
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

lat_sizes = np.arange(50, 501, 50)
times = []

for i, l in enumerate(lat_sizes):
    times.append(LL_cython_mpi_timing.main(200, l, 0.5, comm))

comm.Barrier()
if rank == 0:
    times = np.array(times)
    df = pd.DataFrame(data = times, index = lat_sizes, columns = [f'Cython MPI: {size} core'])
    df.to_csv(f'times/large_c_mpi_{size}.csv') 