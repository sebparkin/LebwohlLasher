import numpy as np
import pandas as pd
from mpi4py import MPI

from LL_mpi import LL_mpi_MCstep_timing
from LL_mpi import LL_mpi_timing
from LL_cython import LL_cython_mpi_timing

params = [200, 50, 0.5]
mpi_time = []
mcstep_time = []
cython_time = []

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


for i in range(3):
    mpi_time.append(LL_mpi_timing.main(200, 50, 0.5, comm))
    mcstep_time.append(LL_mpi_MCstep_timing.main(200, 50, 0.5, comm))
    cython_time.append(LL_cython_mpi_timing.main(200, 50, 0.5, comm))

comm.Barrier()
#mpi_time = comm.bcast(mpi_time, root = 0)
#mcstep_time = comm.bcast(mcstep_time, root = 0)
#cython_time_time = comm.bcast(cython_time, root = 0)

if rank == 0:
    mpi_time = np.average(np.array(mpi_time))
    mcstep_time = np.average(np.array(mcstep_time))
    cython_time = np.average(np.array(cython_time))

    index = ['MPI', 'MPI MC step', 'Cython MPI']

    df = pd.DataFrame(data=[mpi_time, mcstep_time, cython_time], index=index)

    df.to_csv('./times/compare_mpi.csv')
