# Repository for optimising Lebwohl Lasher script

## Instructions for use

### For Default, LL_numpy, and LL_numba:
* Run by typing: ```python {program path and name} {iterations} {size} {temp} {plotflag}```
* For example: ```python ./LL_numpy/LL_numpy.py 50 50 0.5 0``` will run the numpy version of the program for 50 iterations with a lattice size of 50 x 50.

### For all MPI programs:
* On my mac CC=gcc-14 is needed at the start.
* Run by typing ```mpiexec -n {number of cores} python {program path and name} {iterations} {size} {temp} {plotflag}```.
* For example: mpiexec -n 4 python ./LL_numba/LL_numba_mpi.py 200 50 0.5 1 will run the numba version with MPI, with 200 steps, 50 x 50 lattice size, and plots.

### For Cython programs:
* On my mac CC=gcc-14 is needed at the start for all steps.
* You need to be inside the LL_cython directory.
* Run by building the program first: ```python {program name}_setup.py build_ext --inplace```.
* Then run with ```python {program name}_run.py {iterations} {size} {temp} {plotflag}```.
* Cython with MPI is the same but with ```mpiexec -n {number of cores}``` before python on the run step.