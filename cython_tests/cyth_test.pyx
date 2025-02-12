cimport cython

import math
import numpy as np
from libc.math cimport sqrt, cos
cimport numpy as cnp

def calculate_roots(numbers):
    cdef int num_vals = len(numbers)
    result = np.zeros(num_vals, "f")

    cdef double[::1] numbers_view = numbers
    cdef float[::1] result_view = result

    cdef int i = 0

    with nogil:
        for i in range(0, num_vals):
            result_view[i] = cos(numbers_view[i])

    return result

cnp.import_array()

DTYPE = np.double
ctypedef cnp.double_t DTYPE_T

def get_order(cnp.ndarray arr, int nmax):
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


    cdef cnp.ndarray Qab = np.zeros((3,3), dtype=DTYPE)
    cdef cnp.ndarray delta = np.eye(3,3, dtype=DTYPE)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    cdef cnp.ndarray lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr)), dtype = DTYPE).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()