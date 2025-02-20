from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
import numpy
import os

if sys.platform == 'win32':
    compile_args = ['/openmp', '-O3']
    link_args = []
else:
    compile_args = ['-fopenmp', '-O3']
    link_args = ['-fopenmp']
    mpi_link_args = os.popen("mpicxx --showme:link").read().split()

extensions = [
    Extension(
        "LL_cython_mpi",                   # Name of the compiled module
        ["LL_cython_mpi.pyx"],              # Source file
        extra_compile_args=compile_args,  # Compiler flags for OpenMP
        extra_link_args=mpi_link_args + link_args,
        include_dirs=[numpy.get_include()]  # Linker flags for OpenMP
        #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    name="LL_cython_mpi",
    ext_modules=cythonize(extensions, language_level = "3"),
)