try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from Cython.Build import cythonize
import numpy

from distutils.extension import Extension

cytest = Extension(
    "energy_test",
    sources=["cyth_test2.pyx"],
    extra_compile_args=['-O3'],
    extra_link_args=['-O3'],
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules = cythonize(cytest)
)