from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Amp utils',
    ext_modules=cythonize(["utils_base/amp_utils.pyx"]),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)