# setup.py - Distutils script to build the the Cython extension of the model
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# Created: June 25th 2014
# Last modified: <2016-07-22 15:24:39>

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import platform

# Because think different...
if platform.system() == 'Darwin':

    ext_modules1=[
        Extension("the_model",
                  ["the_model.pyx"],
                  libraries=["m"],
                  extra_link_args = ['-Wl,-framework', '-Wl,Accelerate'],                  
                  include_dirs=[numpy.get_include()])
    ]

else:

    ext_modules1=[
        Extension("the_model",
                  ["the_model.pyx"],
                  libraries=["m","blas"], # Unix specific
                  include_dirs=[numpy.get_include()])
    ]

# Build stuff
setup(
  name = "The_Model",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules1
)
