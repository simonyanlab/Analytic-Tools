# df_tools.py - FEniCS related convenience functions
# 
# (C) Stefan Fuertinger
# Juni  5 2012

from __future__ import division
import numpy as np
import dolfin as df

# ------------------------------------------------------------------- #
# Given a digital image (a NumPy array) setup the DOLFIN framework
# ------------------------------------------------------------------- #
def arr2df(f_arr):

    # Some sanity checks assuring that f_arr is indeed a numpy array
    if type(f_arr).__name__ != "ndarray":
        raise TypeError("f_arr has to be a NumPy ndarray!")
    else:
        if len(f_arr.shape) > 2: raise ValueError("f_arr has to be 2-dimensional!")
        try: f_arr.shape[1]
        except: raise ValueError("f_arr has to be a matrix!")
        if np.isnan(f_arr).max() == True or np.isinf(f_arr).max() == True:
            raise ValueError("f_arr must not contain NaNs or Infs!")

    # Extract (discrete) image dimension
    M,N = f_arr.shape

    # Abort if image is not square
    if M!=N: raise ValueError("Image has to be square! Aborting...")

    # Create DOLFIN mesh
    mesh = df.UnitSquare(N,N)

    # Define piecewise constant function space
    S0 = df.FunctionSpace(mesh,"DG",0)

    # Initialize f_dol as member of S0
    f_dol = df.Function(S0)

    # Create empty vector that will hold intensities
    fa = np.zeros((2*N**2,))

    # Each pixel is divided into two triangles; 
    # assign intensity values to "lower" triangles (odd numbered cells)
    fa[::2]  = f_arr.flatten(1)

    # Assign intensities to "upper" triangles (even numbered cells)
    fa[1::2] = f_arr.flatten(1)

    # Update DOLFIN function coefficients
    f_dol.vector()[:] = fa
    
    # Throw back DOLFIN function and generated mesh
    return f_dol, mesh

# ------------------------------------------------------------------- #
# Given a DOLFIN function return a NumPy array (a digital image)
# ------------------------------------------------------------------- #
def df2arr(f_dol):

    # Some sanity checks assuring that f_dol is indeed a DOLFIN function
    if type(f_dol).__name__!="Function":
        raise TypeError("f_dol has to be a DOLFIN function!")

    # Check if f_dol is piecewise constant, if not project 
    if f_dol.is_cellwise_constant() == False:
        mesh = f_dol.function_space().mesh()
        f0   = df.project(f_dol,df.FunctionSpace(mesh,"DG",0))
    else:
        f0 = f_dol
    
    # Get cellwise function values
    fa = f0.vector().array()

    # Get (discrete) image dimension
    N = int(np.sqrt(fa.size/2.0))

    # Take the mean value of "upper" and "lower" triangles as pixelwise intensities
    f_arr = ((fa[::2] + fa[1::2])/2.0).reshape(N,N,order="F")

    # Throw back NumPy array holding NxN pixelwise intensity values
    return f_arr
