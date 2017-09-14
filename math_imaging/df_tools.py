# df_tools.py - FEniCS related convenience functions
# 
# Author: Stefan Fuertinger [stefan.fuertinger@gmx.at]
# Juni  5 2012

from __future__ import division
import numpy as np
import dolfin as df

##########################################################################################
def arr2df(f_arr):
    """
    ARR2DF converts a digital image (NumPy array) into a DOLFIN Function

    Inputs:
    -------
    f_arr : NumPy 2darray
        Array representation of the image (2D array, has to be square!!!)
       
    Returns:
    --------
    f_dol : DOLFIN Function
        A cell-wise constant function representing the given input image 
        in the DOLFIN framework. 
    mesh : DOLFIN Mesh
        A discretization of the unit square (0,1)x(0,1) using NxN 
        rectangles (the DOLFIN routine UnitSquare is used). 

    Notes:
    ------
    Note that DOLFIN at the moment only supports simplex meshes. Thus each 
    pixel is divided into two triangles. Then both triangles covering one 
    pixel are assigned the same intensity value. 

    See also:
    ---------
    DOLFIN's UnitSquare 
    """

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

##########################################################################################
def df2arr(f_dol):
    """
    DF2ARR converts a DOLFIN function into a digital image (NumPy array)

    Inputs:
    -------
    f_dol : DOLFIN Function
        A DOLFIN function given on an square(!!!) grid. 
       
    Returns:
    --------
    f_arr : NumPy 2darray
        Array representation of the image (2D array)

    Notes:
    ------
    Note that DOLFIN at the moment only supports simplex meshes. Thus two 
    neighboring triangles cover one pixel. Hence the mean value of "upper" 
    and "lower" triangle is used as intensity value of the compound pixel. 

    See also:
    ---------
    None
    """

    # Some sanity checks assuring that f_dol is indeed a DOLFIN function
    if type(f_dol).__name__!="Function":
        raise TypeError("f_dol has to be a DOLFIN function!")

    # Check if f_dol is piecewise constant, if not project 
    if f_dol.is_cellwise_constant() == False:
        mesh = f_dol.function_space().mesh()
        f0   = df.project(f_dol,df.FunctionSpace(mesh,"DG",0))
    else:
        f0 = f_dol
    
    # Get cell-wise function values
    fa = f0.vector().array()

    # Get (discrete) image dimension
    N = int(np.sqrt(fa.size/2.0))

    # Take the mean value of "upper" and "lower" triangles as pixelwise intensities
    f_arr = ((fa[::2] + fa[1::2])/2.0).reshape(N,N,order="F")

    # Throw back NumPy array holding NxN pixel-wise intensity values
    return f_arr
