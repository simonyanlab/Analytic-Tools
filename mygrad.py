# mygrad.py - Compute gradient of 2D function

from __future__ import division
import numpy as np

def mygrad(F):
    """
    MYGRAD computes the numerical gradient of a 2D function using central differences

    Parameters:
    -----------
    F: NumPy ndarray
       A function defined on a M-by-N grid, i.e. a two-dimensional NumPy array

    Returns:
    --------
    Fx: NumPy ndarray
       Array of the same dimension as F containing the x-derivatives of F
    Fy: NumPy ndarray
       Array of the same dimension as F containing the y-derivatives of F

    Notes:
    ------
    In contrast to MATLAB Fx contains derivatives along rows and Fy holds
    the derivatives along columns. 

    See also:
    ---------
    MATLAB's gradient function 
    http://www.mathworks.de/help/techdoc/ref/gradient.html

    """

    # Sanity checks
    if type(F).__name__ != "ndarray":
        raise TypeError("F has to be a NumPy ndarray!")
    else:
        if len(F.shape) > 2: raise ValueError("F has to be 2-dimensional!")
        try: F.shape[1]
        except: raise ValueError("F has to be 2-dimensional!")
        if np.isnan(F).max() == True or np.isinf(F).max() == True:
            raise ValueError("F must not contain NaNs or Infs!")

    # Now do something
    M = F.shape[0]
    N = F.shape[1]

    # Allocate space for output
    Fx = np.zeros((M,N))
    Fy = np.zeros((M,N))

    # Compute x-derivatives
    Fx[0,:]     = F[1,:]   - F[0,:]
    Fx[1:M-1,:] = F[2:M,:] - F[0:M-2,:]
    Fx[-1,:]    = F[-1,:]  - F[-2,:]

    # Compute y-derivatives
    Fy[:,0]     = F[:,1]   - F[:,0]
    Fy[:,1:N-1] = F[:,2:N] - F[:,0:N-2]
    Fy[:,-1]    = F[:,-1]  - F[:,-2]

    return Fx, Fy
