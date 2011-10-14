# myquiv.py - Plot a vector field using "sane" settings for quiver

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

def myquiv(u,v):
    """
    IMVIEW plots a vector field w=(u,v) using "sane" defaults for Matplotlib's quiver.

    Parameters:
    -----------
    u: NumPy ndarray
        x components of the vector field w(x,y) = (u(x,y),v(x,y)). Note that u has to be 
        a 2D array of the same dimension as v. 
    v: NumPy ndarray
        y components of the vector field w(x,y) = (u(x,y),v(x,y)). Note that v has to be 
        a 2D array of the same dimension as u. 
       
    Returns
    -------
    None 

    Notes
    -----
    None 

    See also
    --------
    Matplotlib's quiver

    """

    # Sanity checks
    if type(u).__name__ != "ndarray":
        raise TypeError("u has to be a NumPy ndarray!")
    else:
        if len(u.shape) > 2: raise ValueError("u has to be 2-dimensional!")
        try: u.shape[1]
        except: raise ValueError("u has to be a matrix!")
        if np.isnan(u).max() == True or np.isinf(u).max() == True:
            raise ValueError("u must not contain NaNs or Infs!")

    if type(v).__name__ != "ndarray":
        raise TypeError("v has to be a NumPy ndarray!")
    else:
        if len(v.shape) > 2: raise ValueError("v has to be 2-dimensional!")
        try: v.shape[1]
        except: raise ValueError("v has to be a matrix!")
        if np.isnan(v).max() == True or np.isinf(v).max() == True:
            raise ValueError("v must not contain NaNs or Infs!")

    if u.shape[0] != v.shape[0] or u.shape[1] != v.shape[1]:
        raise IndexError("u and v must have the same dimensions!")
        
    # Now do something
    N  = u.shape[0]
    dN = min(N,16)
    plt.quiver(v[N-1:0:(-N/dN),0:N:(N/dN)],-u[N-1:0:(-N/dN),0:N:(N/dN)],color="k")
    plt.axis("image")
    plt.axis("off")
    plt.draw()

    return
