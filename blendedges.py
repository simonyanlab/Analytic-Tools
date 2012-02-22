# blendedges.py  - Superimposes an edge set on an grayscale image

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

def blendedges(Im,chim):
    """
    BLENDEDGES superimposes a (binary) edge set on an grayscale image using Matplotlib's imshow

    Parameters:
    -----------
    Im: NumPy ndarray
        Greyscale image (has to be a 2D array)
    chim: NumPy ndarray
        Binary edge map (has to be a 2D array). Note that the edge map must only contain 
        the valus 0 and 1. 
       
    Returns:
    --------
    None 

    Notes:
    ------
    None 

    See also:
    ---------
    Matplotlib's imshow
    http://stackoverflow.com/questions/2495656/variable-alpha-blending-in-pylab
    """

    # Sanity checks
    if type(Im).__name__ != "ndarray":
        raise TypeError("Im has to be a NumPy ndarray!")
    else:
        if len(Im.shape) > 2: raise ValueError("Im has to be 2-dimensional!")
        try: Im.shape[1]
        except: raise ValueError("Im has to be an image!")
        if np.isnan(Im).max() == True or np.isinf(Im).max() == True:
            raise ValueError("Im must not contain NaNs or Infs!")

    if type(chim).__name__ != "ndarray":
        raise TypeError("chim has to be a NumPy ndarray!")
    else:
        if len(chim.shape) > 2: raise ValueError("chim has to be 2-dimensional!")
        try: chim.shape[1]
        except: raise ValueError("chim has to be an edge map!")
        if np.isnan(chim).max() == True or np.isinf(chim).max() == True:
            raise ValueError("chim must not contain NaNs or Infs!")
        chim = chim.astype(float)
        chiu = np.unique(chim)
        if chiu.size != 2: raise ValueError("chim has to be binary!")
        if chiu.min() != 0 or chiu.max() != 1: raise ValueError("chim has to be a binary edge map!")

    # Now do something
    plt.imshow(Im,cmap="gray",interpolation="nearest")
    plt.hold(True)
    plt.imshow(mycmap(chim))
    plt.axis("off")
    plt.draw()

    return

def mycmap(x):
    """
    Generate a custom color map, setting alpha values to one on edge
    points, and to zero otherwise
    
    Notes:
    ------
    This code is based on the suggestion found at
    http://stackoverflow.com/questions/2495656/variable-alpha-blending-in-pylab 
    """

    # Convert edge map to Matplotlib colormap (shape (N,N,4))
    tmp = plt.cm.hsv(x)

    # Set alpha values to one on edge points
    tmp[:,:,3] = x

    return tmp
