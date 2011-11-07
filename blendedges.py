# blendedges.py  - Superimposes an edge set on an grayscale image

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

def blendedges(Im,chim):
    """
    BLENDEDGES superimposes an edge set on an grayscale image using Matplotlib's imshow and contour

    Parameters:
    -----------
    Im: NumPy ndarray
        Greyscale image (has to be a 2D array)
    chim: NumPy ndarray
        Edge map (has to be a 2D array)
       
    Returns:
    --------
    None 

    Notes:
    ------
    None 

    See also:
    ---------
    imview.py
    Matplotlib's imshow and contour
    
    """

    # Sanity checks
    if type(Im).__name__ != "ndarray":
        raise TypeError("Im has to be a NumPy ndarray!")
    else:
        if len(Im.shape) > 2: raise ValueError("Im has to be 2-dimensional!")
        try: Im.shape[1]
        except: raise ValueError("Im has to be an image")
        if np.isnan(Im).max() == True or np.isinf(Im).max() == True:
            raise ValueError("Im must not contain NaNs or Infs!")

    if type(chim).__name__ != "ndarray":
        raise TypeError("chim has to be a NumPy ndarray!")
    else:
        if len(chim.shape) > 2: raise ValueError("chim has to be 2-dimensional!")
        try: chim.shape[1]
        except: raise ValueError("chim has to be an image")
        if np.isnan(chim).max() == True or np.isinf(chim).max() == True:
            raise ValueError("chim must not contain NaNs or Infs!")

    # Now do something
    plt.imshow(Im,cmap="gray",interpolation="nearest")
    plt.hold(True)
    plt.contour(chim,levels=[0],colors="r")
    plt.axis("off")
    plt.draw()

    return
