# imview.py - Plot greyscale image using "sane" default settings for imshow

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

def imview(Im):
    """
    IMVIEW plots a greyscale image using "sane" defaults for Matplotlib's imshow. 

    Parameters:
    -----------
    Im: NumPy ndarray
        Greyscale image to plot (has to be a 2D array)
       
    Returns
    -------
    None 

    Notes
    -----
    None 

    See also
    --------
    view_image.py
    Matplotlib's imshow

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
        
    # Now do something
    plt.imshow(Im,cmap="gray",interpolation="nearest")
    plt.axis("off")
    plt.draw()

    return
