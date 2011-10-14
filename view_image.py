# Python version of view_image.m
from matplotlib.pyplot import gca, imshow

def view_image(Im):
    """Python implementation of view_image.m
    Input:  Im ndarray (Image)
    Output: --
    """
    
    # Show image and disable axes
    imshow(Im,cmap="gray",interpolation="nearest")
    gca().axes.get_xaxis().set_visible(False)
    gca().axes.get_yaxis().set_visible(False)

    return
