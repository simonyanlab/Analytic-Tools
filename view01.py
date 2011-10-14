# Plot image in the range [0,1]
import matplotlib.pyplot as plt

def view01(Im):
    """Plot image in the range [0,1]
    Input:  Im ndarray (Image)
    Output: --
    """
    
    # Show image and disable axes
    plt.imshow(Im,cmap="gray",interpolation="nearest",vmin=0.0,vmax=1.0)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    return
