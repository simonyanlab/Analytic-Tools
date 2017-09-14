# loadimg.py - Load 2D gray-scale test images
# 
# Author: Stefan Fuertinger [stefan.fuertinger@gmx.at]
# December 23 2014

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import ipdb; import sys
import h5py

##########################################################################################
def barbara(N,ns=0.0):
    """
    Load the famous "Barbara"-testing image

    Parameters
    ----------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns:
    --------
    It : NumPy 2darray
        Array representation of the image (2D array). 

    Notes:
    ------
    None 

    See also:
    ---------
    None 
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # See if we can offer what the user wants
    N = getN(N)

    # Load image
    It = plt.imread(IMGPATH+str(N)+"x"+str(N)+os.sep+"barbara_"+str(N)+".tif") 
    It = It.astype(float)
    It = It[::-1,:]
    It = It/255.0

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def boat(N,ns=0.0):
    """
    BOAT loads the famous "boat"-testing image

    Parameters
    -------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns:
    --------
    It : NumPy 2darray
        Array representation of the image (2D array). 

    Notes:
    ------
    None 

    See also:
    ---------
    None 
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # See if we can offer what the user wants
    N = getN(N)

    # Load image
    It = plt.imread(IMGPATH+str(N)+"x"+str(N)+os.sep+"boat_"+str(N)+".tif") 
    It = It.astype(float)
    It = It[::-1,:]
    It = It/255.0

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def fingerprint(N,ns=0.0):
    """
    FINGERPRINT loads a sample image of a fingerprint

    Parameters
    -------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns:
    --------
    It : NumPy 2darray
        Array representation of the image (2D array). 

    Notes:
    ------
    None 

    See also:
    ---------
    None 
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # See if we can offer what the user wants
    N = getN(N)

    # Load image
    It = plt.imread(IMGPATH+str(N)+"x"+str(N)+os.sep+"fingerprint_"+str(N)+".tif") 
    It = It.astype(float)
    It = It[::-1,:]
    It = It/255.0

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def house(N,ns=0.0):
    """
    HOUSE loads a sample image of a house

    Parameters
    -------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns:
    --------
    It : NumPy 2darray
        Array representation of the image (2D array). 

    Notes:
    ------
    None 

    See also:
    ---------
    None 
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # See if we can offer what the user wants
    N = getN(N)

    # Load image
    It = plt.imread(IMGPATH+str(N)+"x"+str(N)+os.sep+"house_"+str(N)+".tif") 
    It = It.astype(float)
    It = It[::-1,:]
    It = It/255.0

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def I0mri(N,ns=0.0):
    """
    I0MRI loads a sample image of an DCE-MRI sequence

    Parameters
    -------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns:
    --------
    It : NumPy 2darray
        Array representation of the image (2D array). 

    Notes:
    ------
    None 

    See also:
    ---------
    None 
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # See if we can offer what the user wants
    N = getN(N)

    # Load image
    It = plt.imread(IMGPATH+str(N)+"x"+str(N)+os.sep+"I0mri_"+str(N)+".tif") 
    It = It.astype(float)
    It = It[::-1,:]
    It = It/255.0

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def I1mri(N,ns=0.0):
    """
    I1MRI loads a sample image of an DCE-MRI sequence

    Parameters
    -------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns:
    --------
    It : NumPy 2darray
        Array representation of the image (2D array). 

    Notes:
    ------
    None 

    See also:
    ---------
    None 
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # See if we can offer what the user wants
    N = getN(N)

    # Load image
    It = plt.imread(IMGPATH+str(N)+"x"+str(N)+os.sep+"I1mri_"+str(N)+".tif") 
    It = It.astype(float)
    It = It[::-1,:]
    It = It/255.0

    # import pdb;pdb.set_trace()

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def lena(N,ns=0.0):
    """
    LENA loads the famous "Lena"-image

    Parameters
    -------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns:
    --------
    It : NumPy 2darray
        Array representation of the image (2D array). 

    Notes:
    ------
    None 

    See also:
    ---------
    .. http://ndevilla.free.fr/lena/
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # See if we can offer what the user wants
    N = getN(N)

    # Load image
    It = plt.imread(IMGPATH+str(N)+"x"+str(N)+os.sep+"lena_"+str(N)+".tif") 
    It = It.astype(float)
    It = It[::-1,:]
    It = It/255.0

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def peppers(N,ns=0.0):
    """
    PEPPERS loads the "peppers"-image

    Parameters
    -------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns:
    --------
    It : NumPy 2darray
        Array representation of the image (2D array). 

    Notes:
    ------
    None 

    See also:
    ---------
    None 
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # See if we can offer what the user wants
    N = getN(N)

    # Load image
    It = plt.imread(IMGPATH+str(N)+"x"+str(N)+"/peppers_"+str(N)+".tif") 
    It = It.astype(float)
    It = It[::-1,:]
    It = It/255.0

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def getN(N):
    """
    Check if N is either 64, 128, 256 or 512. If not set N to be the 
    closest of those. 
    """
    
    # Image dimensions we have to offer
    Nlist = np.array([64,128,256,512])

    # Now see if the user picked one of those
    if (N != Nlist).all():
        print "WARNING: No images of format N = ",N," found!"
        
        # Pick the closest available image dimension instead
        idx = np.abs(Nlist - N).argmin()
        N   = Nlist[idx]

        print "Using N = ",N," instead..."

    return N
