# makeimg.py - Create 2D gray-scale test images
# 
# Author: Stefan Fuertinger [stefan.fuertinger@gmx.at]
# June  6 2012

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

##########################################################################################
def onesquare(N,ns=0.0):
    """
    Create a simple piece-wise constant gray-scale image: a white square on a black background

    Parameters
    ----------
    N : int
        Image dimension (image is an `N`-by-`N` array) 
    ns : float
        Noise level applied to the image, i.e., 0.1 = 10% noise, 0.2 = 20% noise, etc. Thus `ns` has to 
        satisfy `0 <= ns <= 1`. By default `ns = 0.0`. 
       
    Returns
    -------
    It : NumPy 2darray
        Array representation of the image (2D array)

    See also
    --------
    fivesquares : another piece-wise constant test image
    myphantom : Python implementation of the Shepp--Logan phantom
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # Generate image
    It = np.zeros((N,N));
    It[(3*N/16.0+1):(5*N/16.0),(3*N/16.0+1):(5*N/16.0)] = 1.0;

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def fivesquares(N,ns=0.0):
    """
    Create a piece-wise constant gray-scale image: five squares of different intensities on black background

    Parameters
    ----------
    N : int
        Image dimension (image is an `N`-by-`N` array) 
    ns : float
        Noise level applied to the image, i.e., 0.1 = 10% noise, 0.2 = 20% noise, etc. Thus `ns` has to 
        satisfy `0 <= ns <= 1`. By default `ns = 0.0`. 
       
    Returns
    -------
    It : NumPy 2darray
        Array representation of the image (2D array)

    See also
    --------
    onesquare : another piece-wise constant test image
    myphantom : Python implementation of the Shepp--Logan phantom
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # Generate grid on domain
    x,y,xmin,xmax,ymin,ymax,x1,y1 = gengrid(N)

    # Generate image
    u1 = np.zeros((N,N))
    u1[(3*N/16+1):(5*N/16),(3*N/16+1):(5*N/16)] = 1.0
    u2 = np.zeros((N,N))
    u2[(7*N/16+1):(9*N/16),(7*N/16+1):(9*N/16)] = 1.0
    u3 = np.zeros((N,N))
    u3[(11*N/16+1):(13*N/16),(11*N/16+1):(13*N/16)] = 1.0
    u4 = np.zeros((N,N))
    u4[(3*N/16+1):(5*N/16),(11*N/16+1):(13*N/16)] = 1.0
    u5 = np.zeros((N,N))
    u5[(11*N/16+1):(13*N/16),(3*N/16+1):(5*N/16)] = 1.0
    It = u1*((0.1*x + 0.9*y)/N + 0.9) \
        + u2*((0.3*x + 0.7*y)/N + 0.7) \
        + u3*((0.5*x + 0.5*y)/N + 0.5) \
        + u4*((0.7*x + 0.3*y)/N + 0.3) \
        + u5*((0.9*x + 0.1*y)/N + 0.1)

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def spikes(N,ns=0.0):
    """
    Create a piece-wise linear gray-scale image of four squares with gradually increasing intensities

    Parameters
    ----------
    N : int
        Image dimension (image is an `N`-by-`N` array) 
    ns : float
        Noise level applied to the image, i.e., 0.1 = 10% noise, 0.2 = 20% noise, etc. Thus `ns` has to 
        satisfy `0 <= ns <= 1`. By default `ns = 0.0`. 
       
    Returns
    -------
    It : NumPy 2darray
        Array representation of the image (2D array)

    See also
    --------
    bars : piece-wise linear test image of rectangles
    manysquares : piece-wise linear test image of squares
    tgvtest : gray scale image widely used to illustrate weaknesses of total variation denoising
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # Generate grid on domain
    x,y,xmin,xmax,ymin,ymax,x1,y1 = gengrid(N)

    # Generate image
    It = ((x1 - xmin)/(xmax - xmin));
    It[1:(2*N/16)+1] = 0;
    It[(2*N/16+1):(7*N/16)+1] = It[(2*N/16+1):(7*N/16)+1] /\
        It[(2*N/16+1):(7*N/16)+1].max()
    It[(7*N/16+1):(8*N/16)+1] = 0;
    It[(8*N/16+1):(13*N/16)+1] = It[(8*N/16+1):(13*N/16)+1] /\
        It[(8*N/16+1):(13*N/16)+1].max()
    It[(13*N/16+1):(14*N/16)+1] = 0
    It[(14*N/16+1):N] = 0;
    It = np.array(np.kron(np.matrix(It),np.matrix(It).T))
    It = It/It.max()

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def bars(N,ns=0.0):
    """
    Create a piece-wise linear gray-scale image of rectangles with gradually increasing intensities

    Parameters
    ----------
    N : int
        Image dimension (image is an `N`-by-`N` array) 
    ns : float
        Noise level applied to the image, i.e., 0.1 = 10% noise, 0.2 = 20% noise, etc. Thus `ns` has to 
        satisfy `0 <= ns <= 1`. By default `ns = 0.0`. 
       
    Returns
    -------
    It : NumPy 2darray
        Array representation of the image (2D array)

    See also
    --------
    spikes : piece-wise linear test image of four squares
    manysquares : piece-wise linear test image of squares
    tgvtest : gray scale image widely used to illustrate weaknesses of total variation denoising
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # Generate grid on domain
    x,y,xmin,xmax,ymin,ymax,x1,y1 = gengrid(N)

    # Generate image
    It = ((x1 - xmin)/(xmax - xmin));
    It[1:(2*N/16)+1] = 0;
    It[(2*N/16+1):(7*N/16)+1] = It[(2*N/16+1):(7*N/16)+1] /\
        It[(2*N/16+1):(7*N/16)+1].max()
    It[(7*N/16+1):(8*N/16)+1] = 0;
    It[(8*N/16+1):(10*N/16)+1] = It[(8*N/16+1):(10*N/16)+1] /\
        It[(8*N/16+1):(10*N/16)+1].max()
    It[(10*N/16+1):(13*N/16)+1] = 0;
    It[(13*N/16+1):(14*N/16)+1] = It[(13*N/16+1):(14*N/16)+1] /\
        It[(13*N/16+1):(14*N/16)+1].max()
    It[(14*N/16+1):N] = 0;
    It = np.array(np.kron(np.matrix(It),np.matrix(It).T))

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def manysquares(N,ns=0.0):
    """
    Create a piece-wise linear gray-scale image of 16 squares with gradually increasing intensities

    Parameters
    ----------
    N : int
        Image dimension (image is an `N`-by-`N` array) 
    ns : float
        Noise level applied to the image, i.e., 0.1 = 10% noise, 0.2 = 20% noise, etc. Thus `ns` has to 
        satisfy `0 <= ns <= 1`. By default `ns = 0.0`. 
       
    Returns
    -------
    It : NumPy 2darray
        Array representation of the image (2D array)

    See also
    --------
    spikes : piece-wise linear test image of four squares
    bars : piece-wise linear test image of rectangles
    tgvtest : gray scale image widely used to illustrate weaknesses of total variation denoising
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # Generate grid on domain
    x,y,xmin,xmax,ymin,ymax,x1,y1 = gengrid(N)

    # Generate image
    It = ((x1 - xmin)/(xmax - xmin))**2
    m = 8
    for j in xrange(1,m,2):
        It[((j-1)*N/m+1):(j*N/m)+1] = 0
    for j in xrange(2,m,2):
        It[((j-1)*N/m+1):(j*N/m)+1] = It[((j-1)*N/m+1):(j*N/m)+1] /\
            It[((j-1)*N/m+1):(j*N/m)+1].max()
    It = np.array(np.kron(np.matrix(It),np.matrix(It).T))

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def tgvtest(N,ns=0.0):
    """
    Create a gray-scale image consisting of one shaded square often used to illustrate TV denoising problems

    Parameters
    ----------
    N : int
        Image dimension (image is an `N`-by-`N` array) 
    ns : float
        Noise level applied to the image, i.e., 0.1 = 10% noise, 0.2 = 20% noise, etc. Thus `ns` has to 
        satisfy `0 <= ns <= 1`. By default `ns = 0.0`. 
       
    Returns
    -------
    It : NumPy 2darray
        Array representation of the image (2D array)

    See also
    --------
    spikes : piece-wise linear test image of four squares
    bars : piece-wise linear test image of rectangles
    manysquares : piece-wise linear test image of squares
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # Generate image
    It = np.zeros((N,N))
    It[(N/4):(3*N/4),(N/4):(3*N/4)] = 1
    It[(N/4):(3*N/4),0:(N/4)] = np.array(np.kron(np.matrix(np.arange(0,N/4)/(N/4)),np.ones((N/2,1))))

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

##########################################################################################
def myphantom(N,ns=0.0):
    """
    Create the famous Shepp--Logan phantom

    Parameters
    ----------
    N : int
        Image dimension (image is an `N`-by-`N` array) 
    ns : float
        Noise level applied to the image, i.e., 0.1 = 10% noise, 0.2 = 20% noise, etc. Thus `ns` has to 
        satisfy `0 <= ns <= 1`. By default `ns = 0.0`. 
       
    Returns
    -------
    It : NumPy 2darray
        Array representation of the image (2D array)

    See also
    --------
    onesquare : another piece-wise constant test image
    fivesquares : another piece-wise constant test image

    References
    ----------
    .. [1] Jain, Anil K., Fundamentals of Digital Image Processing, 
           Englewood Cliffs, NJ, Prentice Hall, 1989, p. 439
    """
    
    # Check if input parameters make sense
    checkinput(N,ns)

    # Basic building block for phantom
    shep = np.array([[  1,   .69,   .92,    0,     0,     0],   
                     [-.8,  .6624, .8740,   0,  -.0184,   0],
                     [-.2,  .1100, .3100,  .22,    0,    -18],
                     [-.2,  .1600, .4100, -.22,    0,     18],
                     [.1,  .2100, .2500,   0,    .35,    0],
                     [.1,  .0460, .0460,   0,    .1,     0],
                     [.1,  .0460, .0460,   0,   -.1,     0],
                     [.1,  .0460, .0230, -.08,  -.605,   0], 
                     [.1,  .0230, .0230,   0,   -.606,   0],
                     [.1,  .0230, .0460,  .06,  -.605,   0]])

    # Create placeholder for phantom image
    It = np.zeros((N,N));

    # Create spaced array
    xax = (np.arange(0,N)-(N-1)/2.0)/((N-1)/2.0)

    # Do a "repmat" type thing
    xg = np.tile(xax, (N, 1))

    # Create phantom image (shameless copy of MATLAB code)
    for k in xrange(shep.shape[0]):
        asq = shep[k,1]**2
        bsq = shep[k,2]**2
        phi = shep[k,5]*np.pi/180.0
        x0  = shep[k,3]
        y0  = shep[k,4]
        A   = shep[k,0]
        x = xg - x0; y = np.rot90(xg) - y0
        cosp = np.cos(phi); sinp = np.sin(phi)
        idx=np.nonzero((((x*cosp + y*sinp)**2)/asq + ((y*cosp - x*sinp)**2)/bsq) <= 1.0)
        It[idx] = It[idx] + A
   
    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It

##########################################################################################
def gengrid(N):
    """
    Creates an `N`-by-`N` grid needed to construct most artificial images

    Parameters
    ----------
    N : int
        Image dimension
       
    Returns
    -------
    x : NumPy 2darray
        2D grid array of `x`-values on the domain `[xmin,xmax]`-by-`[ymin,ymax]` 
    y : NumPy 2darray
        2D grid array of `y`-values on the domain `[xmin,xmax]`-by-`[ymin,ymax]`
    xmin : float
        Left boundary of the (rectangular) domain. By default `xmin = 1` 
    xmax : float
        Right boundary of the (rectangular) domain. By default `xmax = N`
    ymin : float
        The lower boundary of the (rectangular) domain. By default `ymin = 1` 
    ymax : float
        The upper boundary of the (rectangular) domain. By default `ymax = N` 
    x1 : NumPy 1darray
        The `x`-spacing on the domain 
    y1 : NumPy 1darray
        The `y`-spacing on the domain 

    See also
    --------
    meshgrid : NumPy's `meshgrid <http://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html>`_
    makegrid : found in the module myvec.makegrid
    """

    # Set domain boundaries
    xmin = 1
    xmax = N
    ymin = 1
    ymax = N

    # Compute step-size
    h  = (xmax - xmin)/(N-1);

    # Compute 1D grid arrays
    x1 = xmin + h*np.arange(0,N)
    y1 = ymin + h*np.arange(0,N)

    # Build 2D grid arrays
    y,x = np.meshgrid(x1,y1)

    return x,y,xmin,xmax,ymin,ymax,x1,y1

##########################################################################################
def addnoise(It,N,ns):
    """
    Add Gaussian noise to an image

    Parameters
    ----------
    It : NumPy 2darray
        Array representation of an image (2D array). 
    N : int
        Image dimension (image is an `N`-by-`N` array) 
    ns : float
        Noise level applied to the image, i.e., 0.1 = 10% noise, 0.2 = 20% noise, etc. Thus `ns` has to 
        satisfy `0 <= ns <= 1`. By default `ns = 0.0`. 
       
    Returns
    -------
    It : NumPy 2darray
        Noisy version of the input array 
    """

    # Fix state of random number generator
    np.random.seed(0)

    # Add noise to the image
    It = It + ns*np.random.randn(N,N)
    It = It/It.max()

    return It

##########################################################################################
def checkinput(N,ns):
    """
    Perform sanity checks on the inputs `N` and `ns`
    """

    # Sanity checks
    # N
    try: bad = (N != int(N))
    except: raise TypeError("N has to be a positive integer!")
    if N <= 1: raise ValueError("N has to be greater than 1!")
    if bad: raise ValueError("N has to be an integer!")

    # ns
    try: float(ns)
    except: raise TypeError("ns has to be the noise level, i.e. 0 <= ns <= 1!")
    if ns < 0 or ns > 1: raise ValueError("ns has to be in [0,1]!")

    return
