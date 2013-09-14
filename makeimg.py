# makeimg.py - Create/load 2D test images
# 
# Author: Stefan Fuertinger
# Juni  6 2012

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os

# Set loading path for images
IMGPATH = os.path.realpath(__file__)
IMGPATH = IMGPATH.replace(os.path.basename(__file__),"")

def onesquare(N,ns=0.0):
    """
    ONESQUARE creates a simple NxN grey-scale image consisting only of one square

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
    None 
    """

    # Check if input parameters make sense
    checkinput(N,ns)

    # Generate image
    It = np.zeros((N,N));
    It[(3*N/16.0+1):(5*N/16.0),(3*N/16.0+1):(5*N/16.0)] = 1.0;

    # Add noise to image (if wanted)
    It = addnoise(It,N,ns)

    return It 

def fivesquares(N,ns=0.0):
    """
    FIVESQUARES creates an NxN grey-scale image consisting of five squares

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
    None 
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

def twospikes(N,ns=0.0):
    """
    TWOSPIKES creates an NxN grey-scale image consisting of two spikes

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
    None 
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

def bars(N,ns=0.0):
    """
    BARS creates the beloved NxN grey-scale image consisting of vertical and horizontal bars

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
    None 
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

def manysquares(N,ns=0.0):
    """
    MANYSQUARES creates an NxN grey-scale image consisting of many squares

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
    None 
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

def tgvtest(N,ns=0.0):
    """
    TGVTEST creates the famous NxN grey-scale image consisting of one shaded square

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
    None 
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

def myphantom(N,ns=0.0):
    """
    MYPHANTOM creates the famous Shepp--Logan phantom as NxN grey-scale image

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
    Jain, Anil K., Fundamentals of Digital Image Processing, Englewood Cliffs, NJ, Prentice Hall, 1989, p. 439
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

    # Create placholder for phantom image
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

def barbara(N,ns=0.0):
    """
    BARBARA loads the famous "Barbara"-testing image

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
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

def boat(N,ns=0.0):
    """
    BOAT loads the famous "boat"-testing image

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
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

def fingerprint(N,ns=0.0):
    """
    FINGERPRINT loads a sample image of a fingerprint

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
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

def house(N,ns=0.0):
    """
    HOUSE loads a sample image of a house

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
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

def I0mri(N,ns=0.0):
    """
    I0MRI loads a sample image of an DCE-MRI sequence

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
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

def I1mri(N,ns=0.0):
    """
    I1MRI loads a sample image of an DCE-MRI sequence

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
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

def lena(N,ns=0.0):
    """
    LENA loads the famous "Lena"-image

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
    http://ndevilla.free.fr/lena/
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

def peppers(N,ns=0.0):
    """
    PEPPERS loads the "peppers"-image

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension. Note that 
        the image is only available in 64x64, 128x128, 256x256 and 512x512. 
        If N is not either 64, 128, 256, 512 the best fit resolution is 
        chosen. 
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Array representation of the image (2D array). 

    Notes
    -----
    None 

    See also
    --------
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

def gengrid(N):
    """
    GENGRID creates an N-by-N grid needed to construct most artificial images

    Parameters:
    -----------
    N : int
        Positive integer determining the wanted image dimension
       
    Returns
    -------
    x : NumPy ndarray
        2D grid array of x-values on the domain [xmin,xmax]x[ymin,ymax]. 
    y : NumPy ndarray
        2D grid array of y-values on the domain [xmin,xmax]x[ymin,ymax]. 
    xmin : float
        The left boundary of the (rectangular) domain. By default xmin = 1. 
    xmax : float
        The right boundary of the (rectangular) domain. By default xmax is 
        dynamically set such that xmax = N. 
    ymin : float
        The lower boundary of the (rectangular) domain. By default ymin = 1. 
    ymin : float
        The lower boundary of the (rectangular) domain. By default ymin = 1. 
    x1 : NumPy ndarray
        1D array holding the x-spacing on the domain 
    y1 : NumPy ndarray
        1D array holding the y-spacing on the domain 

    Notes
    -----
    None 

    See also
    --------
    NumPy's meshgrid. 
    makegrid from myvec
    """

    # Set domain boundaries
    xmin = 1
    xmax = N
    ymin = 1
    ymax = N

    # Compute stepsize
    h  = (xmax - xmin)/(N-1);

    # Compute 1D grid arrays
    x1 = xmin + h*np.arange(0,N)
    y1 = ymin + h*np.arange(0,N)

    # Build 2D grid arrays
    y,x = np.meshgrid(x1,y1)

    return x,y,xmin,xmax,ymin,ymax,x1,y1

def addnoise(It,N,ns):
    """
    ADDNOISE imposes additive Gaussian noise to an image

    Parameters:
    -----------
    It : NumPy ndarray
        Array representation of an image (2D array). 
    N : int
        Positive integer determining the image dimension
    ns : float
        Floating point number setting the noise level to be applied to the 
        image, i.e. 0.1 = 10% noise, 0.2 = 20% noise, ... Thus ns has to 
        satisfy 0 <= ns <= 1. By default ns = 0.0. 
       
    Returns
    -------
    It : NumPy ndarray
        Noisy version of the input array 

    Notes
    -----
    None 

    See also
    --------
    None 
    """

    # Fix state of random number generator
    np.random.seed(0)

    # Add noise to the image
    It = It + ns*np.random.randn(N,N)
    It = It/It.max()

    return It

def checkinput(N,ns):
    """
    Perform sanity checks on the inputs N and ns
    """

    # Sanity checks
    # N
    try: N/2.0
    except: raise TypeError("N has to be a positive integer!")
    if N <= 1: raise ValueError("N has to be greater than 1!")
    if np.round(N) != N:
        raise ValueError("N has to be an integer!")

    # ns
    try: float(ns)
    except: raise TypeError("ns has to be the noise level, i.e. 0 <= ns <= 1!")
    if ns < 0 or ns > 1: raise ValueError("ns has to be in [0,1]!")

    return

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
