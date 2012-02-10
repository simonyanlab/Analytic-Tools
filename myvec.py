# myvec.py - Plotting tools for vector fields

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def myquiv(u,v):
    """
    MYQUIV plots a vector field w=(u,v) using "sane" defaults for Matplotlib's quiver.

    Parameters:
    -----------
    u : NumPy ndarray
        x components of the vector field w(x,y) = (u(x,y),v(x,y)). Note that u has to be 
        a 2D array of the same dimension as v. 
    v : NumPy ndarray
        y components of the vector field w(x,y) = (u(x,y),v(x,y)). Note that v has to be 
        a 2D array of the same dimension as u. 
       
    Returns:
    --------
    None 

    Notes:
    ------
    None 

    See also:
    ---------
    Matplotlib's quiver

    """

    # Check the input vector field
    checkinput(u,v)

    # Now do something
    N  = u.shape[0]
    dN = min(N,16)
    plt.quiver(v[N-1:0:(-N/dN),0:N:(N/dN)],-u[N-1:0:(-N/dN),0:N:(N/dN)],color="k")
    plt.axis("image")
    plt.axis("off")
    plt.draw()

    return

def makegrid(N,M=None,xmin=1,xmax=None,ymin=1,ymax=None):
    """
    MAKEGRID creates an M-by-N grid on the 2D-domain [xmin,xmax]x[ymin,ymax]

    Parameters:
    -----------
    N : int
        Integer determining the number of gridpoints in vertical (i.e. y-)
        direction
    M : int 
        Integer determining the number of gridpoints in vertical (i.e. y-)
        direction. By default M is dynamically set such that M = N. 
    xmin : float
        The left boundary of the (rectangular) domain. By default xmin = 1. 
    xmax : float
        The right boundary of the (rectangular) domain. By default xmax is 
        dynamically set such that xmax = N. 
    ymin : float
        The lower boundary of the (rectangular) domain. By default ymin = 1. 
    ymax : float
        The upper boundary of the (rectangular) domain. By default ymax is 
        dynamically set such that ymax = M. 
    
    Returns:
    --------
    x : NumPy ndarray
        2D grid array of x-values on the domain [xmin,xmax]x[ymin,ymax]. 
    y : NumPy ndarray
        2D grid array of y-values on the domain [xmin,xmax]x[ymin,ymax]. 

    Notes:
    ------
    The call 
        x,y = makegrid(N)
    creates a square [1,N]-by-[1,N] grid given by
       >>> x
       array([[   1.,    1.,    1., ...,    1.],
              [   2.,    2.,    2., ...,    2.],
              ..., 
              [    N,     N,     N, ...,     N]])
    and
       >>> y
       array([[   1.,    2.,    3., ...,     N],
              [   1.,    2.,    3., ...,     N],
              ..., 
              [   1.,    2.,    3., ...,     N]])
    See also:
    ---------
    NumPy's meshgrid. 

    """

    # Sanity checks
    # N
    try: N/2.0
    except: raise TypeError("N has to be a positive integer!")
    if N <= 1: raise ValueError("N has to be greater than 1!")
    if np.round(N) != N:
        N = np.round(N)
        print "WARNING: N has to be an integer - using round(N) = ",N," now..."

    # M
    if M == None:
        M = N
    if M!=N:
        try: M/2.0
        except: raise TypeError("M has to be a positive integer!")
        if M <= 1: raise ValueError("M has to be greater than 1!")
        if np.round(M) != M:
            M = np.round(M)
            print "WARNING: M has to be an integer - using round(M) = ",M," now..."

    # xmin
    if xmin != 1:
        try: xmin/2.0
        except: raise TypeError("xmin has to be a real number!")

    # xmax
    if xmax == None: xmax = N
    try: xmax/2.0
    except: raise TypeError("xmax has to be a real number!")
    if xmax <= xmin: raise ValueError("xmax has to be greater than xmin!")

    # ymin
    if ymin != 1:
        try: ymin/2.0
        except: raise TypeError("ymin has to be a real number!")

    # ymax
    if ymax == None: ymax = M
    try: ymax/2.0
    except: raise TypeError("ymax has to be a real number!")
    if ymax <= ymin: raise ValueError("ymax has to be greater than ymin!")

    # Compute stepsizes
    hx = (xmax - xmin)/(M-1)
    hy = (ymax - ymin)/(N-1)

    # Build 1D grid arrays
    x1 = xmin + hx*np.arange(0,M)
    y1 = ymin + hy*np.arange(0,N)

    # Build 2D grid arrays
    y,x = np.meshgrid(x1,y1)

    return x,y

def mygrid(u,v,x=None,y=None,rowstep=16,colstep=16,interpolation="lanczos"):
    """
    MYGRID plots a vector field w=(u,v) as deformed grid on a 2D lattice. 

    Parameters:
    -----------
    u : NumPy ndarray
        x components of the vector field w(x,y) = (u(x,y),v(x,y)). Note that u has to be 
        a 2D array of the same dimension as v. 
    v : NumPy ndarray
        y components of the vector field w(x,y) = (u(x,y),v(x,y)). Note that v has to be 
        a 2D array of the same dimension as u. 
    x : NumPy ndarray
        2D grid array of x-values on the domain of w(x,y). By default it is assumed that 
        w has the domain [1,N]-by-[1,N]. 
    y : NumPy ndarray
        2D grid array of y-values on the domain of w(x,y). By default it is assumed that 
        w has the domain [1,N]-by-[1,N]. 
    rowstep : int
        Integer determining the array row stride (step size) used to generate the grid. 
        Default value is 16. 
    colstep : int
        Integer determining the array column stride (step size) used to generate the grid. 
        Default value is 16. 
    interpolation : str
        String determining interpolation to be used for plotting. Default 
        value is "lanczos". Recommended other values are "bilinear" or "nearest". 
        See Matplotlib's imshow-documentation for details. 

       
    Returns:
    --------
    None 

    Notes:
    ------
    None

    See also:
    ---------
    None

    """

    # Check the input vector field
    checkinput(u,v)
    (M,N) = u.shape

    # Check the grid
    if (x == None and y != None) or (x != None and y == None):
        print "WARNING: Only x- or y-grid-data specified - switching to default domain [1,N]-by-[1,N]..."
        x,y = makegrid(N,M=M)
    elif x == None:
        x,y = makegrid(N,M=M)
    else:
        checkgrid(x,y)
    if x.shape[0] != u.shape[0] or x.shape[1] != u.shape[1]:
        raise IndexError("Grid and vector field must have the same dimensions!")

    # Sanity checks
    try: rowstep/2.0
    except: raise TypeError("rowstep has to be a positive integer!")
    if rowstep < 1: raise ValueError("rowstep has to be >= 1!")
    if rowstep >= u.shape[0]: raise ValueError("rowstep has to be less than u.shape[0]!")
    if np.round(rowstep) != rowstep:
        rowstep = np.round(rowstep)
        print "WARNING: rowstep has to be an integer - using round(rowstep) = ",rowstep," now..."

    try: colstep/2.0
    except: raise TypeError("colstep has to be a positive integer!")
    if colstep < 1: raise ValueError("colstep has to be >= 1!")
    if colstep >= u.shape[1]: raise ValueError("colstep has to be less than u.shape[1]!")
    if np.round(colstep) != colstep:
        colstep = np.round(colstep)
        print "WARNING: colstep has to be an integer - using round(colstep) = ",colstep," now..."

    if type(interpolation).__name__ != "str":
        raise TypeError("interpolation has to be a string!")

    # Create lattice
    wires = np.ones(u.shape)
    wires[0:-1:rowstep,:] = 0
    wires[:,0:-1:colstep] = 0

    # Apply vector field
    zipyx = zip(y.flatten(1),x.flatten(1))
    wiresr = griddata(zipyx,wires.flatten(1),(y+v,x+u),method="linear",fill_value=1)
    
    # Plot it
    plt.imshow(wiresr,cmap="gray",interpolation=interpolation)
    plt.axis("image")
    plt.axis("off")
    plt.draw()

    return

def mywire(u,v,x=None,y=None,rowstep=1,colstep=1):
    """
    MYWIRE plots a vector field w=(u,v) as 3D wireframe using Matplotlib's 
    plot_wireframe. 

    Parameters:
    -----------
    u : NumPy ndarray
        x components of the vector field w(x,y) = (u(x,y),v(x,y)). Note that u has to be 
        a 2D array of the same dimension as v. 
    v : NumPy ndarray
        y components of the vector field w(x,y) = (u(x,y),v(x,y)). Note that v has to be 
        a 2D array of the same dimension as u. 
    x : NumPy ndarray
        2D grid array of x-values on the domain of w(x,y). By default it is assumed that 
        w has the domain [1,N]-by-[1,N]. 
    y : NumPy ndarray
        2D grid array of y-values on the domain of w(x,y). By default it is assumed that 
        w has the domain [1,N]-by-[1,N]. 
    rowstep : int
        Integer determining the array row stride (step size) used to generate the wireframe
        plot (see plot_wireframe's documentation for details). Default value is one. 
    colstep : int
        Integer determining the array column stride (step size) used to generate the wireframe
        plot (see plot_wireframe's documentation for details). Default value is one. 
       
    Returns:
    --------
    None 

    Notes:
    ------
    None

    See also:
    ---------
    Matplotlib's plot_wireframe

    """

    # Check the input vector field
    checkinput(u,v)
    (M,N) = u.shape

    # Check the grid
    if (x == None and y != None) or (x != None and y == None):
        print "WARNING: Only x- or y-grid-data specified - switching to default domain [1,N]-by-[1,N]..."
        x,y = makegrid(N,M=M)
    elif x == None:
        x,y = makegrid(N,M=M)
    else:
        checkgrid(x,y)
    if x.shape[0] != u.shape[0] or x.shape[1] != u.shape[1]:
        raise IndexError("Grid and vector field must have the same dimensions!")

    # Sanity checks
    try: rowstep/2.0
    except: raise TypeError("rowstep has to be a positive integer!")
    if rowstep < 1: raise ValueError("rowstep has to be >= 1!")
    if rowstep >= u.shape[0]: raise ValueError("rowstep has to be less than u.shape[0]!")
    if np.round(rowstep) != rowstep:
        rowstep = np.round(rowstep)
        print "WARNING: rowstep has to be an integer - using round(rowstep) = ",rowstep," now..."

    try: colstep/2.0
    except: raise TypeError("colstep has to be a positive integer!")
    if colstep < 1: raise ValueError("colstep has to be >= 1!")
    if colstep >= u.shape[1]: raise ValueError("colstep has to be less than u.shape[1]!")
    if np.round(colstep) != colstep:
        colstep = np.round(colstep)
        print "WARNING: colstep has to be an integer - using round(colstep) = ",colstep," now..."

    # Draw the deformed grid
    ax = plt.gca(projection='3d')
    ax.plot_wireframe(x,y,u+v,color="black",rstride=rowstep,cstride=colstep)
    ax.set_axis_off()
    plt.draw()

    return

def checkinput(u,v):
    """
    Perform sanity checks on the input vector field w=(u,v). 
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
        
    return

def checkgrid(x,y):
    """
    Perform sanity checks on the grid (x,y). 
    """

    # Sanity checks
    if type(x).__name__ != "ndarray":
        raise TypeError("x has to be a NumPy ndarray!")
    else:
        if len(x.shape) > 2: raise ValueError("x has to be 2-dimensional!")
        try: x.shape[1]
        except: raise ValueError("x has to be a matrix!")
        if np.isnan(x).max() == True or np.isinf(x).max() == True:
            raise ValueError("x must not contain NaNs or Infs!")

    if type(y).__name__ != "ndarray":
        raise TypeError("y has to be a NumPy ndarray!")
    else:
        if len(y.shape) > 2: raise ValueError("y has to be 2-dimensional!")
        try: y.shape[1]
        except: raise ValueError("y has to be a matrix!")
        if np.isnan(y).max() == True or np.isinf(y).max() == True:
            raise ValueError("y must not contain NaNs or Infs!")

    if u.shape[0] != y.shape[0] or u.shape[1] != y.shape[1]:
        raise IndexError("u and y must have the same dimensions!")
        
    return
