# myvec.py - Plotting tools for vector fields
# 
# Author: Stefan Fuertinger [stefan.fuertinger@gmx.at]
# June 13 2012

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

##########################################################################################
def myquiv(u,v):
    """
    Plots a 2D vector field using "sane" defaults for Matplotlib's `quiver`.

    Parameters
    ----------
    u : NumPy 2darray
        `x` components of the vector field `w(x,y) = (u(x,y),v(x,y))`. Note that `u` has to be 
        a 2D array of the same dimension as `v`. 
    v : NumPy 2darray
        `y` components of the vector field `w(x,y) = (u(x,y),v(x,y))`. Note that `v` has to be 
        a 2D array of the same dimension as `u`. 
       
    Returns
    -------
    Nothing : None 

    See also
    --------
    quiver : in the `Matplotlib Example Code Repository <http://matplotlib.org/examples/pylab_examples/quiver_demo.html>`_
    """

    # Check the input vector field
    checkinput(u,v)

    # Now do something
    N  = u.shape[0]
    dN = min(N,16)
    plt.quiver(v[N-1:0:(-N/dN),0:N:(N/dN)],u[N-1:0:(-N/dN),0:N:(N/dN)],color="k")
    plt.axis("image")
    plt.axis("off")
    plt.draw()

    return

##########################################################################################
def makegrid(N,M=None,xmin=1,xmax=None,ymin=1,ymax=None):
    """
    Create an `M`-by-`N` grid on the 2D-domain `[xmin,xmax]`-by-`[ymin,ymax]`

    Parameters
    ----------
    N : int
        The number of grid-points in vertical (i.e. `y`-) direction
    M : int 
        The number of grid-points in horizontal (i.e. `x`-) direction. By default `M = N`
    xmin : float
        The left boundary of the (rectangular) domain. By default `xmin = 1` 
    xmax : float
        The right boundary of the (rectangular) domain. By default `xmax = N`
    ymin : float
        The lower boundary of the (rectangular) domain. By default `ymin = 1`
    ymax : float
        The upper boundary of the (rectangular) domain. By default `ymax = N`
    
    Returns
    -------
    x : NumPy 2darray
        2D grid array of `x`-values on the domain `[xmin,xmax]`-by-`[ymin,ymax]`
    y : NumPy 2darray
        2D grid array of `y`-values on the domain `[xmin,xmax]`-by-`[ymin,ymax]`

    Examples
    --------
    The call
 
    >>> x,y = makegrid(N)

    creates a square `[1,N]`-by-`[1,N]` grid given by

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

    See also
    --------
    meshgrid : NumPy's `meshgrid <http://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html>`_
    gengrid : found in the module `makeimg <makeimg.gengrid.html>`_
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

##########################################################################################
def mygrid(u,v,x=None,y=None,rowstep=16,colstep=16,interpolation="lanczos"):
    """
    Plot a 2D vector field as deformed grid on a 2D lattice

    Parameters
    ----------
    u : NumPy 2darray
        `x` components of the vector field `w(x,y) = (u(x,y),v(x,y))`. Note that `u` has to be 
        a 2D array of the same dimension as `v`. 
    v : NumPy 2darray
        `y` components of the vector field `w(x,y) = (u(x,y),v(x,y))`. Note that `v` has to be 
        a 2D array of the same dimension as `u`. 
    x : NumPy 2darray
        2D grid array of `x`-values on the domain of `w(x,y)`. By default it is assumed that 
        `w` is defined on `[1,N]`-by-`[1,N]` 
    y : NumPy 2darray
        2D grid array of `y`-values on the domain of `w(x,y)`. By default it is assumed that 
        `w` is defined on `[1,N]`-by-`[1,N]` 
    rowstep : int
        Array row stride (step size) used to generate the grid. Default value is 16. 
    colstep : int
        Array column stride (step size) used to generate the grid. Default value is 16. 
    interpolation : str
        Interpolation to be used for plotting. Default value is "lanczos". 
        Recommended other values are "bilinear" or "nearest". See Matplotlib's 
        `imshow`-documentation for details. 
       
    Returns
    -------
    Nothing : None 

    See also
    --------
    imshow : in the `Matplotlib documentation <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow>`_
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
        checkgrid(u,x,y)
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

    if str(interpolation) != interpolation:
        raise TypeError("interpolation has to be a string!")

    # Create lattice
    wires = np.ones(u.shape)
    wires[0:-1:rowstep,:] = 0
    wires[:,0:-1:colstep] = 0

    # Apply vector field
    zipyx  = zip(y.flatten(1),x.flatten(1))
    wiresr = griddata(zipyx,wires.flatten(1),(y+v,x+u),method="linear",fill_value=1)
    
    # Plot it
    plt.imshow(wiresr,cmap="gray",interpolation=interpolation)
    plt.axis("image")
    plt.axis("off")
    plt.draw()

    return

##########################################################################################
def mywire(u,v,x=None,y=None,rowstep=1,colstep=1):
    """
    Plot a 2D vector field as 3D wire-frame using Matplotlib's `plot_wireframe`

    Parameters
    ----------
    u : NumPy 2darray
        `x` components of the vector field `w(x,y) = (u(x,y),v(x,y))`. Note that `u` has to be 
        a 2D array of the same dimension as `v`. 
    v : NumPy 2darray
        `y` components of the vector field `w(x,y) = (u(x,y),v(x,y))`. Note that `v` has to be 
        a 2D array of the same dimension as `u`. 
    x : NumPy 2darray
        2D grid array of `x`-values on the domain of `w(x,y)`. By default it is assumed that 
        `w` is defined on `[1,N]`-by-`[1,N]` 
    y : NumPy 2darray
        2D grid array of `y`-values on the domain of `w(x,y)`. By default it is assumed that 
        `w` is defined on `[1,N]`-by-`[1,N]` 
    rowstep : int
        Array row stride (step size) used to generate the wire-frame plot (see `plot_wireframe`'s 
        documentation for details). Default value is 1. 
    colstep : int
        Array column stride (step size) used to generate the wire-frame plot (see `plot_wireframe`'s 
        documentation for details). Default value is 1. 
       
    Returns
    -------
    Nothing : None 

    See also
    --------
    plot_wireframe : in the `Matplotlib documentation <http://matplotlib.org/mpl_toolkits/mplot3d/api.html#mpl_toolkits.mplot3d.axes3d.Axes3D.plot_wireframe>`_
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
        checkgrid(u,x,y)
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

##########################################################################################
def checkinput(u,v):
    """
    Perform sanity checks on the input vector field
    """

    # Sanity checks
    if type(u).__name__ != "ndarray":
        raise TypeError("u has to be a NumPy 2darray!")
    else:
        if len(u.shape) > 2: raise ValueError("u has to be 2-dimensional!")
        try: u.shape[1]
        except: raise ValueError("u has to be a matrix!")
        if np.isnan(u).max() == True or np.isinf(u).max() == True or np.isreal(u).min() == False:
            raise ValueError("u must be real and must not contain NaNs or Infs!")

    if type(v).__name__ != "ndarray":
        raise TypeError("v has to be a NumPy 2darray!")
    else:
        if len(v.shape) > 2: raise ValueError("v has to be 2-dimensional!")
        try: v.shape[1]
        except: raise ValueError("v has to be a matrix!")
        if np.isnan(v).max() == True or np.isinf(v).max() == True or np.isreal(v).min() == False:
            raise ValueError("v must be real and must not contain NaNs or Infs!")

    if u.shape[0] != v.shape[0] or u.shape[1] != v.shape[1]:
        raise IndexError("u and v must have the same dimensions!")
        
    return

##########################################################################################
def checkgrid(u,x,y):
    """
    Perform sanity checks on the grid
    """

    # Sanity checks
    if type(x).__name__ != "ndarray":
        raise TypeError("x has to be a NumPy 2darray!")
    else:
        if len(x.shape) > 2: raise ValueError("x has to be 2-dimensional!")
        try: x.shape[1]
        except: raise ValueError("x has to be a matrix!")
        if np.isnan(x).max() == True or np.isinf(x).max() == True or np.isreal(x).min() == False:
            raise ValueError("x must be real and must not contain NaNs or Infs!")

    if type(y).__name__ != "ndarray":
        raise TypeError("y has to be a NumPy 2darray!")
    else:
        if len(y.shape) > 2: raise ValueError("y has to be 2-dimensional!")
        try: y.shape[1]
        except: raise ValueError("y has to be a matrix!")
        if np.isnan(y).max() == True or np.isinf(y).max() == True or np.isreal(y).min() == False:
            raise ValueError("y must be real and must not contain NaNs or Infs!")

    if u.shape[0] != y.shape[0] or u.shape[1] != y.shape[1]:
        raise IndexError("u and y must have the same dimensions!")
        
    return
