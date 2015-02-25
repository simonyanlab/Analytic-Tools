# myat.py - Implementation of an Ambrosio-Tortorelli segmentation
# 
# Author: Stefan Fuertinger [stefan.fuertinger@gmx.at]
# August 22 2012

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.sparse import spdiags, linalg

# If the mypy package is present, import fidop2d from there, otherwise difftools.py has to be 
# in the same directory as this file
try:
    from mypy.difftools import fidop2d
except:
    from difftools import fidop2d

##########################################################################################
def myat(f,ep,nu=1,de=1,la=1,tol=1e-4,itmax=100,iplot=False,Dx=None,Dy=None,Lh=None):
    """
    Solve the Ambrosio--Tortorelli approximation of the Mumford--Shah functional

    Parameters
    ----------
    f : NumPy 2darray
        Raw (noisy) input image to be segmented. Note that `f` has to be square!
    ep : float
        Positive edge-"thickness" parameter in the approximation functional. For `ep -> 0`
        the Ambrosio--Tortorelli approximation Gamma-converges to the Mumford--Shah 
        functional (see Notes below). 
    nu : float
        Positive parameter determining the influence of the Ambrosio--Tortorelli terms in the 
        functional. 
    de : float
        Positive parameter influencing the smoothness regularization term for `u` in the functional. 
    la : float
        Positive parameter weighing the data fidelity term in the functional. 
    tol : float
        Error tolerance for the stopping criterion satisfying `0 < tol << 1`. 
    itmax : int
        Integer, the maximal number of iterations. 
    iplot : bool
        Switch to turn interactive plotting on (`iplot=True`) or off (`iplot=False`)
    Dx : NumPy/SciPy matrix
        Disrecte derivative operator in direction `x` (foward differences are recommended). 
        Note that if `f` is `N`-by-`N` then `Dx` has to be `N**2`-by-`N**2`!
    Dy : NumPy/SciPy matrix
        Discrete derivative operator in direction `y` (forward differences are recommended). 
        Note that if `f` is `N`-by-`N` then `Dy` has to be `N**2`-by-`N**2`!
    Lh : NumPy/SciPy matrix
        Discrete Laplace operator (central differences are recommended). 
        Note that if `f` is `N`-by-`N` then `Dy` has to be `N**2`-by-`N**2`!

    Returns
    -------
    u : NumPy 2darray
        The smoothed version of `f`. 
    v : NumPy 2darray
        The fuzzy edge map of `f`. 

    Notes
    -----
    The Ambrosio--Tortorelli functional [1]_ is given by 

    .. math:: 

       J_{\\varepsilon}[u,v] = \\int_{\\Omega}\\frac{\\nu\\varepsilon}{2}|\\nabla v|^2 + \\frac{\\nu}{2\\varepsilon}(1-v)^2 + \\frac{\\delta}{2} v^2|\\nabla u|^2 + \\frac{\\lambda}{2} (u-f)^2 dx

    The associated Euler--Lagrange equations are

    .. math::

       \\begin{align}
       -\\delta \\mathrm{div}(v^2 \\nabla u) + \\lambda u &= \\lambda f \\\\
       \\frac{\\partial u}{\\partial n} &= 0
       \\end{align}

    and

    .. math::

       \\begin{align}
       -\\nu\\varepsilon\\Delta v + \\frac{\\nu}{\\varepsilon}v + \\delta v|\\nabla u|^2 &= \\frac{\\nu}{\\varepsilon} \\\\
       \\frac{\\partial u}{\\partial n} &= 0
       \\end{align}

    The alternate optimization algorithm is initialized using

    .. math::

       \\begin{align}
       u_0 &= f \\\\
       v_0 &= 1/(1 + \\delta \\frac{\\varepsilon}{\\nu} |\\nabla f|^2)
       \\end{align}

    It can be shown that :math:`J_{\\epsilon}` Gamma-converges to the Mumford--Shah functional 
    for :math:`\\varepsilon \\rightarrow 0` (see, e.g., [2]_). 

    References
    ----------
    .. [1] L. Ambrosio and V.M. Tortorelli. Approximation of functionals depending on
           jumps by elliptic functionals via Gamma-convergence. Communications on Pure and
           Applied Mathematics, 43:999-1036, 1990.

    .. [2] G. Aubert and P. Kornprobst: "Mathematical Problems in Image Processing: Partial Differential 
           Equations and the Calculus of Variations", Springer 2006. 
    """

    # Sanity checks
    if type(f).__name__ != "ndarray":
        raise TypeError("f has to be a (square) NumPy 2darray!")
    else:
        if len(f.shape) > 2: raise ValueError("f has to be 2-dimensional!")
        N = f.shape[0]
        try: M = f.shape[1]
        except: raise ValueError("f has to be square!")
        if N!=M: raise ValueError("f has to be square!")
        if np.isnan(f).max() == True or np.isinf(f).max() == True or np.isreal(f).min() == False:
            raise ValueError("f must be real and must not contain NaNs or Infs!")

    try: ep/2.0
    except: raise TypeError("ep has to be a positive float!")
    if ep < 0: raise ValueError("ep has to be > 0!")

    try: nu/2.0
    except: raise TypeError("nu has to be a positive float!")
    if nu < 0: raise ValueError("nu has to be > 0!")

    try: de/2.0
    except: raise TypeError("de has to be a positive float!")
    if de < 0: raise ValueError("de has to be > 0!")

    try: la/2.0
    except: raise TypeError("la has to be a positive float!")
    if la < 0: raise ValueError("la has to be > 0!")

    try: tol/2.0
    except: raise TypeError("tol has to be a positive integer!")
    if tol > 1: raise ValueError("tol has to be << 1!")

    try: itmax/2.0
    except: raise TypeError("itmax has to be a positive integer!")
    if itmax < 1: raise ValueError("itmax has to be >= 1!")
    if np.round(itmax) != itmax:
        itmax = np.round(itmax)
        print "WARNING: itmax has to be an integer - using round(itmax) = ",itmax," now..."

    msg = "The switch `iplot` has to be Boolean!"
    try:
        bad = (iplot != True and iplot != False)
    except: raise TypeError(msg)
    if bad: raise TypeError(msg)

    if (Dx != None and Dy == None) or (Dx == None and Dy != None):
        print "WARNING: Dx or Dy not provided, switching to default Dx and Dy"
        Dx,Dy = fidop2d(N,'xy','f')
    elif Lh == None and (Dx != None):
        print "WARNING: Dx and Dy given but not Lh - using defaults for Dx and Dy. Lh will be computed as -(Dx.T*Dx + Dy.T*Dy)"
        Dx,Dy = fidop2d(N,'xy','f')
    elif Dx == None and Dy == None:
        Dx,Dy = fidop2d(N,'xy','f')
    else:
        if type(Dx).__name__.rfind("matrix") == -1:
            raise TypeError("Dx has to be a SciPy/Numpy matrix!")
        else:
            NN = Dx.shape[0]
            if NN != Dx.shape[1]: raise ValueError("Dx has to be a square matrix!")
            if NN != N**2: raise ValueError("Dx has to be of dimension %s**2 = %s"%(repr(N),repr(N**2)))

        if type(Dy).__name__.rfind("matrix") == -1:
            raise TypeError("Dy has to be a SciPy/NumPy matrix!")
        else:
            NN = Dy.shape[0]
            if NN != Dy.shape[1]: raise ValueError("Dy has to be a square matrix!")
            if NN != N**2: raise ValueError("Dy has to be of dimension %s**2 = %s"%(repr(N),repr(N**2)))
        
    if Lh != None:
        if type(Lh).__name__.rfind("matrix") == -1:
            raise TypeError("Lh has to be a SciPy/NumPy matrix!")
        else:
            NN = Lh.shape[0]
            if NN != Lh.shape[1]: raise ValueError("Lh has to be a square matrix!")
            if NN != N**2: raise ValueError("Lh has to be of dimension %s**2 = %s"%(repr(N),repr(N**2)))
    else:
        Lh = -(Dx.T*Dx + Dy.T*Dy)

    # Get squared image dimension
    NN = N**2

    # Allocate memory for u and v
    u = np.zeros(f.shape)
    v = np.zeros(f.shape)

    # Initial guess for u
    u = f.copy()
    
    # Initial guess for v
    v = (1 + de*ep/nu*((Dx*f.flatten(1))**2 + (Dy*f.flatten(1))**2))**(-1)

    # Convert u,v and f to vectors
    u = u.flatten(1)
    v = v.flatten(1)
    f = f.flatten(1)

    # Set up plotting stuff if necessary
    if (iplot): fig = getfig(f,de,la,nu,ep)

    # Show initial guess(es)
    if (iplot): showit(fig,u,v)

    # Initialize iteration parameters
    it    = 0;
    rerru = 0
    rerrv = 0
    rerr  = 2*tol
    Jold  = 0.0
    nfo   = 'inc','dec'
    ep1   = 1.0e-6

    # Allocate memory for right-hand-sides, iterates and the constant matrix
    rhsv = nu/ep*np.ones((NN,))
    rhsu = la*f
    Dla  = spdiags(la*np.ones((NN,)),0,NN,NN)
    un   = np.zeros(u.shape)
    vn   = np.zeros(v.shape)
    JN   = np.zeros((NN,))

    # Allocate non-zero structure of varying matrices
    Av = Lh.copy()
    Au = Dx.T*Dx
    Dv = spdiags(np.ones((NN,)),0,NN,NN)

    # The fpi-loop
    while rerr > tol and it <= itmax:

        # Update iteration counter
        it += 1

        # Compute new v
        Av = spdiags(de*((Dx*u)**2 + (Dy*u)**2) + nu/ep,0,NN,NN) - nu*ep*Lh
        vn = linalg.spsolve(Av.tocsr(),rhsv)
        
        # Compute new u
        Dv = spdiags(v**2,0,NN,NN)
        Au = de*Dx.T*Dv*Dx + de*Dy.T*Dv*Dy + Dla
        un = linalg.spsolve(Au.tocsr(),rhsu)

        # Compute value of cost
        JN = nu*ep/2*((Dx*v)**2 + (Dy*v)**2) + nu/(2*ep)*(1-v)**2 \
            + de/2*v**2*((Dx*u)**2 + (Dy*u)**2) + la/2*(u-f)**2
        J  = np.sum(JN)

        # Compute relative errors
        rerru = norm(un - u)/(norm(un + ep1))
        rerrv = norm(vn - v)/(norm(vn + ep1))
        rerr  = max(rerru,rerrv)
    
        # Show info in prompt
        print "it = %s, rerr = %s, J = %s, %s"%(repr(it),repr(rerr),repr(J),nfo[Jold>J])

        # Update iterates and cost
        u    = un.copy()
        v    = vn.copy()
        Jold = J

        # Plot intermediate results every mplot steps
        if (iplot): showit(fig,u,v)

    # Show final results
    if (iplot): showit(fig,u,v)

    # Convert u and v back to images
    u = u.reshape(N,N,order="F")
    v = v.reshape(N,N,order="F")

    return u,v

##########################################################################################
def getfig(f,de,la,nu,ep):
    """
    Set up Figure for interactive plotting
    """

    # Set up figure and assign window- and sup-title
    fig = plt.figure()
    fig.canvas.set_window_title("Ambrosio-Tortorelli")
    fig.suptitle(r'$\delta = %s,\quad \lambda = %s,\quad \nu = %s,\quad \varepsilon = %s$'\
                        %(repr(de),repr(la),repr(nu),repr(ep)), fontsize=14)

    # Get image dimension
    N = np.sqrt(f.shape[0])

    # Plot original image f
    ax = fig.add_subplot(1,3,1)
    plt.sca(ax)
    ax.set_title(r"$f$")
    plt.imshow(f.reshape(N,N,order="F"),interpolation='nearest',cmap='gray')
    plt.draw()

    return fig

##########################################################################################
def showit(fig,u,v):
    """
    Show iteration process
    """

    # Get image dimension
    N = np.sqrt(u.shape[0])

    # Plot u
    ax = fig.add_subplot(1,3,2)
    plt.sca(ax)
    ax.set_title(r"$u$")
    plt.imshow(u.reshape(N,N,order="F"),interpolation='nearest',cmap='gray')
    plt.draw()

    # Plot v
    ax = fig.add_subplot(1,3,3)
    plt.sca(ax)
    ax.set_title(r"$v$")
    plt.imshow(v.reshape(N,N,order="F"),interpolation='nearest',cmap='gray')
    plt.draw()
