# chambolle.py - Solve nonlinear TV-denoising using Chambolle's projection algorithm
# 
# Author: Stefan Fuertinger [stefan.fuertinger@gmx.at]
# August 22 2012

from __future__ import division

import numpy as np
import scipy as sp

# If the mypy package is present, import fidop2d from there, otherwise difftools.py has to be in the same
# directory as this file
try:
    from mypy.difftools import fidop2d
except:
    from difftools import fidop2d

##########################################################################################
def chambolle(ut, Dx=None, Dy=None, mu=1.0e-5, dt=0.25, itmax=10000, tol=1.0e-3):
    """
    Solve the nonlinear TV-denoising problem using Chambolle's projection algorithm 

    Parameters
    ----------
    ut : NumPy 2darray
        Raw grey-scale image to denoise. Note that `ut` has to be square!
    Dx : NumPy/SciPy matrix
        Discrete derivative operator in direction `x` (forward differences are recommended). 
        Note that if `ut` is `N`-by-`N` then `Dx` has to be `N**2`-by-`N**2`!
    Dy : NumPy/SciPy matrix
        Discrete derivative operator in direction `y` (forward differences are recommended). 
        Note that if `ut` is `N`-by-`N` then `Dy` has to be `N**2`-by-`N**2`!
    mu : non-negative float
        Regularization parameter in the TV functional. Note that `mu >= 0` has to hold!
    dt : positive float
        Pseudo time step. Note that `dt` has to satisfy `0 < dt < 1`. 
    itmax : positive integer
        Maximal number of Chambolle iterations. Note that `itmax` has to be > 0!
    tol : positive float
        Error tolerance in Chambolle iterations. Note that `tol` has to satisfy `0 < tol << 1`!

    Returns
    -------
    u : NumPy 2darray
        Denoised Chambolle-image. Has the same dimension as the input image `ut`. 
    p : NumPy 3darray
        Dual variable ("Chambolle-edge-set"). If `ut` is `N`-by-`N` then `p` is `2`-by-`N`-by-`N`, i.e. 
        a tensor. 

    References
    ----------
    .. [1] A. Chambolle. An algorithm for total variation minimization and applications.
           Journal of Mathematical Imaging and Vision, 20(1-2):89-97, 2004.
    """

    # Sanity checks
    if type(ut).__name__ != "ndarray":
        raise TypeError("ut has to be a (square) NumPy ndarray!")
    else:
        if len(ut.shape) > 2: raise ValueError("ut has to be 2-dimensional!")
        N = ut.shape[0]
        try: M = ut.shape[1]
        except: raise ValueError("ut has to be square!")
        if N!=M: raise ValueError("ut has to be square!")

    if Dx != None:
        if type(Dx).__name__.rfind("matrix") == -1:
            raise TypeError("Dx has to be a SciPy/NumPy matrix!")
        else:
            NN = Dx.shape[0]
            if NN != Dx.shape[1]: raise ValueError("Dx has to be a square matrix!")
            if NN != N**2: raise ValueError("Dx has to be of dimension %s**2 = %s"%(repr(N),repr(N**2)))

    if Dy != None:
        if type(Dy).__name__.rfind("matrix") == -1:
            raise TypeError("Dy has to be a SciPy/NumPy matrix!")
        else:
            NN = Dy.shape[0]
            if NN != Dy.shape[1]: raise ValueError("Dy has to be a square matrix!")
            if NN != N**2: raise ValueError("Dy has to be of dimension %s**2 = %s"%(repr(N),repr(N**2)))
        
    try: mu = float(mu)
    except: raise TypeError("mu has to be a non-negative scalar!")
    if mu < 0: raise ValueError("mu has to be non-negative!")

    try: dt = float(dt)
    except: raise TypeError("dt has to be a positive scalar!")
    if dt <= 0 or dt > 1: raise ValueError("dt has to be 0 < dt < 1!")

    try: itmax = int(itmax)
    except: raise TypeError("itmax has to be a positive scalar!")
    if itmax <= 0: raise ValueError("itmax has to be > 0!")

    try: tol = float(tol)
    except: raise TypeError("tol has to be a positive scalar!")
    if tol <= 0 or tol > 1: raise ValueError("tol has to be 0 < tol << 1!")

    # Now start to actually do something: get squared image dimension
    NN = N**2

    # If code was called without operators generate them
    if Dx == None or Dy == None:
        Dx,Dy = fidop2d(N,'xy','f')

    # Convert the image to a vector
    ut = ut.flatten(1)

    # Initialize necessary variables
    ux = np.zeros((NN,))
    uy = np.zeros((NN,))
    gu = np.zeros((NN,))
    dn = np.zeros((NN,))
    dp = np.zeros((NN,))
    du = np.zeros((NN,))
    u  = np.zeros((NN,))
    p  = np.zeros((2,N,N))

    # Iteration parameters
    relerr = 2*tol
    res    = 0
    relres = 0
    it     = 0
    ep1    = 1.0e-8
    nt     = np.linalg.norm(ut)

    # Start Chambolle Iteration
    while relerr > tol and it <= itmax:

        # Update iteration counter
        it += 1

        # grad u
        ux = Dx*u
        uy = Dy*u

        # |grad u|
        gu = np.sqrt(ux**2 + uy**2)

        # p = (1 + dt grad u)p / (1 + dt |grad u|)
        dn     = 1 + dt*gu
        p[0,:] = (p[0,:] + dt*ux.reshape(N,N,order="F"))/dn.reshape(N,N,order="F")
        p[1,:] = (p[1,:] + dt*uy.reshape(N,N,order="F"))/dn.reshape(N,N,order="F")

        # div p
        dp = -Dx.T*p[0,:].flatten(1) - Dy.T*p[1,:].flatten(1)

        # residual
        res = np.sum(np.abs(mu*(u - dp) - ut))

        # u = ut + mu * div p
        du = u.copy()
        u  = ut/mu + dp
        du = u - du

        # Relative error
        relerr = np.linalg.norm(u + ep1)
        relerr = np.linalg.norm(du)/relerr 
        relres = res/nt

        # Print output to prompt
        print "it = %s, |du|/|u|= %s, |res|/|ut| = %s"%(repr(it),repr(relerr),repr(relres))

        # Modify stopping criterion
        relerr = relres

    # Convert u back to matrix and compute Chambolle image
    u = u.reshape(N,N,order="F")
    u = u*mu

    return u, p
