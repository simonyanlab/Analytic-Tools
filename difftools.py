# difftools.py - Several useful utilities for numerical differentiation
# 
# (C) Stefan Fuertinger
# Juni 13 2012

from __future__ import division
import numpy as np
from scipy.sparse import spdiags, eye, kron 

def fidop2d(N, drt='x', fds='c'):
    """
    FIDOP2D computes 2D finite difference operators

    Parameters:
    -----------
    N : integer
        Positive integer holding the grid dimension (has to be square, i.e.
        N-by-N!)
    drt : str
        String determining the direction of the discrete derivatives. 
        Can be either 'x', 'y' or 'xy'. 
    fds : str
        String determining the employed finite difference scheme. Can be 
        either 'c' for centered, 'f' for forward or 'b' for backward. 
       
    Returns
    -------
    D : SciPy sparse matrix/matrices
        Depending on the given input one or two sparse matrices 
        corresponding to the wanted discrete derivative operators are 
        returned. 

    Notes
    -----
    Dx,Dy = FIDOP2D(N) returns the sparse N^2-by-N^2 matrices Dx and Dy such that
    Dh = [Dx,Dy] is the discrete gradient operator in lexicographic 
    ordering for a function F given on a grid of size NxN.
    Dx corresponds to the discrete derivative operator of first order 
    in x-direction, Dy corresponds to the discrete derivative operator 
    of first order in y-direction. The spacing between points
    in each direction is assumed to be one (i.e. the step size h = 1). 
  
    D = FIDOP2D(N,drt) returns the sparse N^2-by-N^2 matrix D corresponding
    to the discrete derivative operator in direction drt
    in lexicographic ordering. The string drt has to be either 'x' (DEFAULT)
    , 'y' or 'xy'. 
    Note: only 'xy' will return two matrices, namely Dx, Dy. 

    D = FIDOP2D(N,drt,fds) returns sparse the N^2-by-N^2 matrix D corresponding
    to the discrete derivative operator in direction drt
    in lexicographic ordering using the finite difference scheme fds. 
    The char fds can either be 'c' (centered differences, in cells, DEFAULT), 
    'f' (forward differences, in interfaces) or 'b' (backward differences, in interfaces). 

    Examples:
    ---------
    Dx,Dy = fidop2d(4,fds='f')
    Dy = fidop2d(N,'y')

    Some mathematical notes:
    ------------------------
    For clarity here some comments on the rationale behind the code. 

    The Laplacian:
    --------------
    If Dxf,Dyf are computed using forward differences, i.e.
        Dxf,Dyf = FIDOP2D(N,'xy','f')
    Then the discrete Laplacian Lh as if using centered differences is
    obtained by 
        Lh = -(Dxf.T*Dxf + Dyf.T*Dyf)
    or if using backward differences, i.e.
        Dxb,Dyb = FIDOP2D(N,'xy','b')
    then
        Lh = Dxb.T*Dxb + Dyb.T*Dyb
    
    The Divergence:
    ---------------
    Note adjoints: For p in H_0(div):={p in L^2(Om)|div(p) in L^2(Om), p*n = 0 on the boundary of Om}
    we have
        \int_Om grad(u)*p dx =  \int_Om u*p*n dS - \int_Om u*div(p) dx
                                \-------------/
                                     =0
                             = -\int_Om u*div(p) dx
    or in short
        adjoint(div) = -grad
    Hence for a spatial discretization take
        Div_h ~ -Dh.T
    as approximation for the Divergence. 

    See also
    --------
    None
    """

    # Check correctness of input
    try: 
        if int(N) != N:
            raise ValueError("N has to be a positive integer")
        elif N <= 0:
            raise ValueError("N has to be a positive integer")
    except TypeError: raise TypeError("N has to be a positive integer")

    if drt != 'x' and drt != 'y' and drt != 'xy':
        raise ValueError("drt has to be x, y or xy")
    elif drt == 'x' or drt == 'y':
        myout = 1
    else:
        myout = 2

    if fds != 'c' and fds != 'b' and fds != 'f':
        raise ValueError("fds has to be c (centered), b (backward) or f (forward)")

    # Initialize vector of ones needed to build matrices
    e = np.ones((1,N));

    # Building blocks for the operators
    if fds=='c':
        # Centered differences
        A = spdiags(np.r_[-e,e],np.array([-1,1]),N,N,format='lil');
        A[0,0] = -2.0; A[0,1] = 2.0; A[N-1,N-2] = -2.0; A[N-1,N-1] = 2.0;
        A = 0.5*A;
        A = A.tocsr()
    elif fds=='f':
        # Forward differences
        A = spdiags(np.r_[-e,e],np.array([0,1]),N,N,format='lil');
        A[N-1,:] = 0.0;
        A = A.tocsr()
    else:
        # Backward differences
        A = spdiags(np.r_[-e,e],np.array([-1,0]),N,N,format='lil');
        A[0,:] = 0.0;
        A = A.tocsr()

    # Second building block needed: the identity
    IN = eye(N,N,format='csr');

    # Compute output matrix/matrices
    if myout==1:
        # Only one output: determine direction
        if drt=='x':
            # x-direction
            return kron(IN,A,format='csr')
        else:
            # y-direction
            return kron(A,IN,format='csr')
    else:
        # Two outputs: order of directions is always d/dx,d/dy]
        return kron(IN,A,format='csr'), kron(A,IN,format='csr')

def myff2n(n):
    """
    MYFF2N gives factor settings dFF2 for a two-level full factorial 

    Parameters:
    -----------
    n : integer
        Positive integer holding the number of factors (i.e. n >= 1)
       
    Returns
    -------
    dff2 : NumPy ndarray
        m-by-n array holding the factor settings 

    Notes
    -----
    From the MATLAB(TM) help:
    dFF2 = ff2n(n) gives factor settings dFF2 for a two-level full factorial 
    design with n factors. dFF2 is m-by-n, where m is the number of treatments 
    in the full-factorial design. Each row of dFF2 corresponds to a single 
    treatment. Each column contains the settings for a single factor, 
    with values of 0 and 1 for the two levels.

    Examples:    
    ---------

    dFF2 = ff2n(3)

    dFF2 =
       0   0   0

       0   0   1

       0   1   0

       0   1   1

       1   0   0

       1   0   1

       1   1   0

       1   1   1
    See also
    --------
    None
    """

    # Check correctness of input
    try: 
        if int(n) != n:
            raise ValueError("n has to be a positive integer")
        elif n <= 0:
            raise ValueError("n has to be a positive integer")
    except TypeError: raise TypeError("n has to be a positive integer")

    # Output array dff2 has 2^n rows 
    rows = 2**n
    ncycles = rows
    dff2 = np.zeros((rows,n))

    # This is adapted from the MATLAB source file ff2n.m
    for k in xrange(0,n):
        settings = np.arange(0,2)
        settings.shape = (1,2)
        ncycles = ncycles/2.0
        nreps = rows/(2*ncycles)
        settings = settings[np.zeros((1,nreps)).astype(int),:]
        settings = settings.flatten(1)
        settings.shape = (settings.size,1)
        settings = settings[:,np.zeros((1,ncycles)).astype(int)]
        dff2[:,n-k-1] = settings.flatten(1)

    return dff2

def mygrad(F):
    """
    MYGRAD computes the numerical gradient of a 2D function using central differences

    Parameters:
    -----------
    F: NumPy ndarray
       A function defined on a M-by-N grid, i.e. a two-dimensional NumPy array

    Returns:
    --------
    Fx: NumPy ndarray
       Array of the same dimension as F containing the x-derivatives of F
    Fy: NumPy ndarray
       Array of the same dimension as F containing the y-derivatives of F

    Notes:
    ------
    In contrast to MATLAB Fx contains derivatives along rows and Fy holds
    the derivatives along columns. 

    See also:
    ---------
    MATLAB's gradient function 
    http://www.mathworks.de/help/techdoc/ref/gradient.html
    """

    # Sanity checks
    if type(F).__name__ != "ndarray":
        raise TypeError("F has to be a NumPy ndarray!")
    else:
        if len(F.shape) > 2: raise ValueError("F has to be 2-dimensional!")
        try: F.shape[1]
        except: raise ValueError("F has to be 2-dimensional!")
        if np.isnan(F).max() == True or np.isinf(F).max() == True:
            raise ValueError("F must not contain NaNs or Infs!")

    # Now do something
    M = F.shape[0]
    N = F.shape[1]

    # Allocate space for output
    Fx = np.zeros((M,N))
    Fy = np.zeros((M,N))

    # Compute x-derivatives
    Fx[0,:]     = F[1,:]   - F[0,:]
    Fx[1:M-1,:] = F[2:M,:] - F[0:M-2,:]
    Fx[-1,:]    = F[-1,:]  - F[-2,:]

    # Compute y-derivatives
    Fy[:,0]     = F[:,1]   - F[:,0]
    Fy[:,1:N-1] = F[:,2:N] - F[:,0:N-2]
    Fy[:,-1]    = F[:,-1]  - F[:,-2]

    return Fx, Fy
