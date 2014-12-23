# difftools.py - Several useful utilities for numerical differentiation
# 
# Author: Stefan Fuertinger [stefan.fuertinger@gmx.at]
# June 13 2012

from __future__ import division
import numpy as np
from scipy.sparse import spdiags, eye, kron 

##########################################################################################
def fidop2d(N, drt='xy', fds='c'):
    """
    Compute 2D finite difference operators

    Parameters
    ----------
    N : integer
        Positive integer holding the grid dimension (has to be square, i.e.
        `N`-by-`N`!)
    drt : string
        String determining the direction of the discrete derivatives. 
        Can be either 'x', 'y' or 'xy'. 
    fds : string
        String determining the employed finite difference scheme. Can be 
        either 'c' for centered, 'f' for forward or 'b' for backward. 
       
    Returns
    -------
    D : SciPy sparse matrix/matrices
        Depending on the given input one or two sparse matrices 
        corresponding to the wanted discrete derivative operators are 
        returned. 

    Examples
    --------
    The command `Dx,Dy = fidop2d(N)` returns the sparse `N**2`-by-`N**2` 
    matrices `Dx` and `Dy` such that `Dh` defined by

    >>> import numpy as np
    >>> Dh = np.hstack([Dx,Dy])

    is the discrete gradient operator in lexicographic 
    ordering for a function `F` given on a grid of size `N`-by-`N`.
    The matrix `Dx` is a discrete approximation of the first-order derivative operator 
    in `x`-direction, Analogously, `Dy` discretizes the first order derivative operator 
    in `y`-direction. The spacing between points in each direction is assumed to be one 
    (i.e. the step size `h = 1`). 
    
    The command `D = fidop2d(N,drt)` returns the sparse `N**2`-by-`N**2` matrix `D` corresponding
    to the discrete derivative operator in direction `drt`
    in lexicographic ordering. The string `drt` has to be either 'x' (default)
    , 'y' or 'xy'. Note: only 'xy' will return two matrices, namely `Dx`, `Dy`. 

    The command `D = fidop2d(N,drt,fds)` returns sparse the `N**2`-by-`N**2` matrix `D` corresponding
    to the discrete derivative operator in direction `drt`
    in lexicographic ordering using the finite difference scheme `fds`. 
    The string `fds` can either be 'c' (centered differences, in cells, default), 
    'f' (forward differences, in interfaces) or 'b' (backward differences, in interfaces). 

    To discretize first order differential operators on a 4-by-4 grid in both `x`- and `y`-directions 
    using forward differences use

    >>> Dx,Dy = fidop2d(4,fds='f')

    To get only the discrete first order differential operator in `y`-direction on a grid of size `N` use

    >>> Dy = fidop2d(N,'y')

    Notes
    -----
    For clarity here some comments on the rationale behind the code. 

    *The Laplacian*

    If `Dxf` and `Dyf` are computed using forward differences, i.e.

    >>> Dxf,Dyf = fidop2d(N,'xy','f')

    then the discrete Laplacian `Lh` based on centered differences is
    obtained by 

    >>> Lh = -(Dxf.T*Dxf + Dyf.T*Dyf)

    or analogously for backward differences, 

    >>> Dxb,Dyb = fidop2d(N,'xy','b')

    then

    >>> Lh = Dxb.T*Dxb + Dyb.T*Dyb
    
    *The Divergence*

    Note adjoints: for :math:`p` in 
    :math:`H_0(\\mathrm{div}):=\\{p \\in L^2(\\Omega)` | div :math:`(p) \\in L^2(\\Omega), pn = 0` 
    on the boundary of :math:`\\Omega\\}` we have

    .. math:: 

       \\begin{align}
       \\int_{\\Omega} \\nabla u\\cdot p dx &=  \\int_{\\Omega} u p \\cdot  n dS - \\int_{\\Omega} u \\textrm{ div }(p) dx\\\\
                                     & = 0 -\\int_{\\Omega} u \\textrm{ div }(p) dx
       \\end{align}

    or in short :math:`\\textrm{div}^\\ast = -\\nabla`. Hence for a spatial discretization take 
    `Div_h ~ -Dh.T` as approximation for the divergence. 
    """

    # Check correctness of input
    try: tmp = int(N) != N
    except TypeError: raise TypeError("N has to be a positive integer")
    if (tmp): raise ValueError("N has to be a positive integer")
    if N <= 0: raise ValueError("N has to be a positive integer")

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

##########################################################################################
def myff2n(n):
    """
    Give factor settings for a two-level full factorial design

    Parameters
    ----------
    n : integer
        Number of factors
       
    Returns
    -------
    dff2 : NumPy 2darray
        An `m`-by-`n` array of factor settings 

    Examples    
    --------
    >>> dFF2 = ff2n(3)
    >>> dFF2
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  1.],
               [ 0.,  1.,  0.],
               [ 0.,  1.,  1.],
               [ 1.,  0.,  0.],
               [ 1.,  0.,  1.],
               [ 1.,  1.,  0.],
               [ 1.,  1.,  1.]])
    """

    # Check correctness of input
    try: tmp = int(n) != n
    except TypeError: raise TypeError("n has to be a positive integer")
    if tmp: raise ValueError("n has to be a positive integer")
    if n <= 0: raise ValueError("n has to be a positive integer")

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
