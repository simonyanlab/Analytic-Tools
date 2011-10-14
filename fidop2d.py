# fidop2d.py - computes 2D finite difference operators

# External imports
from scipy.sparse import spdiags, eye, kron 
from numpy import ones, r_, array

def fidop2d(N, drt='x', fds='c'):
    """FIDOP2D computes 2D finite difference operators
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
    Dx,Dy = fidop2d(4,fds='f')
    Dy = fidop2d(N,'y')

    ------------------------
    Some mathematical notes:

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
    e = ones((1,N));

    # Building blocks for the operators
    if fds=='c':
        # Centered differences
        A = spdiags(r_[-e,e],array([-1,1]),N,N,format='lil');
        A[0,0] = -2.0; A[0,1] = 2.0; A[N-1,N-2] = -2.0; A[N-1,N-1] = 2.0;
        A = 0.5*A;
        A = A.tocsr()
    elif fds=='f':
        # Forward differences
        A = spdiags(r_[-e,e],array([0,1]),N,N,format='lil');
        A[N-1,:] = 0.0;
        A = A.tocsr()
    else:
        # Backward differences
        A = spdiags(r_[-e,e],array([-1,0]),N,N,format='lil');
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
