# normalize.py - rescale input image I

# Imports
from __future__ import division

from numpy import absolute, nonzero, finfo

def normalize(I,a=0,b=1):
    """NORMALIZE rescales a numpy ndarray
    In = NORMALIZE(I) returns the [0,1]-normalized version of the 
    numpy ndarray I. 

    In = NORMALIZE(I,a=val1,b=val2) for some scalars a<b returns 
    the numpy ndarray In scaled so that a = In.min() and 
    b = In.max(). 

    Examples:

    I = array([[-1,.2],[100,0]])
    In = normalize(I,a=-10,b=12)
    In 
    array([[-10.        ,  -9.73861386],
           [ 12.        , -10.        ]])
    """

    # Ensure that I is a numpy-ndarray
    try: 
        if I.size == 1:
            raise ValueError('I has to be a numpy ndarray of size > 1!')
    except TypeError: raise TypeError('I has to be a numpy ndarray!')

    # If normalization bounds are user specified, check them
    try:
        if b <= a:
            raise ValueError('a has to be strictly smaller than b!')
        elif absolute(a - b) < finfo(float).eps:
            raise ValueError('|a-b|<eps, no normalization possible')
    except TypeError: raise TypeError('a and b have to be scalars satisfying a < b!')

    # Get min and max of I
    Imin   = I.min()
    Imax   = I.max()

    # If min and max values of I are identical do nothing, if they differ close to machine precision abort
    if Imin == Imax:
        return I
    elif absolute(Imin - Imax) < finfo(float).eps:
        raise ValueError('|Imin-Imax|<eps, no normalization possible')

    # Make a local copy of I
    I = I.copy()

    # Here the normalization is done
    I = (I - Imin)*(b - a)/(Imax - Imin) + a

    # Return normalized array
    return I
