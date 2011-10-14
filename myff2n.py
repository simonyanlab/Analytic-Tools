# myff2n.py - computes the two-level full factorial design matrix

from __future__ import division

from numpy import zeros, arange

def myff2n(n):
    """From the MATLAB(TM) help:
    dFF2 = ff2n(n) gives factor settings dFF2 for a two-level full factorial 
    design with n factors. dFF2 is m-by-n, where m is the number of treatments 
    in the full-factorial design. Each row of dFF2 corresponds to a single 
    treatment. Each column contains the settings for a single factor, 
    with values of 0 and 1 for the two levels.

    Examples:    

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
    """

    # Check correctness of input
    try: 
        if int(n) != n:
            raise ValueError("n has to be a positive integer")
        elif n <= 0:
            raise ValueError("n has to be a positive integer")
    except TypeError: raise TypeError("n has to be a positive integer")

    # Output array x has 2^n rows 
    rows = 2**n
    ncycles = rows
    x = zeros((rows,n))

    # This is adapted from the MATLAB source file ff2n.m
    for k in xrange(0,n):
        settings = arange(0,2)
        settings.shape = (1,2)
        ncycles = ncycles/2.0
        nreps = rows/(2*ncycles)
        settings = settings[zeros((1,nreps)).astype(int),:]
        settings = settings.flatten(1)
        settings.shape = (settings.size,1)
        settings = settings[:,zeros((1,ncycles)).astype(int)]
        x[:,n-k-1] = settings.flatten(1)

    return x
