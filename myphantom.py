# Python version of the Shep-Logan Phantom
from numpy import array, zeros, arange, tile, pi, rot90, cos, sin, nonzero

def myphantom(N):
    """Python implementation of the Shep-Logan Phantom
    Input:  N scalar, image dimension
    Output: p matrix, the phantom-image
    """
    
    # Check if N is a scalar and convert it to float (if N is not a scalar float conversion doesn't work)
    try: N = float(N)
    except TypeError: raise TypeError("N has to be an integer!")

    # If N has nonzero decimal places the user just didn't get it
    if round(N) != N:
        raise ValueError("N has to be an integer")

    shep = array([[  1,   .69,   .92,    0,     0,     0],   
                  [-.8,  .6624, .8740,   0,  -.0184,   0],
                  [-.2,  .1100, .3100,  .22,    0,    -18],
                  [-.2,  .1600, .4100, -.22,    0,     18],
                  [.1,  .2100, .2500,   0,    .35,    0],
                  [.1,  .0460, .0460,   0,    .1,     0],
                  [.1,  .0460, .0460,   0,   -.1,     0],
                  [.1,  .0460, .0230, -.08,  -.605,   0], 
                  [.1,  .0230, .0230,   0,   -.606,   0],
                  [.1,  .0230, .0460,  .06,  -.605,   0]])

    p = zeros((N,N));

    # Here we need float NOT integer division!
    xax = (arange(0,N)-(N-1)/2.0)/((N-1)/2.0)

    xg = tile(xax, (N, 1))

    for k in xrange(shep.shape[0]):
        asq = shep[k,1]**2
        bsq = shep[k,2]**2
        phi = shep[k,5]*pi/180.0
        x0 = shep[k,3]
        y0 = shep[k,4]
        A = shep[k,0]
        x=xg-x0; y=rot90(xg)-y0
        cosp = cos(phi); sinp = sin(phi)
        idx=nonzero((((x*cosp + y*sinp)**2)/asq + ((y*cosp - x*sinp)**2)/bsq) <= 1)
        p[idx] = p[idx] + A
   
    return p
