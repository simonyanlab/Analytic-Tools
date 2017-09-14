# tools1d.py - Stuff that makes one-dimensional life easier
# 
# Author: Stefan Fuertinger
# Mai 10 2012

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.sparse import linalg, spdiags, eye

##########################################################################################
def makegrid(N,xmin=0,xmax=1):
    """
    MAKEGRID generates a grid on unit interval
    """

    # Make 1D-grid
    h    = (xmax - xmin)/(N-1);
    x1   = xmin + h*np.arange(0,N)

    return x1

##########################################################################################
def getimg(x1,img="2spikes",ns=0.0):
    """
    GETIMG makes an artificial image
    """

    # Extract dimension
    N = x1.size
    xmin = x1[0]
    xmax = x1[-1]

    # Depending on the given img construct image
    if img == "2spikes":
        It = ((x1 - xmin)/(xmax - xmin));
        It[1:(2*N/16)+1] = 0;
        It[(2*N/16+1):(7*N/16)+1] = It[(2*N/16+1):(7*N/16)+1] /\
            It[(2*N/16+1):(7*N/16)+1].max()
        It[(7*N/16+1):(8*N/16)+1] = 0;
        It[(8*N/16+1):(13*N/16)+1] = It[(8*N/16+1):(13*N/16)+1] /\
            It[(8*N/16+1):(13*N/16)+1].max()
        It[(13*N/16+1):(14*N/16)+1] = 0
        It[(14*N/16+1):N] = 0;
        It = It/It.max()
    elif img == "3spikes":
        It = ((x1 - xmin)/(xmax - xmin));
        It[1:(2*N/16)+1] = 0;
        It[(2*N/16+1):(7*N/16)+1] = It[(2*N/16+1):(7*N/16)+1] /\
            It[(2*N/16+1):(7*N/16)+1].max()
        It[(7*N/16+1):(8*N/16)+1] = 0;
        It[(8*N/16+1):(10*N/16)+1] = It[(8*N/16+1):(10*N/16)+1] /\
            It[(8*N/16+1):(10*N/16)+1].max()
        It[(10*N/16+1):(13*N/16)+1] = 0;
        It[(13*N/16+1):(14*N/16)+1] = It[(13*N/16+1):(14*N/16)+1] /\
            It[(13*N/16+1):(14*N/16)+1].max()
        It[(14*N/16+1):N] = 0;
    elif img == "4spikes":
        It = ((x1 - xmin)/(xmax - xmin))**2
        m = 8
        for j in xrange(1,m,2):
            It[((j-1)*N/m+1):(j*N/m)+1] = 0
            for j in xrange(2,m,2):
                It[((j-1)*N/m+1):(j*N/m)+1] = It[((j-1)*N/m+1):(j*N/m)+1] /\
                    It[((j-1)*N/m+1):(j*N/m)+1].max()
    else: raise ValueError("Unrecognized option %s for img!"%img)

    # Fix random number generator and add noise to the image
    np.random.seed(0)
    It = It + ns*np.random.randn(N)

    # Normalize It
    It = It/It.max()

    return It
    
##########################################################################################
def getchi(I,koff=3):

    # Make an educated guess for chi given an image I
    N   = I.size
    chi = np.zeros((N,))
    chi[(2*N/16+1+koff):(7*N/16)+1-koff] = 1
    chi[(8*N/16+1)+koff:(13*N/16)+1-koff] = 1
    
    return chi

##########################################################################################
def makeopers(N):
    """
    MAKEOPERS builds finite difference operators in 1D 
    """

    # Building block for differential operators
    e = np.ones((1,N));
    h = 1.0
    
    # Forward differences
    Dx        = spdiags(np.r_[-e,e],np.array([0,1]),N,N,format='lil');
    Dx[N-1,:] = 0.0;
    Dx        = 1/h*Dx.tocsr()
    
    # Second order
    Dxx = -1/(h**2)*Dx.T*Dx

    return Dx, Dxx
