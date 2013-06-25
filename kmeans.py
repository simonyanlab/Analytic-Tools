# kmeans.py - Implementation of k-means clustering

from __future__ import division
from numpy import ones, zeros, nonzero, abs, mean, unique, size, sort, argsort, hstack
from numpy.random import rand

def kmeans(u,M):

    itmax = 1000

    U = sort(u.flatten(1))
    L = U.size
    M = min(M,L)
    dU = U[1:L] - U[0:(L-1)]
    dUi = argsort(-dU,axis=-1,kind='mergesort')
    b = hstack((0,sort(dUi[0:(M-1)]),L-1))
    a = zeros((M,))
    for m in xrange(0,M):
        a[m] = (U[b[m]+1] + U[b[m+1]])/2

    # uu  = unique(u)
    # L   = uu.size
    # a   = zeros((M,))
    # LoM = max(L/M,1)

    # # Compute variation in unique(u), sort i for initial guess
    # du = abs(uu[1:L] - uu[0:L-1])
    # if M == 2: figure(112); plot(du)
    # i  = zeros((M,))
    # for m in xrange(0,M):
    #     j     = du.argmax()
    #     du[j] = 0.0
    #     i[m]  = j
    # i.sort()

    # ii        = zeros((M+1,))
    # ii[1:M+1] = i
    # for m in xrange(0,M,2):
    #     a[m] = (uu[ii[m]] + uu[ii[m+1]])/2.0

    # if M == 6:
    #     import pdb
    #     pdb.set_trace()

    # # Old version
    # for m in xrange(0,M):
    #     m1   = round(m*LoM)
    #     m2   = round((m+1)*LoM)
    #     m2   = min(m2,L)
    #     m1   = min(m1,m2)
    #     m1   = min(m1,m2,L-1)
    #     a[m] = uu[m1:m2].mean()

    # # Sledgehammer
    # a = rand(M,)

    # Actual computation
    chi = zeros(u.shape)
    for it in xrange(1,itmax+1):
        avail = ones(u.shape)
        ass   = a.copy()
        for m in xrange(0,M):
            psi = avail.copy()
            for n in xrange(0,M):
                if n != m:
                    psi = psi * (abs(a[m]-u) <= abs(a[n]-u))
            dom = nonzero(psi)
            if size(dom) != 0:
                chi[dom]   = m + 1
                avail[dom] = 0
        for m in xrange(0,M):
            dom = nonzero(chi == (m+1))
            if size(dom) != 0:
                a[m] = u[dom].mean()
            else:
                a[m] = 0
        if (abs(a - ass)).max() == 0: break
    if (it >= itmax): 
        print 'Maximum number of iterations reached in kmeans!'
    c = zeros(u.shape)
    for m in xrange(0,M):
        dom = nonzero(chi == (m+1))
        if size(dom) != 0:
            c[dom] = a[m]

    return c, chi, a
