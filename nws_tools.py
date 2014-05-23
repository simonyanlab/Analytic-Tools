# nws_tools.py - Collection of network creation/processing/analysis/plotting routines
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# September 25 2013

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import natsort
import os
import csv
import inspect
from scipy import weave
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D

##########################################################################################
def strengths_und(CIJ):
    """
    Compute nodal strengths in an undirected graph

    Parameters
    ----------
    CIJ : NumPy 2darray
        Undirected binary/weighted connection matrix

    Returns
    -------
    st : NumPy 1darray
        Nodal strength vector

    Notes
    -----
    This function does *not* do any error checking and assumes you know what you are doing

    See also
    --------
    strengths_und.m : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                      `here <https://sites.google.com/site/bctnet/>`_

    bctpy : An unofficial Python port of the BCT is currently available at the 
            `Python Package Index <https://pypi.python.org/pypi/bctpy>`_
            and can be installed using pip. 
    """

    return np.sum(CIJ,axis=0)

##########################################################################################
def degrees_und(CIJ):
    """
    Compute nodal degrees in an undirected graph

    Parameters
    ----------
    CIJ : NumPy 2darray
        Undirected binary/weighted connection matrix

    Returns
    -------
    deg : NumPy 1darray
        Nodal degree vector

    Notes
    -----
    This function does *not* do any error checking and assumes you know what you are doing

    See also
    --------
    degrees_und.m : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                    `here <https://sites.google.com/site/bctnet/>`_

    bctpy : An unofficial Python port of the BCT is currently available at the 
            `Python Package Index <https://pypi.python.org/pypi/bctpy>`_
            and can be installed using pip. 
    """

    return (CIJ != 0).sum(1)

##########################################################################################
def density_und(CIJ):
    """
    Compute the connection density of an undirected graph

    Parameters
    ----------
    CIJ : NumPy 2darray
        Undirected binary/weighted connection matrix

    Returns
    -------
    den : float
        density (fraction of present connections to possible connections)

    Notes
    -----
    This function does *not* do any error checking and assumes you know what you are doing

    See also
    --------
    density_und.m : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                    `here <https://sites.google.com/site/bctnet/>`_

    bctpy : An unofficial Python port of the BCT is currently available at the 
            `Python Package Index <https://pypi.python.org/pypi/bctpy>`_
            and can be installed using pip. 
    """

    N = CIJ.shape[0]                    # no. of nodes
    K = (np.triu(CIJ,1)!=0).sum()       # no. of edges
    return K/((N**2 - N)/2.0)

##########################################################################################
def get_corr(txtpath,corrtype='pearson',**kwargs):
    """
    Compute correlation matrices of time-series using `corrtype` as measure of statistical dependence

    Parameters
    ----------
    txtpath : string
        Path to directory holding ROI-averaged time-series dumped in `txt` files.
        The following file-naming convention is required

                `sNxy_bla_bla.txt`,

        where `N` is the group id (1,2,3,...), `xy` denotes the subject number 
        (01,02,...,99 or 001,002,...,999) and anything else is separated 
        by an underscore. The files will be read in lexicographic order,
        i.e., `s101_1.txt`, `s101_2.txt`,... or `s101_Amygdala.txt`, `s101_Beemygdala`,...
    corrtype : string
        Specifier indicating which type of statistical dependence to use to compute 
        pairwise correlations. Currently supported options are 

                `pearson`: the classical zero-lag Pearson correlation coefficient 
                (see NumPy's corrcoef for details)

                `mi`: (normalized) mutual information 
                (see the docstring of mutual_info in this module for details)

    **kwargs : keyword arguments
        Additional keyword arguments to be passed on to the function computing 
        the pairwise correlations (currently either NumPy's corrcoef or mutual_info
        in this module). 
       
    Returns
    -------
    corrs : NumPy 3darray
        `N`-by-`N` correlation matrices of numsubs subjects. Format is 
                `corrs.shape = (N,N,numsubs)`,
        s.t.
                `corrs[:,:,i]` = `N x N` correlation matrix of `i`-th subject 
    bigmat : NumPy 3darray
        Tensor holding unprocessed time series of all subjects. Format is 
                `bigmat.shape = (tlen,N,numsubs)`,
        where `tlen` is the length of the time-series and `N` is the number of                 
        regions (=nodes in a network). 
    sublist : list of strings
        List of subjects found in folder specified by `txtpath`, e.g.,
                `sublist = ['s101','s103','s110','s111','s112',...]`

    Notes
    -----
    None

    See also
    --------
    numpy.corrcoef, mutual_info
    """

    # Sanity checks 
    if type(txtpath).__name__ != 'str':
        raise TypeError('Input has to be a string specifying the path to the txt-file directory!')
    try:
        corrtype = corrtype.lower()
    except: raise TypeError('Correlation type input must be a string!')

    # Get list of all txt-files in txtpath and order them lexicographically
    if txtpath[-1] == ' '  or txtpath[-1] == os.sep: txtpath = txtpath[:-1]
    txtfiles = natsort.natsorted(myglob(txtpath,"*.[Tt][Xx][Tt]"), key=lambda y: y.lower())

    # Load very first file to get length of time-series
    firstsub = txtfiles[0]
    tlen     = get_numlines(firstsub)

    # Search from left in file-name for first "s" (naming scheme: sNxy_bla_bla_.txt)
    firstsub  = firstsub.replace(txtpath+os.sep,'')
    s_in_name = firstsub.find('s')

    # The characters right of "s" until the first "_" are the subject identifier
    udrline = firstsub[s_in_name::].find('_')
    subject = firstsub[s_in_name:s_in_name+udrline]

    # Get number of regions
    numregs = ''.join(txtfiles).count(subject)
    
    # Generate list of subjects
    sublist = [subject]
    for fl in txtfiles:
        if fl.count(subject) == 0:
            s_in_name = fl.rfind('s')
            udrline   = fl[s_in_name::].find('_')
            subject   = fl[s_in_name:s_in_name+udrline]
            sublist.append(subject)

    # Get number of subjects
    numsubs = len(sublist)

    # Allocate tensor to hold all time series
    bigmat = np.zeros((tlen,numregs,numsubs))

    # Allocate tensor holding correlation matrices of all subjects 
    corrs = np.zeros((numregs,numregs,numsubs))

    # Cycle through subjects and save per-subject time series data column-wise
    for k in xrange(numsubs):
        col = 0
        for fl in txtfiles:
            if fl.count(sublist[k]):
                bigmat[:,col,k] = [float(line.strip()) for line in open(fl)]
                col += 1

        # Safeguard: stop if subject is missing, i.e., col = 0 still (weirder things have happened...)
        if col == 0: 
            raise ValueError("Subject "+sub+" is missing!")

        # Compute correlations based on corrtype
        if corrtype == 'pearson':
            corrs[:,:,k] = np.corrcoef(bigmat[:,:,k],rowvar=0,**kwargs)
        elif corrtype == 'mi':
            corrs[:,:,k] = mutual_info(bigmat[:,:,k],**kwargs)
            

    # Happy breakdown
    return bigmat, corrs, sublist

##########################################################################################
def corrcheck(*args,**kwargs):
    """
    Sanity checks for correlation matrices (Pearson or NMI)
    
    Parameters
    ----------
    Dynamic : Usage as follows
    corrcheck(A) : input is NumPy 2darray                    
        shows some statistics for the correlation matrix `A`
    corrcheck(A,label) : input is NumPy 2darray and ['string']                    
        shows some statistics for the correlation matrix `A` and uses
        `label`, a list containing one string, as title in figures. 
    corrcheck(A,B,C,...) : input are many NumPy 2darrays            
        shows some statistics for the correlation matrices `A`, `B`, `C`,....
    corrcheck(A,B,C,...,label) : input are many NumPy 2darrays and a list of strings      
        shows some statistics for the correlation matrices `A`, `B`, `C`,....
        and uses the list of strings `label` to generate titles in figures. 
        Note that `len(label)` has to be equal to the number of 
        input matrices. 
    corrcheck(T) : input is NumPy 3darray                    
        shows some statistics for correlation matrices stored 
        in the tensor `T`. The storage scheme has to be
                `T[:,:,0] = A`

                `T[:,:,1] = B`

                `T[:,:,2] = C`

                etc.

        where `A`, `B`, `C`,... are correlation matrices. 
    corrcheck(T,label) : input is NumPy 3darray and list of strings
        shows some statistics for correlation matrices stored 
        in the tensor `T`. The storage scheme has to be
                `T[:,:,0] = A`

                `T[:,:,1] = B`

                `T[:,:,2] = C`

                etc.

        where `A`, `B`, `C`,... are correlation matrices. The list of strings `label`
        is used to generate titles in figures. Note that `len(label)`
        has to be equal to `T.shape[2]`
    corrcheck(...,title='mytitle') : input is any of the above
        same as above and and uses the string `mytitle` as window name for figures. 

    Returns
    -------
    Nothing : None
    
    Notes
    -----
    None
    
    See also
    --------
    None
    """

    # Plotting params used later (max. #plots per row)
    cplot = 5

    # Sanity checks
    myin = len(args)
    if myin == 0: raise ValueError('At least one input required!')

    # Assign global name for all figures if provided by additional keyword argument 'title'
    figtitle = kwargs.get('title',None); nofigname = False
    if figtitle == None: nofigname = True

    # If labels have been provided, extract them now
    if type(args[-1]).__name__ == 'list':
        myin  -= 1
        labels = args[-1]
        usrlbl = 1
    elif type(args[-1]).__name__ == 'str':
        myin  -= 1
        labels = [args[-1]]
        usrlbl = 1
    else:
        usrlbl = 0

    # Get shape of input
    szin = len(args[0].shape)

    # If input is a list of matrices, store them in a tensor
    if szin == 2:
        rw,cl = args[0].shape
        if (rw != cl) or (min(args[0].shape)==1):
            raise ValueError('Input matrices must be square!')
        corrs = np.zeros((rw,cl,myin))
        for i in xrange(myin):
            try:
                corrs[:,:,i] = args[i]
            except:
                raise ValueError('All input matrices have to be of the same size!')

    # If input is a tensor, there's not much to do  
    elif szin == 3:
        if myin > 1: raise ValueError('Not more than one input tensor supported!')
        shv = args[0].shape
        if (min(shv[0],shv[1]) == 1) or (shv[0]!=shv[1]):
            raise ValueError('Input tensor must be of the format N-by-N-by-k!')
        corrs = args[0]
    else:
        raise TypeError('Input has to be either a matrix/matrices or a tensor!')

    # Count number of matrices and get their dimension
    nmat = corrs.shape[-1]
    N    = corrs.shape[0]

    # Check if those matrices are real and "reasonable"
    if np.isnan(corrs).max() == True or np.isinf(corrs).max() == True or np.isreal(corrs).min() == False:
        raise ValueError("All matrices must be real without NaNs or Infs!")

    # Check if we're dealing with Pearson or NMI matrices (or something completely unexpected)
    cmin = corrs.min(); cmax = corrs.max()
    if cmax > 1 or cmin < -1:
        msg = "Input has to have values between -1/+1 or 0/+1. Found "+str(cmin)+" to "+str(cmax)
        raise ValueError(msg)
    maxval = 1
    if corrs.min() < 0:
        minval = -1
    else:
        minval = 0

    # If labels have been provided, check if we got enough of'em; if there are no labels, generate defaults
    if (usrlbl):
        if len(labels) != nmat: raise ValueError('Numbers of labels and matrices do not match up!')
    else:
        labels = ['Matrix '+str(i+1) for i in xrange(nmat)]

    # Set subplot params and turn on interactive plotting
    rplot = int(np.ceil(nmat/cplot))
    if nmat <= cplot: cplot = nmat
    plt.ion()

    # Now let's actually do something and plot the correlation matrices (show warning matrix if is not symmetric)
    fig = plt.figure(figsize=(8,8))
    if nofigname: figtitle = fig.canvas.get_window_title()
    fig.canvas.set_window_title(figtitle+': '+str(N)+' Nodes',)
    for i in xrange(nmat):
        plt.subplot(rplot,cplot,i+1)
        im = plt.imshow(corrs[:,:,i],cmap='jet',interpolation='nearest',vmin=minval,vmax=maxval)
        plt.axis('off')
        plt.title(labels[i])
        if issym(corrs[:,:,i]) == False:
            print "WARNING: "+labels[i]+" is not symmetric!"
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.draw()

    # Plot correlation histograms
    meanval = np.mean([minval,maxval])
    idx = np.nonzero(np.triu(np.ones((N,N)),1))
    NN  = (N**2 - N)/2
    fig = plt.figure(figsize=(8,8))
    if nofigname: figtitle = fig.canvas.get_window_title()
    fig.canvas.set_window_title(figtitle+': '+"Correlation Histograms")
    bars = []; ylims = []
    for i in xrange(nmat):
        cvec = corrs[idx[0],idx[1],i]
        [corrcount,corrbins] = np.histogram(cvec,bins=20,range=(minval,maxval))
        bars.append(plt.subplot(rplot,cplot,i+1))
        plt.bar(corrbins[:-1],corrcount/NN,width=np.abs(corrbins[0]-corrbins[1]))
        ylims.append(bars[-1].get_ylim()[1])
        plt.xlim(minval,maxval)
        plt.xticks((minval,meanval,maxval),(str(minval),str(meanval),str(maxval)))
        plt.title(labels[i])
        if np.mod(i+1,cplot) == 1: plt.ylabel('Frequency')
    ymax = max(ylims)
    for mybar in bars: mybar.set_ylim(top=ymax)
    plt.draw()

    # Show negative correlations (for Pearson matrices)
    if minval < 0:
        fig = plt.figure(figsize=(8,8))
        if nofigname: figtitle = fig.canvas.get_window_title()
        fig.canvas.set_window_title(figtitle+': '+"Negative Correlations Are BLACK")
        for i in xrange(nmat):
            plt.subplot(rplot,cplot,i+1)
            plt.imshow((corrs[:,:,i]>0).astype(float),cmap='gray',interpolation='nearest',vmin=0,vmax=1)
            plt.axis('off')
            plt.title(labels[i])
        plt.draw()

    # Diversity
    fig = plt.figure(figsize=(8,8))
    if nofigname: figtitle = fig.canvas.get_window_title()
    fig.canvas.set_window_title(figtitle+': '+"Diversity of Correlations")
    xsteps = np.arange(1,N+1)
    stems = []; ylims = []
    for i in xrange(nmat):
        stems.append(plt.subplot(rplot,cplot,i+1))
        varc = np.var(corrs[:,:,i],0,ddof=1)
        plt.stem(xsteps,varc)
        ylims.append(stems[-1].get_ylim()[1])
        plt.xlim(-1,N+1)
        plt.xticks((0,N),('1',str(N)))
        plt.title(labels[i])
    ymax = max(ylims)
    for mystem in stems: mystem.set_ylim(top=ymax)
    plt.draw()

##########################################################################################
def get_meannw(nws,percval=0.75):
    """
    Helper function to compute group-averaged networks

    Parameters
    ----------
    nws : NumPy 3darray
        `N`-by-`N` connectivity matrices of numsubs subjects. Format is 
                `nws.shape = (N,N,numsubs)`,
        s.t.
                `nws[:,:,i] = N x N` connectivity matrix of `i`-th subject 
    percval : float
        Percentage value, s.t. connections not present in at least `percval`
        percent of subjects are not considered, thus `0 <= percval <= 1`.
        Default setting is `percval = 0.75` (following van den Heuvel's and Sporns' 
        rich club paper, see below). 
       
    Returns
    -------
    mean_wghted : NumPy 2darray
        `N`-by-`N` mean value matrix of `numsubs` matrices stored in `nws` where
        only connections present in at least `percval` percent of subjects
        are considered
    percval : float
        Percentage value used to generate `mean_wghted`
       
    Notes
    -----
    If the current setting of `percval` leads to a disconnected network, 
    the code increases `percval` in 5% steps to assure connectedness of the group-averaged graph. 
    The concept of using only a certain percentage of edges present in subjects was taken from 
    M. van den Heuvel, O. Sporns: "Rich-Club Organization of the Human Connectome" (2011), J. Neurosci. 
    Currently available `here <http://www.jneurosci.org/content/31/44/15775.full>`_
    
    See also
    --------
    None
    """

    # Sanity checks
    tensorcheck(nws)
    try: tmp = percval > 1 or percval < 0
    except: raise TypeError("Percentage value must be a floating point number >= 0 and <= 1!")
    if (tmp): raise ValueError("Percentage value must be >= 0 and <= 1!")
    
    # Get shape of input tensor
    N       = nws.shape[0]
    numsubs = nws.shape[-1]

    # Allocate memory for binary/weighted group averaged networks
    mean_binary = np.zeros((N,N))
    mean_wghted = np.zeros((N,N))

    # Compute mean network and keep increasing percval until we get a connected mean network
    docalc = True
    while docalc:

        # Reset matrices 
        mean_binary[:] = 0
        mean_wghted[:] = 0

        # Cycle through subjects to compute average network
        for i in xrange(numsubs):
            mean_binary = mean_binary + (nws[:,:,i]>0).astype(float)
            mean_wghted = mean_wghted + nws[:,:,i]

        # Kick out connections not present in at least percval% of subjects (in binary and weighted NWs)
        mean_binary = (mean_binary/numsubs >= percval).astype(float)
        mean_wghted = mean_wghted/numsubs * mean_binary

        # Check connectedness of mean network
        if degrees_und(mean_binary).min() == 0:
            print "WARNING: Mean network disconnected for percval = "+str(np.round(1e2*percval))+"%"
            if percval < 1:
                print "Decreasing percval by 5%..."
                percval -= 0.05
                print "New value for percval is now "+str(np.round(1e2*percval))+"%"
            else:
                msg = "Mean network disconnected for percval = 0%. That means at least one node is "+\
                      "disconnected in ALL per-subject networks..."
                raise ValueError(msg)
        else:
            docalc = False

    return mean_wghted, percval

##########################################################################################
def rm_negatives(corrs):
    """
    Remove negative correlations from correlation matrices

    Parameters
    ----------
    corrs : NumPy 3darray
        `N`-by-`N` correlation matrices of `numsubs` subjects. Format is 
                `corrs.shape = (N,N,numsubs)`,
        s.t.
                `corrs[:,:,i] = N x N` connectivity matrix of `i`-th subject 

    Returns
    -------
    corrs : NumPy 3darray
        Same format as input tensor but `corrs >= 0`. 

    Notes
    -----
    None

    See also
    --------
    None
    """

    # Sanity checks
    tensorcheck(corrs)

    # Get dimensions
    N       = corrs.shape[0]
    numsubs = corrs.shape[-1]

    # Zero diagonals of connectivity matrices
    for i in xrange(numsubs):
        np.fill_diagonal(corrs[:,:,i],0)

    # Remove negative correlations
    nws = (corrs > 0)*corrs

    # Check if we lost some nodes...
    for i in xrange(numsubs):
        deg = degrees_und(corrs[:,:,i])
        if deg.min() == 0:
            print "WARNING: In subject "+str(i)+" node(s) "+str(np.nonzero(deg==deg.min())[0])+" got disconnected!"

    return nws

##########################################################################################
def thresh_nws(nws,userdens=None):
    """
    Threshold networks based on connection density

    Parameters
    ----------
    nws : NumPy 3darray
        Undirected `N`-by-`N` connectivity matrices of `numsubs` subjects. Format is 
                `corrs.shape = (N,N,numsubs)`,
        s.t.
                `corrs[:,:,i] = N x N` connectivity matrix of `i`-th subject 
    userdens : int
        By default, the input networks are thresholded down to the lowest common 
        connection density without disconnecting any nodes in the networks. If `userdens` 
        is provided, then it is used as density level to which all networks should be 
        thresholded, i.e., `0 < userdens < 100`. See Notes below for details. 
               
    Returns
    -------
    th_nws : NumPy 3darray
        Thresholded networks for all subjects. Format is the same as for `nws`. 
    tau_levels : NumPy 1darray
        The threshold values for each subject's network corresponding to the 
        networks stored in `th_nws`, i.e. `tau_levels[i]` is the threshold that 
        generated the network `th_nws[:,:,i]`, i.e., the network of subject `i`. 
    den_values : NumPy 1darray
        Same format as `tau_levels` but holding the exact density values for each subject
    th_mnw : NumPy 2darray
        The group averaged (across all subjects) weighted network
    mnw_percval: float
        Percentage value used to compute `th_mnw` (see documentation of `get_meannw` for
        details)

    Notes
    -----
    By default, the thresholding algorithm uses the lowest common connection density 
    before a node is disconnected. 
    That means, if networks `A`, `B` and `C` can be thresholded down to 40%, 50% and 60% density, 
    respectively, without disconnecting any nodes, then the lowest common density for thresholding 
    `A`, `B` and `C` together is 60%. If, e.g., the raw network `A` already has a density of 
    60% or lower, it is excluded from thresholding and the original network is copied 
    into `th_nws`. If a density level is provided by the user, then the code tries to use 
    it unless it violates connectedness of all thresholded networks - in this case 
    the lowest common density of all networks is used. 
    The code below relies on the routine `get_meannw` in this module to compute the group-averaged
    network. 

    See also
    --------
    None
    """

    # Sanity checks
    tensorcheck(nws)
    if userdens != None:
        try: tmp = np.round(userdens) != userdens 
        except: raise TypeError('Density level has to be a number between 0 and 100!')
        if (tmp): raise ValueError('The density level must be an integer!')
        if (userdens <= 0) or (userdens >= 100):
            raise ValueError('The density level must be between 0 and 100!')

    # Get dimension of per-subject networks
    N       = nws.shape[0]
    numsubs = nws.shape[-1]

    # Zero diagonals and check for symmetry
    for i in xrange(numsubs):
        np.fill_diagonal(nws[:,:,i],0)
        if issym(nws[:,:,i]) == False:
            raise ValueError("Matrix "+str(i)+" is not symmetric!")

    # Get max. and min. weights (min weight should be >= 0 otherwise the stuff below makes no sense...)
    maxw = nws.max()
    if nws.min() < 0: raise ValueError('Only non-negative weights supported!')

    # Allocate vector for original densities 
    raw_den = np.zeros((numsubs,))

    # Compute densities of raw networks 
    for i in xrange(numsubs):
        raw_den[i] = density_und(nws[:,:,i])

    # Compute min/max density in raw data
    min_raw = int(np.floor(1e2*raw_den.min()))
    max_raw = int(np.ceil(1e2*raw_den.max()))

    # Break if a nw has density zero or if max. density is below desired dens.
    if min_raw == 0: raise ValueError('Network '+str(raw_den.argmin())+' has density 0%!')
    if userdens >= max_raw:
        print "All networks have density lower than desired density "+str(userdens)+"%"
        th_nws = nws; tau_levels = None; den_values = raw_den; th_mnw,mnw_percval = get_meannw(nws) 
        return th_nws, tau_levels, den_values, th_mnw, mnw_percval

    # Inform user about minimal/maximal density in raw data
    print "\nRaw data has following density values: \n"
    print "\tMinimal density: "+str(min_raw)+"%"
    print "\tMaximal density: "+str(max_raw)+"%"

    # Create vector of thresholds to iterate on
    dt      = 1e-3
    threshs = np.arange(0,1+2*dt,dt)

    # Allocate space for output
    th_nws     = np.zeros(nws.shape)
    tau_levels = np.zeros((numsubs,))
    den_values = np.zeros((numsubs,))
    th_mnw     = np.zeros((N,N))

    # Cycle through subjects and threshold the connectivity matrices until a node disconnects
    for i in xrange(numsubs):
        tau = -1
        mnw = nws[:,:,i]
        den = density_und(mnw)

        # Start with 1%-weight threshold and increase 
        for th in threshs:

            # Save old iterates
            den_old = den
            tau_old = tau
            mnw_old = mnw

            # Threshold based on percentage of max. weight: throw out all weights < than 1%-max. weight, 2%, ...
            # Thin out connectivity matrix step by step (that's why we only have to load nws(:,:,i) once
            tau = th*maxw
            mnw = mnw*(mnw >= tau).astype(float)

            # Compute density of thinned graph (weight info is discarded)
            den     = density_und(mnw)

            # Compute nodal degrees of network 
            deg = degrees_und(mnw)

            # As soon as one node gets disconnected (i.e. deg[i]=0) stop thresholding and save previous dens
            if deg.min() == 0:
                th_nws[:,:,i] = mnw_old
                tau_levels[i] = tau_old
                den_values[i] = den_old
                break

    # Compute minimal density before fragmentation across all subjects (ceil is important here: min = 0.734 -> 74% NOT 73%)
    densities = np.round(1e2*den_values)
    print "\nDensities of per-subject networks are as follows: "
    for i in xrange(densities.size): print "Subject #"+str(i)+": "+str(int(densities[i]))+"%"
    min_den = int(np.round(1e2*den_values.max()))
    print "\nMinimal density before fragmentation across all subjects is "+str(min_den)+"%"

    # Assign thresholding density level
    if userdens == None:
        thresh_dens = min_den
    else:
        if userdens < min_den:
            print "\n User provided density of "+str(int(userdens))+"% lower than minimal admissible density of "+str(min_den)+"%. "
            print "Using minimal admissible density instead. "
            thresh_dens = min_den
        else: 
            thresh_dens = int(userdens)

    # Inform the user about what's gonna happen 
    print "\nUsing density of "+str(int(thresh_dens))+"%. Starting thresholding procedure...\n"

    # Backtracking parameter
    beta = 0.3

    # Cycle through subjects
    for i in xrange(numsubs):

        den_perc = 100
        th       = -dt
        mnw      = nws[:,:,i]
        raw_dper = int(np.round(1e2*raw_den[i]))

        if raw_dper <= thresh_dens:

            print "Density of raw network #"+str(i)+" is "+str(raw_dper)+"%"+\
                " which is already lower than thresholding density of "+str(thresh_dens)+"%"
            print "Returning original unthresholded network"
            th_nws[:,:,i] = mnw
            tau_levels[i] = 1
            den_values[i] = raw_den[i]

        else:

            while den_perc > thresh_dens:

                th  += dt
                tau = th*maxw
                mnw = mnw*(mnw >= tau).astype(float)
                
                den      = density_und(mnw)
                den_perc = np.round(1e2*den)

                if den_perc < thresh_dens:
                    th *= beta

            th_nws[:,:,i] = mnw
            tau_levels[i] = tau
            den_values[i] = den

    # Compute group average network
    th_mnw,mnw_percval = get_meannw(th_nws)

    # Be polite and dismiss the user 
    print "Done...\n"

    return th_nws, tau_levels, den_values, th_mnw, mnw_percval

##########################################################################################
def normalize(I,a=0,b=1):
    """
    Re-scales a NumPy ndarray

    Parameters
    ----------
    I : NumPy ndarray
        An array of size > 1 (shape can be arbitrary)
    a : float
        Floating point number representing the lower normalization bound. 
        (Note that it has to hold that `a < b`)
    b : float
        Floating point number representing the upper normalization bound. 
        (Note that it has to hold that `a < b`)
       
    Returns
    -------
    In : NumPy ndarray
        Scaled version of the input array `I`, such that `a = In.min()` and 
        `b = In.max()`

    Notes
    -----
    None 

    Examples
    --------
    >>> I = array([[-1,.2],[100,0]])
    >>> In = normalize(I,a=-10,b=12)
    >>> In 
    array([[-10.        ,  -9.73861386],
           [ 12.        , -10.        ]])

    See also
    --------
    None 
    """

    # Ensure that I is a NumPy-ndarray
    try: tmp = I.size == 1
    except TypeError: raise TypeError('I has to be a NumPy ndarray!')
    if (tmp): raise ValueError('I has to be a NumPy ndarray of size > 1!')

    # If normalization bounds are user specified, check them
    try: tmp = b <= a
    except TypeError: raise TypeError('a and b have to be scalars satisfying a < b!')
    if (tmp): raise ValueError('a has to be strictly smaller than b!')
    if np.absolute(a - b) < np.finfo(float).eps:
            raise ValueError('|a-b|<eps, no normalization possible')

    # Get min and max of I
    Imin   = I.min()
    Imax   = I.max()

    # If min and max values of I are identical do nothing, if they differ close to machine precision abort
    if Imin == Imax:
        return I
    elif np.absolute(Imin - Imax) < np.finfo(float).eps:
        raise ValueError('|Imin-Imax|<eps, no normalization possible')

    # Make a local copy of I
    I = I.copy()

    # Here the normalization is done
    I = (I - Imin)*(b - a)/(Imax - Imin) + a

    # Return normalized array
    return I

##########################################################################################
def csv2dict(csvfile):
    """
    Reads 3D nodal coordinates of from a csv file into a Python dictionary

    Parameters
    ----------
    csvfile : string 
        File-name of (or full path to) the csv file holding the nodal coordinates.
        The format of this file HAS to be 
                 `x, y, z` 

                 `x, y, z` 

                 `x, y, z` 

                 .

                 .

        for each node. Thus #rows = #nodes. 

    Returns
    -------
    mydict : dictionary 
        Nodal coordinates as read from the input csv file. Format is
                `{0: (x, y, z),`
                `{1: (x, y, z),`

                `{2: (x, y, z),`

                 .

                 .

        Thus the dictionary has #nodes keys. 
    
    Notes
    -----
    None 

    See also
    --------
    None 
    """

    # Sanity checks
    if type(csvfile).__name__ != 'str': raise TypeError('Input has to be a string!')
    
    # Open csvfile
    fh = open(csvfile,'rU')
    fh.seek(0)
    
    # Read nodal coordinates using csv module
    reader = csv.reader(fh, dialect='excel',delimiter=',', quotechar='"')
    
    # Iterate through rows and convert coordinates from string lists to float tuples
    mydict = {}
    i = 0
    for row in reader:
        mydict[i] = tuple([float(r) for r in row])
        i += 1
    
    return mydict

##########################################################################################
def shownet(A,coords,colorvec=None,sizevec=None,labels=[],threshs=[.8,.3,0],lwdths=[5,2,.1],nodecmap='jet',edgecmap='jet',textscale=1):
    """
    Plots a 3d network using Mayavi

    Parameters
    ----------
    A : NumPy 2darray
        Square `N`-by-`N` connectivity matrix of the network
    coords: dictionary 
        Nodal coordinates of the graph. Format is
                `{0: (x, y, z),`

                `{1: (x, y, z),`

                `{2: (x, y, z),`

                 .

                 .

        Note that the dictionary has to have `N` keys. 
    colorvec : NumPy 1darray
        Vector of color-values for each node. This could be nodal strength or modular information of nodes 
        (i.e., to which module does node `i` belong to). Thus `colorvec` has to be of length `N` and all its
        components must be in `[0,1]`. 
    sizevec : NumPy 1darray 
        Vector of nodal sizes. This could be degree, centrality, etc. Thus `sizevec` has to be of length 
        `N` and all its components must be `>= 0`. 
    labels : list
        Nodal labels. Format is `['Name1','Name2','Name3',...]` where the ordering HAS to be the same
        as in the `coords` dictionary. Note that the list has to have length `N`. 
    threshs : list 
        List of thresholds for visualization. Edges with weights larger than `threshs[0]` are drawn 
        thickest, weights `> threshs[1]` are thinner and so on. Note that if `threshs[-1]>0` not all 
        edges of the network are plotted (since edges with `0 < weight < threshs[-1]` will be ignored). 
    lwdths : list 
        List of line-widths associated to the list of thresholds. Edges with weights larger than 
        `threshs[0]` are drawn with line-width `lwdths[0]`, edges with `weights > threshs[1]` 
        have line-width `lwdths[1]` and so on. Thus `len(lwdths) == len(threshs)`. 
    nodecmap : string 
        Mayavi colormap to be used for plotting nodes. See Notes for details. 
    edgecmap : string 
        Mayavi colormap to be used for plotting edges. See Notes for details. 
    textscale : positive number 
        Scaling factor for labels (larger numbers -> larger text)

    Returns
    -------
    Nothing : None

    Notes
    -----
    A list of available colormaps in Mayavi is currently available 
    `here <http://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html>`_. 
    See the 
    `Mayavi documentation <http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html>`_
    for more info. 

    See also
    --------
    show_nw : A Matplotlib based implementation with extended functionality (but MUCH slower rendering)
    """

    # For those lucky enough to have a running installation of Mayavi...
    try:
        from mayavi import mlab
    except: 
        msg = 'Mayavi could not be imported. You might want to try show_nw, a slower (but more feature '\
              +'rich) graph rendering routine based on Matplotlib.'
        raise ImportError(msg)

    # Sanity checks and assign default values
    try:
        (N,M) = A.shape
    except: 
        raise TypeError('A has to be a square NumPy array!')
    if N != M: raise ValueError('A has to be square!')
    if np.isnan(A).max() == True or np.isinf(A).max() == True or np.isreal(A).min() == False:
        raise ValueError("A must be real-valued without NaNs or Infs!")

    if type(coords).__name__ != 'dict':
        raise TypeError("The coordinates have to be a dictionary!")
    if len(coords.keys()) != N:
        raise ValueError('The coordinate dictionary has to have N keys!')

    if colorvec != None:
        try: tmp = colorvec.size != N
        except: raise TypeError('colorvec has to be a NumPy array!')
        if (tmp): raise ValueError('colorvec has to have length N!')
        if np.isnan(colorvec).max() == True or np.isinf(colorvec).max() == True or np.isreal(colorvec).min()==False:
            raise ValueError("colorvec must be real-valued without NaNs or Infs!")
        if colorvec.min() < 0 or colorvec.max() > 1:
            raise ValueError('colorvec values must be between 0 and 1!')
    else: colorvec = np.ones((N,))

    if sizevec != None:
        try: tmp = sizevec.size != N
        except: raise TypeError('sizevec has to be a NumPy array!')
        if (tmp): raise ValueError('sizevec has to have length N!')
        if np.isnan(sizevec).max() == True or np.isinf(sizevec).max() == True or np.isreal(sizevec).min()==False:
            raise ValueError("sizevec must be real-valued without NaNs or Infs!")
        if sizevec.min() < 0:
            raise ValueError('sizevec values must be >= 0!')
    else: sizevec = np.ones((N,))

    if type(labels).__name__ != 'list':
        raise TypeError("Nodal labels have to be provided as list!")

    if type(threshs).__name__ != 'list':
        raise TypeError("Thresholds have to be provided as list!")

    if type(lwdths).__name__ != 'list':
        raise TypeError("Linewidths have to be provided as list!")
    if len(lwdths) != len(threshs):
        raise ValueError("Same number of thresholds and linewidths required!")

    if type(nodecmap).__name__ != 'str':
        raise TypeError("Colormap for nodes has to be provided as string!")

    if type(edgecmap).__name__ != 'str':
        raise TypeError("Colormap for edges has to be provided as string!")

    if type(textscale).__name__ != 'float' and type(textscale).__name__ != 'int':
        raise TypeError("Scaling factor of text has to be provided as number!")

    # Now start to actually do something...
    pts = mlab.quiver3d(np.array([coords[i][0] for i in coords.keys()]),\
                        np.array([coords[i][1] for i in coords.keys()]),\
                        np.array([coords[i][2] for i in coords.keys()]),\
                        sizevec,sizevec,sizevec,scalars=colorvec,\
                        scale_factor = 1,mode='sphere',colormap=nodecmap)

    # Coloring of the balls is based on the provided scalars 
    pts.glyph.color_mode = 'color_by_scalar'

    # Finally, center the glyphs on the data point
    pts.glyph.glyph_source.glyph_source.center = [0, 0, 0]

    # Cycle through threshold levels to generate different line-widths of networks 
    srcs = []; lines = []
    for k in xrange(len(threshs)):

        # Generate empty lists to hold (x,y,z) data and color information
        x = list()
        y = list()
        z = list()
        s = list()
        connections = list()
        index = 0
        b = 2

        # Get matrix entries > current threshold level
        for i in xrange(N):
            for j in xrange(i+1,N):
                if A[i,j] > threshs[k]:
                    x.append(coords[i][0])
                    x.append(coords[j][0])
                    y.append(coords[i][1])
                    y.append(coords[j][1])
                    z.append(coords[i][2])
                    z.append(coords[j][2])
                    s.append(A[i][j])
                    s.append(A[i][j])
                    connections.append(np.vstack([np.arange(index, index + b - 1.5), np.arange(index+1, index + b - 0.5)]).T)
                    index += b

        # Finally generate lines connecting dots
        srcs.append(mlab.pipeline.scalar_scatter(x,y,z,s))
        srcs[-1].mlab_source.dataset.lines = connections
        lines.append(mlab.pipeline.stripper(srcs[-1]))
        mlab.pipeline.surface(lines[-1], colormap=edgecmap, line_width=lwdths[k], vmax=1, vmin=0)

    # Label nodes if wanted
    for i in xrange(len(labels)):
        mlab.text3d(coords[i][0]+2,coords[i][1],coords[i][2],labels[i],color=(0,0,0),scale=3)

    return

##########################################################################################
def show_nw(A,coords,colorvec=None,sizevec=None,labels=[],nodecmap=plt.get_cmap(name='jet'),edgecmap=plt.get_cmap(name='jet'),linewidths=None,nodes3d=False,viewtype='axial'):
    """
    Matplotlib-based plotting routine for 3d networks

    Parameters
    ----------
    A : NumPy 2darray
        Square `N`-by-`N` connectivity matrix of the network
    coords: dictionary 
        Nodal coordinates of the graph. Format is
                `{0: (x, y, z),`

                `{1: (x, y, z),`

                `{2: (x, y, z),`

                 .

                 .

        Note that the dictionary has to have `N` keys. 
    colorvec : NumPy 1darray
        Vector of color-values for each node. This could be nodal strength or modular information of nodes 
        (i.e., to which module does node i belong to). Thus `colorvec` has to be of length `N` and all its
        components must be in `[0,1]`. 
    sizevec : NumPy 1darray 
        Vector of nodal sizes. This could be degree, centrality, etc. Thus `sizevec` has to be of 
        length `N` and all its components must be `>= 0`. 
    labels : list
        Nodal labels. Format is `['Name1','Name2','Name3',...]` where the ordering HAS to be the same
        as in the `coords` dictionary. Note that the list has to have length `N`. 
    nodecmap : Matplotlib colormap
        Colormap to use for plotting nodes
    edgecmap : Matplotlib colormap
        Colormap to use for plotting edges
    linewidths : NumPy 2darray
        Same format and nonzero-pattern as `A`. If no linewidhts are provided then the edge connecting 
        nodes `v_i` and `v_j` is plotted using the linewidth `A[i,j]`. By specifying, e.g., 
        `linewidhts = (1+A)**2`, the thickness of edges in the network-plot can be scaled. 
    nodes3d : bool
        If `nodes3d=True` then nodes are plotted using 3d spheres in space (with `diameters = sizevec`). 
        If `nodes3d=False` then the Matplotlib `scatter` function is used to plot nodes as flat 
        2d disks (faster).
    viewtype : str
        Camera position. Viewtype can be one of the following
                `axial (= axial_t)`       : Axial view from top down

                `axial_t`                 : Axial view from top down

                `axial_b`                 : Axial view from bottom up

                `sagittal (= sagittal_l)` : Sagittal view from left

                `sagittal_l`              : Sagittal view from left

                `sagittal_r`              : Sagittal view from right

                `coronal (= coronal_f)`   : Coronal view from front

                `coronal_f`               : Coronal view from front

                `coronal_b`               : Coronal view from back

    
    Returns
    -------
    Nothing : None

    Notes
    -----
    See Matplotlib's `mplot3d tutorial <http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html>`_

    See also
    --------
    shownet : A Mayavi based implementation with less functionality but MUCH faster rendering
    """

    # Sanity checks and assign default values
    try:
        (N,M) = A.shape
    except: raise TypeError('A has to be a square NumPy array!')
    if N != M: raise ValueError('A has to be square!')
    if np.isnan(A).max() == True or np.isinf(A).max() == True or np.isreal(A).min() == False:
        raise ValueError("A must be real-valued without NaNs or Infs!")
    if type(coords).__name__ != 'dict':
        raise TypeError("The coordinates have to be a dictionary!")
        if len(coords.keys()) != N:
            raise ValueError('The coordinate dictionary has to have N keys!')

    if colorvec != None:
        try: tmp = colorvec.size != N
        except: raise TypeError('colorvec has to be a NumPy array!')
        if (tmp): raise ValueError('colorvec has to have length N!')
        if np.isnan(colorvec).max() == True or np.isinf(colorvec).max() == True or np.isreal(colorvec).min()==False:
            raise ValueError("colorvec must be real-valued without NaNs or Infs!")
        if colorvec.min() < 0 or colorvec.max() > 1:
            raise ValueError('colorvec values must be between 0 and 1!')
    else: colorvec = np.ones((N,))

    if sizevec != None:
        try: tmp = sizevec.size != N
        except: raise TypeError('sizevec has to be a NumPy array!')
        if (tmp): raise ValueError('sizevec has to have length N!')
        if np.isnan(sizevec).max() == True or np.isinf(sizevec).max() == True or np.isreal(sizevec).min()==False:
            raise ValueError("sizevec must be real-valued without NaNs or Infs!")
        if sizevec.min() < 0:
            raise ValueError('sizevec values must be >= 0!')
    else: sizevec = np.ones((N,))

    if type(labels).__name__ != 'list':
        raise TypeError("Nodal labels have to be provided as list!")

    if type(nodecmap).__name__ != 'LinearSegmentedColormap':
        raise TypeError('Nodal colormap has to be a Matplotlib colormap!')

    if type(edgecmap).__name__ != 'LinearSegmentedColormap':
        raise TypeError('Edge colormap has to be a Matplotlib colormap!')

    if linewidths != None:
        try:
            (N,M) = linewidths.shape
        except: raise TypeError('Linewidths have to be a square NumPy array!')
        if N != M: raise ValueError('Linewidths have to be provided as square matrix!')
        if np.isnan(linewidths).max() == True or np.isinf(linewidths).max() == True or np.isreal(linewidths).min() == False:
                raise ValueError("Linewidths must be real-valued without NaNs or Infs!")

    if type(nodes3d).__name__ != 'bool':
        raise TypeError('The nodes3d flag has to be a Boolean variable!')

    if type(viewtype).__name__ != 'str':
        raise TypeError("Viewtype must be 'axial(_{t/b})', 'sagittal(_{l/r})' or 'coronal(_{f/b})'")

    # Turn on 3d projection
    ax = plt.gcf().gca(projection='3d')
    ax.hold(True)

    # Extract nodal x-, y-, and z-coordinates from the coords-dictionary
    x = np.array([coords[i][0] for i in coords.keys()])
    y = np.array([coords[i][1] for i in coords.keys()])
    z = np.array([coords[i][2] for i in coords.keys()])

    # Order matters here: FIRST plot connection, THEN nodes on top of the connections (looks weird otherwise)
    # Cycle through the matrix and plot every single connection line-by-line (this is *really* slow)
    if linewidths == None:
        for i in xrange(N):
            for j in xrange(i+1,N):
                if A[i,j] > 0:
                    plt.plot([x[i],x[j]],[y[i],y[j]],[z[i],z[j]],linewidth=A[i][j],color=edgecmap(A[i][j]))
    else: 
        for i in xrange(N):
            for j in xrange(i+1,N):
                if A[i,j] > 0:
                    plt.plot([x[i],x[j]],[y[i],y[j]],[z[i],z[j]],linewidth=linewidths[i][j],color=edgecmap(A[i][j]))

    # Plot nodes (either 3d spheres or flat scatter points)
    if nodes3d == False:
        plt.scatter(x,y,zs=z,marker='o',s=sizevec,c=colorvec,cmap=nodecmap)
    else:
        n      = 20#10
        theta  = np.arange(-n,n+1,2)/n*np.pi
        phi    = np.arange(-n,n+1,2)/n*np.pi/2
        cosphi = np.cos(phi); cosphi[0] = 0; cosphi[-1] = 0
        sinth  = np.sin(theta); sinth[0] = 0; sinth[-1] = 0    

        xx = np.outer(cosphi,np.cos(theta))
        yy = np.outer(cosphi,sinth)
        zz = np.outer(np.sin(phi),np.ones((n+1,)))

        for i in xrange(x.size):
            rd = sizevec[i]
            ax.plot_surface(rd*xx+x[i],rd*yy+y[i],rd*zz+z[i],\
                            color = nodecmap(colorvec[i]),\
                            cstride=1,rstride=1,linewidth=0)

    # Label nodes if wanted
    for i in xrange(len(labels)):
        ax.text(x[i]+2,y[i]+2,z[i]+2,labels[i],color='k',fontsize=14)

    # If viewtype was just specified as 'axial', 'coronal' or 'sagittal' use default (top, front, left) viewtypes
    if viewtype == 'axial':
        viewtype = 'axial_t'
    elif viewtype == 'sagittal' or viewtype == 'sagital':
        viewtype = 'sagittal_l'
    elif viewtype == 'coronal':
        viewtype = 'coronal_f'

    # Turn off axis (don't really mean anything in this context anyway...) and set up view
    if viewtype == 'axial_t':
        ax.view_init(elev=90,azim=-90)
    elif viewtype == 'axial_b':
        ax.view_init(elev=-90,azim=90)
    elif viewtype == 'coronal_f':
        ax.view_init(elev=0,azim=90)
    elif viewtype == 'coronal_b':
        ax.view_init(elev=0,azim=270)
    elif viewtype == 'sagittal_l':
        ax.view_init(elev=0,azim=180)
    elif viewtype == 'sagittal_r':
        ax.view_init(elev=0,azim=0)
    else: 
        print "WARNING: Unrecognized viewtype: "+viewtype
        print "Using default viewtype axial instead"
        ax.view_init(elev=90,azim=-90)
    plt.axis('scaled')
    plt.axis('off')
    
    return 

##########################################################################################
def generate_randnws(nw,M=100,method="auto",rwr=5):
    """
    Generate random networks given an input network

    Parameters
    ----------
    nw : NumPy 2darray
        Undirected binary/weighted connection matrix
    M : integer > 1
        Number of random networks to generate
    method : string
        String specifying which method to use to randomize 
        the input network. Currently supported options are 
        `'auto'` (default), `'randmio_und'`, and `'randmio_und_connected'`. 
        If `'auto'` is chosen then `'randmio_und_connected'` is used
        to compute the random networks unless the input graph is very dense
        (then `'randmio_und'` is used). 
    rwr : integer
        Number of approximate rewirings per edge (default 5). 

    Returns
    -------
    rnws : NumPy 3darray
        Random networks based on input graph `nw`. Format is
                `rnws.shape = (N,N,M)`
        s.t.
                `rnws[:,:,m] = m-th N x N` random network

    Notes
    -----
    This function requires `bctpy` to be installed! The `bctpy` package is 
    an unofficial Python port of the Brain Connectivity Toolbox for MATLAB. 
    It is currently available at the 
    `Python Package Index <https://pypi.python.org/pypi/bctpy>`_
    and can be installed using `pip`. 

    See also
    --------
    randmio_und_connected : in bctpy
    randmio_und : in bctpy
    """

    # Try to import bct
    try: import bct
    except: raise ImportError("Could not import bctpy! Consider installing it using pip install bctpy")

    # Sanity checks
    try:
        shn = nw.shape; N = shn[0]
    except:
        raise TypeError('Input must be a N-by-N NumPy array, not '+type(nw).__name__+'!')
    if len(shn) != 2:
        raise ValueError('Input must be a N-by-N NumPy array')
    if (min(shn)==1) or (shn[0]!=shn[1]):
        raise ValueError('Input must be a N-by-N NumPy array!')
    if np.isnan(nw).max()==True or np.isinf(nw).max()==True or np.isreal(nw).min()==False:
        raise ValueError('Input must be a real valued NumPy array without Infs or NaNs!')
    try: tmp = (round(M) != M) or (M < 1)
    except: raise TypeError("M has to be a natural number > 1!")
    if (tmp): raise ValueError("M has to be a natural number > 1!")

    if type(method).__name__ != 'str':
        raise TypeError("Input method must be a string, not "+type(method).__name__+"!")
    mthopts = ["auto","randmio_und_connected","randmio_und"]
    mthstr  = str(mthopts)
    mthstr  = mthstr.replace("(","")
    mthstr  = mthstr.replace(")","")
    try:
        mthopts.index(method)
    except: raise ValueError("Method has to be one of: "+mthstr)

    try:
        bad = (np.round(rwr) != rwr)
    except: TypeError("Rewiring parameter rwr has to be an integer, not "+type(rwr).__name__+"!")
    if bad or rwr < 1: 
        raise TypeError("Rewiring parameter has to be a strictly positve integer!")

    # Try to import progressbar module
    try: 
        import progressbar as pb
        showbar = True
    except: 
        print "WARNING: progressbar module not found - consider installing it using pip install progressbar"
        showbar = False

    # Allocate space for random networks and convert input network to list
    rnws = np.zeros((N,N,M))

    # Check if input network is fully connected
    if method == "auto":
        if density_und(nw) > .95:
            print "Network has maximal density. Using randmio_und instead of randmio_und_connected..."
            method = "randmio_und"
        else:
            method = "randmio_und_connected"

    # If available, initialize progressbar
    if (showbar): 
        widgets = ['Calculating Random Networks: ',pb.Percentage(),' ',pb.Bar(marker='#'),' ',pb.ETA()]
        pbar = pb.ProgressBar(widgets=widgets,maxval=M)

    # Populate tensor either using randmio (mthd = 0) or null_model_und (mthd = 1)
    if (showbar): pbar.start()
    if method == "randmio_und_connected":
        counter  = 0
        for m in xrange(M):
            rnws[:,:,m], eff = bct.randmio_und_connected(nw,rwr)
            counter += eff
            if (showbar): pbar.update(m)
    else:
        counter = 0
        for m in xrange(M):
            rnws[:,:,m] = bct.randmio_und(nw,rwr)[0]
            counter += eff
            if (showbar): pbar.update(m)
    if (showbar): pbar.finish()

    # If networks have not been randomized, let the user know
    if counter == 0:
        print "WARNING: Number of effective re-wirings is zero. Networks have not been randomized!"

    return rnws

##########################################################################################
def hdfburp(f):
    """
    Pump out everything stored in a HDF5 file. 

    Parameters
    ----------
    f : h5py file object
        File object created using `h5py.File()`

    Returns
    -------
    Nothing : None

    Notes
    -----
    This function takes an `h5py`-file object and creates variables in the caller's
    local namespace corresponding to the respective dataset-names in the file. 
    The naming format of the generated variable is 
        `groupname_datasetname`,
    where the `groupname` is empty for datasets in the `root` directory of the file. 
    Thus, if a HDF5 file contains the datasets
        `/a`

        `/b`

        `/group1/c`

        `/group1/d`

        `/group2/a`

        `/group2/b`

    then the code below creates the variables
        `a`

        `b`

        `group1_c`

        `group1_d`

        `group2_a`

        `group2_b`

    in the caller's workspace. 

    The black magic part of the code was taken from Pykler's answer to 
    `this stackoverflow question <http://stackoverflow.com/questions/2515450/injecting-variables-into-the-callers-scope>`_
    
    WARNING: EXISTING VARIABLES IN THE CALLER'S WORKSPACE WILL BE OVERWRITTEN!!!

    See also
    --------
    h5py : a Pythonic interface to the HDF5 binary data format.
    """

    # Sanity checks
    if str(f).find('HDF5 file') < 0:
        raise TypeError('Input must be a valid HDF5 file identifier!')

    # Initialize necessary variables
    mymap      = {}
    grplist    = [f]
    nameprefix = ''

    # As long as we find groups in the file, keep iterating
    while len(grplist) > 0:

        # Get current group (in the first iteration, that's just the file itself)
        mygrp = grplist[0]

        # If it actually is a group, extract the group name to prefix to variable names
        if len(mygrp.name) > 1:
            nameprefix = mygrp.name[1::]+'_'

        # Iterate through group items
        for it in mygrp.items():

            # If the current item is a group, add it to the list and keep going
            if str(it[1]).find('HDF5 group') >= 0:
                grplist.append(f[it[0]])

            # If we found a variable, name it following the scheme: groupname_varname 
            else:
                varname = nameprefix + it[0]
                mymap[varname] = it[1].value

        # Done with the current group, pop it from list
        grplist.pop(grplist.index(mygrp))

    # Update caller's variable scope (this is black magic...)
    stack = inspect.stack()
    try:
        locals_ = stack[1][0].f_locals
    finally:
        del stack
    locals_.update(mymap)

##########################################################################################
def normalize_time_series(time_series_array):
    """
    Normalizes a (real/complex) time series to zero mean and unit variance. 
    WARNING: Modifies the given array in place!
    
    Parameters
    ----------
    time_series_array : NumPy 2d array
        Array of data values per time point. Format is: `timepoints`-by-`N`

    Returns
    -------
    Nothing : None

    Notes
    -----
    This function does *not* do any error checking and assumes you know what you are doing
    This function is part of the `pyunicorn` package, developed by 
    Jonathan F. Donges and Jobst Heitzig. The package is currently available 
    `here <http://www.pik-potsdam.de/~donges/pyunicorn/index.html>`_

    See also
    --------
    pyunicorn : A UNIfied COmplex Network and Recurrence aNalysis toolbox

    Examples
    --------
    >>> ts = np.arange(16).reshape(4,4).astype("float")
    >>> normalize_time_series(ts)
    >>> ts.mean(axis=0)
    array([ 0.,  0.,  0.,  0.])
    >>> ts.std(axis=0)
    array([ 1.,  1.,  1.,  1.])
    >>> ts[:,0]
    array([-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079])
    """

    #  Remove mean value from time series at each node (grid point)
    time_series_array -= time_series_array.mean(axis=0)
    
    #  Normalize the variance of anomalies to one
    time_series_array /= np.sqrt((time_series_array * 
                                time_series_array.conjugate()).mean(axis=0))
        
    #  Correct for grid points with zero variance in their time series
    time_series_array[np.isnan(time_series_array)] = 0
    
##########################################################################################
def mutual_info(tsdata, n_bins=32, normalized=True, fast=True):
    """
    Calculate the (normalized) mutual information matrix at zero lag

    Parameters
    ----------
    tsdata : NumPy 2d array
        Array of data values per time point. Format is: `timepoints`-by-`N`. Note that 
        both `timepoints` and `N` have to be `>= 2` (i.e., the code needs at least two time-series 
        of minimum length 2)
    n_bins : int 
        Number of bins for estimating probability distributions
    normalized : bool
        If `True`, the normalized mutual information (NMI) is computed
        otherwise the raw mutual information (not bounded from above) is calculated
        (see Notes for details). 
    fast : bool
        Use C++ code to calculate (N)MI. If `False`, then 
        a (significantly) slower Python implementation is employed 
        (provided in case the compilation of the C++ code snippets 
        fails on a system)

    Returns
    -------
    mi : NumPy 2d array
        `N`-by-`N` matrix of pairwise (N)MI coefficients of the input time-series

    Notes
    -----
    For two random variables :math:`X` and :math:`Y` the raw mutual information 
    is given by 

    .. math:: MI(X,Y) = H(X) + H(Y) - H(X,Y),

    where :math:`H(X)` and :math:`H(Y)` denote the Shannon entropies of 
    :math:`X` and :math:`Y`, respectively, and :math:`H(X,Y)` is their joint 
    entropy. By default, this function normalizes the raw mutual information 
    :math:`MI(X,Y)` by the geometric mean of :math:`H(X)` and :math:`H(Y)`

    .. math:: NMI(X,Y) = {MI(X,Y)\over\sqrt{H(X)H(Y)}}.

    The heavy lifting in this function is mainly done by code parts taken from 
    the `pyunicorn` package, developed by Jonathan F. Donges  
    and Jobst Heitzig. It is currently available 
    `here <http://www.pik-potsdam.de/~donges/pyunicorn/index.html>`_
    The code has been modified so that weave and pure Python codes are now 
    part of the same function. Further, the original code computes the raw mutual information 
    only. Both Python and C++ parts have been extended to compute a normalized 
    mutual information too. 

    See also
    --------
    pyunicorn.pyclimatenetwork.mutual_info_climate_network : classes in this module

    Examples
    --------
    >>> tsdata = np.random.rand(150,2) # 2 time-series of length 150
    >>> NMI = mutual_info(tsdata)
    """

    # Sanity checks
    try:
        shtsdata = tsdata.shape
    except:
        raise TypeError('Input must be a timepoint-by-index NumPy 2d array, not '+type(tsdata).__name__+'!')
    if len(shtsdata) != 2:
        raise ValueError('Input must be a timepoint-by-index NumPy 2d array')
    if (min(shtsdata)==1):
        raise ValueError('At least two time-series/two time-points are required to compute (N)MI!')
    if np.isnan(tsdata).max()==True or np.isinf(tsdata).max()==True or np.isreal(tsdata).min()==False:
        raise ValueError('Input must be a real valued NumPy 2d array without Infs or NaNs!')

    try:
        tmp = (n_bins != int(n_bins))
    except:
        raise TypeError('Bin number must be an integer!')
    if (tmp): raise ValueError('Bin number must be an integer!')

    if type(normalized).__name__ != 'bool':
        raise TypeError('The normalized flag must be Boolean!')

    if type(fast).__name__ != 'bool':
        raise TypeError('The fast flag must be Boolean!')
    
    #  Get faster reference to length of time series = number of samples
    #  per grid point.
    (n_samples,N) = tsdata.shape

    #  Normalize tsdata time series to zero mean and unit variance
    normalize_time_series(tsdata)

    #  Initialize mutual information array
    mi = np.zeros((N,N), dtype="float32")

    # Execute C++ code
    if (fast):

        #  Create local transposed copy of tsdata
        tsdata = np.fastCopyAndTranspose(tsdata)
                
        #  Get common range for all histograms
        range_min = float(tsdata.min())
        range_max = float(tsdata.max())
        
        #  Re-scale all time series to the interval [0,1], 
        #  using the maximum range of the whole dataset.
        scaling  = float(1. / (range_max - range_min))
        
        #  Create array to hold symbolic trajectories
        symbolic = np.empty(tsdata.shape, dtype="int32")
        
        #  Initialize array to hold 1d-histograms of individual time series
        hist = np.zeros((N,n_bins), dtype="int32")
        
        #  Initialize array to hold 2d-histogram for one pair of time series
        hist2d = np.zeros((n_bins,n_bins), dtype="int32")
                
        # C++ code to compute NMI
        code_nmi = r"""
        int i, j, k, l, m;
        int symbol, symbol_i, symbol_j; 
        double norm, rescaled, hpl, hpm, plm, Hl, Hm;

        //  Calculate histogram norm
        norm = 1.0 / n_samples;

        for (i = 0; i < N; i++) {
            for (k = 0; k < n_samples; k++) {

                //  Calculate symbolic trajectories for each time series, 
                //  where the symbols are bins.
                rescaled = scaling * (tsdata(i,k) - range_min);

                if (rescaled < 1.0) {
                    symbolic(i,k) = rescaled * n_bins;
                }
                else {
                    symbolic(i,k) = n_bins - 1;
                }

                //  Calculate 1d-histograms for single time series
                symbol = symbolic(i,k);
                hist(i,symbol) += 1;
            }
        }

        for (i = 0; i < N; i++) {
            for (j = 0; j <= i; j++) {

                //  The case i = j is not of interest here!
                if (i != j) {
                    //  Calculate 2d-histogram for one pair of time series 
                    //  (i,j).
                    for (k = 0; k < n_samples; k++) {
                        symbol_i = symbolic(i,k);
                        symbol_j = symbolic(j,k);
                        hist2d(symbol_i,symbol_j) += 1;
                    }

                    //  Calculate mutual information for one pair of time 
                    //  series (i,j).
                    Hl = 0;
                    for (l = 0; l < n_bins; l++) {
                        hpl = hist(i,l) * norm;
                        if (hpl > 0.0) {
                            Hl += hpl * log(hpl);
                            Hm = 0;
                            for (m = 0; m < n_bins; m++) {
                                hpm = hist(j,m) * norm;
                                if (hpm > 0.0) {
                                    Hm += hpm * log(hpm);
                                    plm = hist2d(l,m) * norm;
                                    if (plm > 0.0) {
                                        mi(i,j) += plm * log(plm/hpm/hpl);
                                    }
                                }
                            }
                        }
                    }

                    // Divide by the marginal entropies to normalize MI
                    mi(i,j) = mi(i,j) / sqrt(Hm * Hl);

                    //  Symmetrize MI
                    mi(j,i) = mi(i,j);

                    //  Reset hist2d to zero in all bins
                    for (l = 0; l < n_bins; l++) {
                        for (m = 0; m < n_bins; m++) {
                            hist2d(l,m) = 0;
                        }
                    }
                }
                // Put ones on the diagonal
                else {
                    mi(i,j) = 1.0;
                }

            }
        }
        """

        # C++ code to compute MI
        code_mi = r"""
        int i, j, k, l, m;
        int symbol, symbol_i, symbol_j; 
        double norm, rescaled, hpl, hpm, plm;

        //  Calculate histogram norm
        norm = 1.0 / n_samples;

        for (i = 0; i < N; i++) {
            for (k = 0; k < n_samples; k++) {

                //  Calculate symbolic trajectories for each time series, 
                //  where the symbols are bins.
                rescaled = scaling * (tsdata(i,k) - range_min);

                if (rescaled < 1.0) {
                    symbolic(i,k) = rescaled * n_bins;
                }
                else {
                    symbolic(i,k) = n_bins - 1;
                }

                //  Calculate 1d-histograms for single time series
                symbol = symbolic(i,k);
                hist(i,symbol) += 1;
            }
        }

        for (i = 0; i < N; i++) {
            for (j = 0; j <= i; j++) {

                //  The case i = j is not of interest here!
                if (i != j) {
                    //  Calculate 2d-histogram for one pair of time series 
                    //  (i,j).
                    for (k = 0; k < n_samples; k++) {
                        symbol_i = symbolic(i,k);
                        symbol_j = symbolic(j,k);
                        hist2d(symbol_i,symbol_j) += 1;
                    }

                    //  Calculate mutual information for one pair of time 
                    //  series (i,j).
                    // Hl = 0;
                    for (l = 0; l < n_bins; l++) {
                        hpl = hist(i,l) * norm;
                        if (hpl > 0.0) {
                            // Hl += hpl * log(hpl);
                            // Hm = 0;
                            for (m = 0; m < n_bins; m++) {
                                hpm = hist(j,m) * norm;
                                if (hpm > 0.0) {
                                    // Hm += hpm * log(hpm);
                                    plm = hist2d(l,m) * norm;
                                    if (plm > 0.0) {
                                        mi(i,j) += plm * log(plm/hpm/hpl);
                                    }
                                }
                            }
                        }
                    }

                    //  Symmetrize MI
                    mi(j,i) = mi(i,j);

                    //  Reset hist2d to zero in all bins
                    for (l = 0; l < n_bins; l++) {
                        for (m = 0; m < n_bins; m++) {
                            hist2d(l,m) = 0;
                        }
                    }
                }
                // Put ones on the diagonal
                else {
                    mi(i,j) = 1.0;
                }

            }
        }
        """

        # Initialize necessary variables to pass on to C++ code snippets above
        vars = ['tsdata', 'n_samples', 'N', 'n_bins', 'scaling', 'range_min', 
                'symbolic', 'hist', 'hist2d', 'mi']

        # Compute normalized or non-normalized mutual information
        if (normalized):
            weave.inline(code_nmi, vars, type_converters=weave.converters.blitz, 
                         compiler='gcc', extra_compile_args=['-O3'])
        else:
            weave.inline(code_mi, vars, type_converters=weave.converters.blitz, 
                         compiler='gcc', extra_compile_args=['-O3'])
        
    # Python version of (N)MI computation (slower)
    else:

        #  Define references to numpy functions for faster function calls
        histogram = np.histogram
        histogram2d = np.histogram2d
        log = np.log 

        #  Get common range for all histograms
        range_min = tsdata.min()
        range_max = tsdata.max()

        #  Calculate the histograms for each time series
        p = np.zeros((N, n_bins))
        for i in xrange(N):
            p[i,:] = (histogram(tsdata[:, i], bins=n_bins, 
                            range=(range_min,range_max))[0]).astype("float64")

        #  Normalize by total number of samples = length of each time series
        p /= n_samples

        #  Make sure that bins with zero estimated probability are not counted 
        #  in the entropy measures.
        p[p == 0] = 1

        #  Compute the information entropies of each time series
        H = - (p * log(p)).sum(axis = 1)

        #  Calculate only the lower half of the MI matrix, since MI is 
        #  symmetric with respect to X and Y.
        for i in xrange(N):

            for j in xrange(i):

                #  Calculate the joint probability distribution
                pxy = (histogram2d(tsdata[:,i], tsdata[:,j], bins=n_bins, 
                            range=((range_min, range_max), 
                            (range_min, range_max)))[0]).astype("float64")

                #  Normalize joint distribution
                pxy /= n_samples

                #  Compute the joint information entropy
                pxy[pxy == 0] = 1
                HXY = - (pxy * log(pxy)).sum()

                # Normalize by entropies (or not)
                if (normalized):
                    mi.itemset((i,j), (H.item(i) + H.item(j) - HXY)/(np.sqrt(H.item(i)*H.item(j))))
                else:
                    mi.itemset((i,j), H.item(i) + H.item(j) - HXY)

                # Symmetrize MI
                mi.itemset((j,i), mi.item((i,j)))

        # Put ones on the diagonal
        np.fill_diagonal(mi,1)

    # Return (N)MI matrix
    return mi

##########################################################################################
def get_numlines(fname):
    """
    Get number of lines of a txt-file

    Inputs:
    -------
    fname : str
        Path to file to be read

    Returns:
    --------
    lineno : int
        Number of lines in the file

    Notes:
    ------
    This code was written by Mark Byers as part of a Stackoverflow submission, 
    see .. http://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python

    See also:
    ---------
    None
    """

    # Check if input makes sense
    if type(fname).__name__ != "str":
        raise TypeError("Filename has to be a string!")

    # Cycle through lines of files and do nothing
    with open(fname) as f:
        for lineno, l in enumerate(f):
            pass

    return lineno + 1

##########################################################################################
def issym(A,tol=1e-9):
    """
    Check for symmetry of a 2d NumPy array A

    Inputs:
    -------
    A : square NumPy 2darray
        A presumably symmetric matrix
    tol : positive real scalar
        Tolerance for checking if (A - A.T) is sufficiently small

    Returns:
    --------
    is_sym : bool
        True if A satisfies
                |A - A.T| <= tol * |A|,
        where |.| denotes the Frobenius norm. Thus, if the above inequality 
        holds, A is approximately symmetric. 

    Notes:
    ------
    An absolute-value based comparison is readily provided by NumPy's isclose

    See also:
    ---------
    The following thread at MATLAB central
    .. http://www.mathworks.com/matlabcentral/newsreader/view_thread/252727
    """

    # Check if Frobenius norm of A - A.T is sufficiently small (respecting round-off errors)
    try:
        is_sym = (norm(A-A.T,ord='fro') <= tol*norm(A,ord='fro'))
    except:
        raise TypeError('Input argument has to be a square matrix and a scalar tol (optional)!')

    return is_sym

##########################################################################################
def myglob(flpath,spattern):
    """
    Return a glob-like list of paths matching a pathname pattern BUT support fancy shell syntax

    Parameters
    ----------
    flpath : str
        Path to search (to search current directory use `flpath=''` or `flpath='.'`
    spattern : str
        Pattern to search for in `flpath`

    Returns
    -------
    flist : list
        A Python list of all files found in `flpath` that match the input pattern `spattern`

    Examples
    --------
    List all png/PNG files in the folder `MyHolidayFun` found under `Documents`

    >> filelist = myglob('Documents/MyHolidayFun','*.[Pp][Nn][Gg]')
    >> print filelist
    >> ['Documents/MyHolidayFun/img1.PNG','Documents/MyHolidayFun/img1.png']
        
    Notes
    -----
    None

    See also
    --------
    glob
    """

    # Sanity checks
    if type(flpath).__name__ != 'str':
        raise TypeError('Input has to be a string specifying a path!')
    if type(spattern).__name__ != 'str':
        raise TypeError('Pattern has to be a string!')

    # If user wants to search current directory, make sure that works as expected
    if (flpath == '') or (flpath.count(' ') == len(flpath)):
        flpath = '.'

    # Append trailing slash to filepath
    else:
        if flpath[-1] != os.sep: flpath = flpath + os.sep

    # Return glob-like list
    return [os.path.join(flpath, fnm) for fnm in fnmatch.filter(os.listdir(flpath),spattern)]

##########################################################################################
def printdata(data,leadrow,leadcol,fname=None):
    """
    Pretty-print/-save array-like data

    Parameters
    ----------
    data : NumPy 2darray
        An `M`-by-`N` array of data
    leadrow : Python list or NumPy 1darray
        List/array of length `M` providing labels to be printed in the first column of the table
        (strings/numerals or both)
    leadcol : Python list or NumPy 1darray
        List/array of length `N` or `N+1` providing labels to be printed in the first row of the table
        (strings/numerals or both). See Examples for details
    fname : string
        Name of a csv-file (with or without extension `.csv`) used to save the table 
        (WARNING: existing files will be overwritten!). Can also be a path + filename 
        (e.g., `fname='path/to/file.csv'`). By default output is not saved. 

    Returns
    -------
    Nothing : None

    Notes
    -----
    Uses the `texttable` module to print results

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(2,3)
    >>> row1 = ["a","b",3]
    >>> col1 = np.arange(2)
    >>> printdata(data,row1,col1)
    +--------------------+--------------------+--------------------+--------------------+
    |                    |         a          |         b          |         3          |
    +====================+====================+====================+====================+
    | 0                  | 0.994018537964     | 0.707532139166     | 0.767497407803     |
    +--------------------+--------------------+--------------------+--------------------+
    | 1                  | 0.914193045048     | 0.758181936461     | 0.216752553325     |
    +--------------------+--------------------+--------------------+--------------------+
    >>> row1 = ["labels"] + row1
    >>> printdata(data,row1,col1,fname='dummy')
    +--------------------+--------------------+--------------------+--------------------+
    |       labels       |         a          |         b          |         3          |
    +====================+====================+====================+====================+
    | 0                  | 0.994018537964     | 0.707532139166     | 0.767497407803     |
    +--------------------+--------------------+--------------------+--------------------+
    | 1                  | 0.914193045048     | 0.758181936461     | 0.216752553325     |
    +--------------------+--------------------+--------------------+--------------------+
    >>> cat dummy.csv
    labels, a, b, 3
    0,0.994018537964,0.707532139166,0.767497407803
    1,0.914193045048,0.758181936461,0.216752553325

    See also
    --------
    texttable : a module for creating simple ASCII tables (currently available at the 
                `Python Package Index <https://pypi.python.org/pypi/texttable/0.8.1>`_)
    """

    # Try to import Texttable object
    try: from texttable import Texttable
    except: 
        raise ImportError("Could not import texttable! Consider installing it using pip install texttable")

    # Sanity checks
    try:
        ds = data.shape
    except:
        raise TypeError('Input must be a M-by-N NumPy array, not ' + type(data).__name__+'!')
    if len(ds) > 2:
        raise ValueError('Input must be a M-by-N NumPy array!')

    try:
        m = len(leadcol)
    except: 
        raise TypeError('Input must be a Python list or NumPy 1d array, not '+type(leadcol).__name__+'!')
    try:
        n = len(leadrow)
    except: 
        raise TypeError('Input must be a Python list or NumPy 1d array, not '+type(leadrow).__name__+'!')

    if fname != None:
        if type(fname).__name__ != 'str':
            raise TypeError('Input fname has to be a string specifying an output filename, not '\
                            +type(fname).__name__+'!')
        if fname[-4::] != '.csv':
            fname = fname + '.csv'
        save = True
    else: save = False

    # Get dimension of data and corresponding leading column/row
    if len(ds) == 1: 
        K = ds[0]
        if K == m:
            N = 1; M = K
        elif K == n or K == (n-1):
            M = 1; N = K
        else: 
            raise ValueError('Number of elements in heading column/row and data don not match up!')
        data = data.reshape((M,N))
    else:
        M,N = ds

    if M != m:
        raise ValueError('Number of rows and no. of elements leading column do not match up!')
    elif N == n:
        head = [' '] + list(leadrow)
    elif N == (n-1):
        head = list(leadrow)
    else:
        raise ValueError('Number of columns and no. of elements in head row do not match up!')

    # Do something: create big data array including leading column
    Data = np.column_stack((leadcol,data.astype('str')))
    
    # Initialize table object and fill it with stuff
    table = Texttable()
    table.set_cols_align(["l"]*(N+1))
    table.set_cols_valign(["c"]*(N+1))
    table.set_cols_dtype(["t"]*(N+1))
    table.set_cols_width([18]*(N+1))
    table.add_rows([head],header=True)
    table.add_rows(Data.tolist(),header=False)
    
    # Pump out table
    print table.draw() + "\n"

    # If wanted, save stuff in a csv file
    if save:
        head = str(head)
        head = head.replace("[","")
        head = head.replace("]","")
        head = head.replace("'","")
        np.savetxt(fname,Data,delimiter=",",fmt="%s",header=head,comments="")

    return

##########################################################################################
def tensorcheck(corrs):
    """
    Local helper function performing sanity checks on a N-by-N-by-k tensor
    """

    try:
        shc = corrs.shape
    except:
        raise TypeError('Input must be a N-by-N-by-k NumPy array, not '+type(corrs).__name__+'!')
    if len(shc) != 3:
        raise ValueError('Input must be a N-by-N-by-k NumPy array')
    if (min(shc[0],shc[1])==1) or (shc[0]!=shc[1]):
        raise ValueError('Input must be a N-by-N-by-k NumPy array!')
    if np.isnan(corrs).max()==True or np.isinf(corrs).max()==True or np.isreal(corrs).min()==False:
        raise ValueError('Input must be a real valued NumPy array without Infs or NaNs!')
