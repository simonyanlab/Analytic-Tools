# nws_tools.py - Collection of network processing/analysis routines
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# September 25 2013

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from glob import glob 
import natsort
import os
import csv

from mypy.recipes import get_numlines, issym

##########################################################################################
def degrees_und(CIJ):
    """
    Compute nodal degree of undirected graph

    Inputs:
    -------
    CIJ : NumPy 2darray
        Undirected binary/weighted connection matrix

    Returns:
    --------
    deg : NumPy 1darray
        Nodal degree vector

    Notes:
    ------
    This function does *not* do any error checking and assumes you know what you are doing

    See also:
    ---------
    degrees_und.m in the Brain Connectivity Toolbox for MATLAB, currently available at
    .. https://sites.google.com/site/bctnet/
    
    A C++ version of the BCT can be found on the same site. Python bindings are provided 
    in the module bct_py/bct_gsl

    An inofficial Python port of the BCT is currently available at the Python Package Index 
    and can be installed using pip. 
    """

    return (CIJ != 0).sum(1)

##########################################################################################
def density_und(CIJ):
    """
    Compute density of undirected graph

    Inputs:
    -------
    CIJ : NumPy 2darray
        Undirected binary/weighted connection matrix

    Returns:
    --------
    den : float
        density (fraction of present connections to possible connections)

    Notes:
    ------
    This function does *not* do any error checking and assumes you know what you are doing

    See also:
    ---------
    density_und.m in the Brain Connectivity Toolbox for MATLAB, currently available at
    .. https://sites.google.com/site/bctnet/
    
    A C++ version of the BCT can be found on the same site. Python bindings are provided 
    in the module bct_py/bct_gsl

    An inofficial Python port of the BCT is currently available at the Python Package Index 
    and can be installed using pip. 
    """

    N = CIJ.shape[0]                    # no. of nodes
    K = (np.triu(CIJ,1)!=0).sum()       # no. of edges
    return K/((N**2 - N)/2.0)

##########################################################################################
def get_corr(txtpath):
    """
    Compute correlation matrices of time-series using Pearson's correlation coefficient

    Inputs:
    -------
    txtpath : string
        Path to directory holding voxelwise time-series dumped in txt files.
        The following file-naming convention HAS to be obeyed:
                sNxy_bla_bla.txt,
        where N is the group id (1,2,3,...), xy denotes the subject number 
        (01,02,...,99 or 001,002,...,999) and anything else is separated 
        by an underscore. The files will be read in lexicographic order,
        i.e., s101_1.txt, s101_2.txt,... or s101_Amygdala.txt, s101_Beemygdala,...
       
    Returns:
    --------
    corrs : NumPy 3darray
        NxN correlation matrices of numsubs subjects. Format is 
                corrs.shape = (N,N,numsubs),
        s.t.
                corrs[:,:,i] = N x N correlation matrix of i-th subject 
    bigmat : NumPy 3darray
        Tensor holding unprocessed time series of all subjects. Format is 
                bigmat.shape = (tlen,N,numsubs),
        where tlen is the length of the time-series and N is the number of                 
        regions (=nodes in a network). 
    sublist : list of strings
        List of subjects found in folder specified by txtpath, e.g.,
                sublist = ['s101','s103','s110','s111','s112',...]

    Notes:
    ------
    None

    See also:
    ---------
    get_corr.m and references therein and
    NumPy's corrcoef
    """

    # Sanity checks 
    if type(txtpath).__name__ != 'str':
        raise TypeError('Input has to be a string specifying the path to the txt-file directory!')

    # Get list of all txt-files in txtpath and order them lexigraphically
    if txtpath[-1] == ' '  or txtpath[-1] == os.sep: txtpath = txtpath[:-1]
    txtfiles = natsort.natsorted(glob(txtpath+os.sep+"*.txt"), key=lambda y: y.lower())

    # Load very first file to get length of time-series
    firstsub = txtfiles[0]
    tlen     = get_numlines(firstsub)

    # Search from left in filename for first "s" (naming scheme: sNxy_bla_bla_.txt)
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

        # Saveguard: stop if subject is missing, i.e., col = 0 still (weirder things have happened...)
        if col == 0: 
            raise ValueError("Subject "+sub+" is missing!")

        # Compute correlation coefficients for current subject
        corrs[:,:,k] = np.corrcoef(bigmat[:,:,k],rowvar=0)

    # Happy breakdown
    return bigmat, corrs, sublist

##########################################################################################
def corrcheck(*args,**kwargs):
    """
    Sanity checks for correlation matrices
    
    Inputs:
    -------
    Dynamic

    Usage:
    ------
    corrcheck(A)                    shows some statistics for the correlation matrix A.
    corrcheck(A,label)              shows some statistics for the correlation matrix A and uses
                                    the list of strings label as title in figures. 
    corrcheck(A,B,C,...)            shows some statistics for the correlation matrices A,B,C,....
    corrcheck(A,B,C,...,label)      shows some statistics for the correlation matrices A,B,C,....
                                    and uses the list of strings label to generate titles in figures. 
                                    Note that len(label) has to be equal to the number of 
                                    input matrices. 
    corrcheck(T)                    shows some statistics for correlation matrices stored 
                                    in the tensor T. The storage scheme is 
                                        T[:,:,0] = A
                                        T[:,:,1] = B
                                        T[:,:,2] = C
                                        etc.
                                    where A,B,C,... are correlation matrices. 
    corrcheck(T,label)              shows some statistics for correlation matrices stored 
                                    in the tensor T. The storage scheme is 
                                        T[:,:,0] = A
                                        T[:,:,1] = B
                                        T[:,:,2] = C
                                        etc.
                                    where A,B,C,... are correlation matrices. The list of strings label 
                                    is used to generate titles in figures. Note that len(label)
                                    has to be equal to T.shape[2]
    corrcheck(...,title='mytitle')  same as above and and uses the string 'mytitle' as window name for 
                                    figures. 
                               
    Returns:
    --------
    None 
    
    Notes
    -----
    None
    
    See also
    --------
    corrcheck.m and references therein
    """

    # Plotting params used later (max. #plots per row)
    cplot = 5

    # Sanity checks
    myin = len(args)
    if myin == 0: raise ValueError('At least one input required!')

    # Assign global name for all figures if provided by additional keyword argument 'title'
    figtitle = kwargs.get('title',None); nofigname = False
    if figtitle == None: nofigname = True

    # If labels have been provided, exract them now
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
        if (min(shv) == 1) or (shv[0]!=shv[1]):
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
        im = plt.imshow(corrs[:,:,i],cmap='jet',interpolation='nearest',vmin=-1,vmax=1)
        plt.axis('off')
        plt.title(labels[i])
        if issym(corrs[:,:,i]) == False:
            print "WARNING: "+labels[i]+" is not symmetric!"
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.draw()

    # Plot correlation histograms
    idx = np.nonzero(np.triu(np.ones((N,N)),1))
    NN  = (N**2 - N)/2
    fig = plt.figure(figsize=(8,8))
    if nofigname: figtitle = fig.canvas.get_window_title()
    fig.canvas.set_window_title(figtitle+': '+"Correlation Histograms")
    bars = []; ylims = []
    for i in xrange(nmat):
        cvec = corrs[idx[0],idx[1],i]
        [corrcount,corrbins] = np.histogram(cvec,bins=20,range=(-1,1))
        bars.append(plt.subplot(rplot,cplot,i+1))
        plt.bar(corrbins[:-1],corrcount/NN,width=np.abs(corrbins[0]-corrbins[1]))
        ylims.append(bars[-1].get_ylim()[1])
        plt.xlim(-1,1)
        plt.xticks((-1,0,1),('-1','0','1'))
        plt.title(labels[i])
        if np.mod(i+1,cplot) == 1: plt.ylabel('Frequency')
    ymax = max(ylims)
    for mybar in bars: mybar.set_ylim(top=ymax)
    plt.draw()

    # Show negative correlations
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
    Helper function to compute mean networks

    Inputs:
    -------
    nws : NumPy 3darray
        NxN connectivity matrices of numsubs subjects. Format is 
                nws.shape = (N,N,numsubs),
        s.t.
                nws[:,:,i] = N x N conn. matrix of i-th subject 
    percval : float
        Percentage value, s.t. connections not present in at least percval
        percent of subjects are not considered, thus 0 <= percval <= 1.
        Default setting is percval = 0.75 (following Sporns' rich club
        paper).
       
    Returns:
    --------
    mean_wghted : NumPy 2darray
        NxN mean value matrix of numsubs matrices stored in nws where
        only connections present in at least percval percent of subjects
        are considered
       
    Notes:
    ------
    None
    
    See also:
    ---------
    get_meannw.m and 
    Sporns' rich club paper, currently available at
    .. http://www.jneurosci.org/content/31/44/15775.full
    """

    # Sanity checks
    tensorcheck(nws)
    try:
        if percval > 1 or percval < 0:
            raise ValueError("Percentage value must be >= 0 and <= 1!")
    except: raise TypeError("Percentage value must be a floating point number >= 0 and <= 1!")
    
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

    return mean_wghted

##########################################################################################
def rm_negatives(corrs):
    """
    Remove negative corelations from correlation matrices

    Inputs:
    -------
    corrs : NumPy 3darray
        NxN correlation matrices of numsubs subjects. Format is 
                corrs.shape = (N,N,numsubs),
        s.t.
                corrs[:,:,i] = N x N conn. matrix of i-th subject 

    Returns:
    --------
    corrs : NumPy 3darray
        Same format as input tensor but corrs >= 0. 

    Notes
    -----
    None

    See also:
    ---------
    rm_negatives.m
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
            print "WARNING: In sujbect "+str(i)+" node(s) "+str(np.nonzero(deg==deg.min())[0])+" got disconnected!"

    return nws

##########################################################################################
def thresh_nws(nws,userdens=None):
    """
    Threshold networks based on connection density

    Inputs:
    -------
    nws : NumPy 3darray
        Undirected NxN connectivity matrices of numsubs subjects. Format is 
                corrs.shape = (N,N,numsubs),
        s.t.
                corrs[:,:,i] = N x N conn. matrix of i-th subject 
    userdens : int
        Density level to which networks should be thresholded,i.e., 0 < userdens < 100
               
    Returns:
    --------
    th_nws : NumPy 3darray
        Thresholded networks for all subjects. Format is the same as nws. 
    tau_levels : NumPy 1darray
        The threshold values for each subject's network corresponding to the 
        networks stored in th_nws, i.e. tau_levels[i] is the threshold that 
        generated the network th_nws[:,:,i], i.e., the network of subject i. 
    den_values : NumPy 1darray
        Same format as tau_levels but holding the density values for each subject
    th_mnw : NumPy 2darray
        The group averaged (across all subjects) weighted network

    Notes
    -----
    Uses get_meannw to compute mean network

    See also
    --------
    binarize.m and thresh_nws.m
    """

    # Sanity checks
    tensorcheck(nws)
    if userdens != None:
        try:
            if np.round(userdens) != userdens:
                raise ValueError('The density level must be an integer!')
                if (userdens <= 0) or (userdens >= 100):
                    raise ValueError('The density level must be between 0 and 100!')
        except: raise TypeError('Density level has to be a number between 0 and 100!')

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
        th_nws = nws; tau_levels = None; den_values = raw_den; threshmnw = get_meannw(nws) 
        return th_nws, tau_levels, den_values, th_mnw

    # Inform user about minimal/maximal density in raw data
    print "\nRaw data has following density values: \n"
    print "\tMinimal density: "+str(min_raw)+"%"
    print "\tMaximal density: "+str(max_raw)+"%"

    # Create vector of thresholds to iterate on
    dt      = 1e-3
    threshs = np.arange(0,1+dt,dt)

    # Allocate space for output
    th_nws     = np.zeros(nws.shape)
    tau_levels = np.zeros((numsubs,))
    den_values = np.zeros((numsubs,))
    th_mnw     = np.zeros((N,N))

    # Cycle through subjects and threshold the connectivity matrices until a node disconnects
    for i in xrange(numsubs):
        tau = -1
        den = 1
        mnw = nws[:,:,i]

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
        # return th_nws, tau_levels, den_values, th_mnw
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
    th_mnw = get_meannw(th_nws)

    # Be polite and dismiss the user 
    print "Done...\n"

    return th_nws, tau_levels, den_values, th_mnw

##########################################################################################
def normalize(I,a=0,b=1):
    """
    Rescales a numpy ndarray

    Inputs:
    -------
    I: NumPy ndarray
        An array of size > 1 (shape can be arbitrary)
    a : float
        Floating point number being the lower normalization bound. 
        By default a = 0. (Note that it has to hold that a < b)
    b : float
        Floating point number being the upper normalization bound. 
        By default b = 1. (Note that it has to hold that a < b)
       
    Returns
    -------
    In : NumPy ndarray
        Scaled version of the input array I, such that a = In.min() and 
        b = In.max()

    Notes
    -----
    None 

    Examples:
    ---------
    I = array([[-1,.2],[100,0]])
    In = normalize(I,a=-10,b=12)
    In 
    array([[-10.        ,  -9.73861386],
           [ 12.        , -10.        ]])

    See also
    --------
    None 
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
        elif np.absolute(a - b) < np.finfo(float).eps:
            raise ValueError('|a-b|<eps, no normalization possible')
    except TypeError: raise TypeError('a and b have to be scalars satisfying a < b!')

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
    Converts a csv file of nodal coordinates to a Python dictionary

    Inputs:
    -------
    csvfile : string 
        Filename (or path) of the csv file holding the nodal coordinates.
        The format of this file HAS to be 
                 x, y, z 
                 x, y, z 
                 x, y, z 
                 .
                 .
        for each node. Thus #rows = #nodes. 

    Returns:
    --------
    mydict : dictionary 
        Nodal coordinates of as read from the input csv file. Format is
                {0: (x, y, z),
                {1: (x, y, z),
                {2: (x, y, z),
                 .
                 .
        Thus the dictionary has #nodes keys. 
    
    Notes:
    ------
    None 

    See also:
    ---------
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
    Plots a 3d graph using Mayavi

    Inputs:
    -------
    A : NumPy 2darray
        Square N-by-N connectivity matrix of the graph
    coords: dictionary 
        Nodal coordinates of the graph. Format is
                {0: (x, y, z),
                {1: (x, y, z),
                {2: (x, y, z),
                 .
                 .
        Note that the dictionary has to have N keys. 
    colorvec : NumPy 1darray
        Vector of colorvalues for each node. This could nodal strength or modular information of nodes, 
        i.e., to which module does node i belong to. Thus colorvec has to be of length N. 
    sizevec : NumPy 1darray 
        Vector of nodal sizes. This could be degree, centrality, etc. Thus colorvec has to be of length N. 
    labels : list
        Nodal labels. Format is ['Name1','Name2','Name3',...] where the ordering HAS to be the same
        as in the coords dictionary. Note that the list has to have length N. 
    threshs : list 
        List of thresholds for visualization. Edges with weights larger than threshs[0] are drawn 
        thickest, weights > threshs[1] are thinner and so on. Note that if threshs[-1]>0 not all 
        edges of the network are plotted (since edges with 0<weight<threshs[-1] will be ignored). 
    lwdths : list 
        List of linewidths associated to the list of thresholds. Edges with weights larger than 
        threshs[0] are drawn with linewidth lwdths[0], edges with weights > threshs[1] have linewidth 
        lwdths[1] and so on. Thus len(lwdths) == len(threshs). 
    nodecmap : string 
        Colormap to use for plotting nodes. 
    edgecmap : string 
        Colormap to use for plotting edges. 
    textscale : positive number 
        Scaling factor for labels

    Returns:
    --------
    None 

    Notes:
    ------
    See the Mayavi docu for more info. Currently available at 
    .. http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html

    See also:
    ---------
    A matplotlib based implementation with extended functionality (but MUCH slower rendering) is show_nw
    """

    # For those lucky enough to have a running installation of Mayavi...
    try:
        from mayavi import mlab
    except: 
        msg = 'Mayavi could not be imported. You might want to try show_nw, a slower (but more feature '\
              +'rich) graph rendering routine based on matplotlib.'
        raise ImportError(msg)

    # Sanity checks and assign default values
    try:
        (N,M) = A.shape
        if N != M: raise ValueError('A has to be square!')
        if np.isnan(A).max() == True or np.isinf(A).max() == True or np.isreal(A).min() == False:
            raise ValueError("A must be real-valued without NaNs or Infs!")
    except: raise TypeError('A has to be a square NumPy array!')
    if type(coords).__name__ != 'dict':
        raise TypeError("The coordinates have to be a dictionary!")
        if len(coords.keys()) != N:
            raise ValueError('The coordinate dictionary has to have N keys!')

    if colorvec != None:
        try: 
            if colorvec.size != N:
                raise ValueError('colorvec has to have length N!')
        except: raise TypeError('colorvec has to be a NumPy array!')
        if np.isnan(colorvec).max() == True or np.isinf(colorvec).max() == True or np.isreal(colorvec).min()==False:
            raise ValueError("colorvec must be real-valued without NaNs or Infs!")
        if colorvec.min() < 0 or colorvec.max() > 1:
            raise ValueError('colorvec values must be between 0 and 1!')
    else: colorvec = np.ones((N,))

    if sizevec != None:
        try: 
            if sizevec.size != N:
                raise ValueError('sizevec has to have length N!')
        except: raise TypeError('sizevec has to be a NumPy array!')
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
    Matplotlib-based plotting routine for networks

    Inputs:
    -------
    A : NumPy 2darray
        Square N-by-N connectivity matrix of the graph
    coords: dictionary 
        Nodal coordinates of the graph. Format is
                {0: (x, y, z),
                {1: (x, y, z),
                {2: (x, y, z),
                 .
                 .
        Note that the dictionary has to have N keys. 
    colorvec : NumPy 1darray
        Vector of colorvalues for each node. This could nodal strength or modular information of nodes, 
        i.e., to which module does node i belong to. Thus colorvec has to be of length N and all its
        components must be in [0,1]. 
    sizevec : NumPy 1darray 
        Vector of nodal sizes. This could be degree, centrality, etc. Thus colorvec has to be of length N
        and all its components must be >= 0. 
    labels : list
        Nodal labels. Format is ['Name1','Name2','Name3',...] where the ordering HAS to be the same
        as in the coords dictionary. Note that the list has to have length N. 
    nodecmap : matplotlib colormap
        Colormap to use for plotting nodes
    edgecmap : matplotlib colormap
        Colormap to use for plotting edges
    linewidths : NumPy 2darray
        Same format and nonzero-pattern as A. If no linewidhts are provided then the edge connecting 
        nodes v_i and v_j is plotting using the linewidth A[i,j]. Using, e.g., linewidhts = (1+A)**2 
        the thickness of edges in the plot can be scaled. 
    nodes3d : bool
        If nodes3d=True then nodes are plotted using 3d spheres in space (with diameter=sizevec). If 
        nodes3d=False then the matplotlib scatter function is used to plot nodes as flat 2d disks (faster).
    viewtype : str
        Camera position. Viewtype can be one of the following
                axial (= axial_t)       : Axial view from top down
                axial_t                 : Axial view from top down
                axial_b                 : Axial view from bottom up
                sagittal (= sagittal_l) : Sagittal view from left
                sagittal_l              : Sagittal view from left
                sagittal_r              : Sagittal view from right
                coronal (= coronal_f)   : Coronal view from front
                coronal_f               : Coronal view from front
                coronal_b               : Coronal view from back
    
    Returns:
    --------
    None

    Notes:
    ------
    See matplotlib's mplot3d tutorial. Currently available at
    .. http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

    See also:
    ---------
    A Mayavi based implementation with less functionality but MUCH faster rendering is shownet
    """

    # Sanity checks and assign default values
    try:
        (N,M) = A.shape
        if N != M: raise ValueError('A has to be square!')
        if np.isnan(A).max() == True or np.isinf(A).max() == True or np.isreal(A).min() == False:
            raise ValueError("A must be real-valued without NaNs or Infs!")
    except: raise TypeError('A has to be a square NumPy array!')
    if type(coords).__name__ != 'dict':
        raise TypeError("The coordinates have to be a dictionary!")
        if len(coords.keys()) != N:
            raise ValueError('The coordinate dictionary has to have N keys!')

    if colorvec != None:
        try: 
            if colorvec.size != N:
                raise ValueError('colorvec has to have length N!')
        except: raise TypeError('colorvec has to be a NumPy array!')
        if np.isnan(colorvec).max() == True or np.isinf(colorvec).max() == True or np.isreal(colorvec).min()==False:
            raise ValueError("colorvec must be real-valued without NaNs or Infs!")
        if colorvec.min() < 0 or colorvec.max() > 1:
            raise ValueError('colorvec values must be between 0 and 1!')
    else: colorvec = np.ones((N,))

    if sizevec != None:
        try: 
            if sizevec.size != N:
                raise ValueError('sizevec has to have length N!')
        except: raise TypeError('sizevec has to be a NumPy array!')
        if np.isnan(sizevec).max() == True or np.isinf(sizevec).max() == True or np.isreal(sizevec).min()==False:
            raise ValueError("sizevec must be real-valued without NaNs or Infs!")
        if sizevec.min() < 0:
            raise ValueError('sizevec values must be >= 0!')
    else: sizevec = np.ones((N,))

    if type(labels).__name__ != 'list':
        raise TypeError("Nodal labels have to be provided as list!")

    if type(nodecmap).__name__ != 'LinearSegmentedColormap':
        raise TypeError('Nodal colormap has to be a matplotlib colormap!')

    if type(edgecmap).__name__ != 'LinearSegmentedColormap':
        raise TypeError('Edge colormap has to be a matplotlib colormap!')

    if linewidths != None:
        try:
            (N,M) = linewidths.shape
            if N != M: raise ValueError('Linewidths have to be provided as square matrix!')
            if np.isnan(linewidths).max() == True or np.isinf(linewidths).max() == True or np.isreal(linewidths).min() == False:
                raise ValueError("Linewidths must be real-valued without NaNs or Infs!")
        except: raise TypeError('Linewidths have to be a square NumPy array!')

    if type(nodes3d).__name__ != 'bool':
        raise TypeError('The nodes3d flag has to be a boolean variable!')

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
def generate_randnws(nw,M=100):
    """
    Generate random networks given an input network

    Inputs:
    -------
    nw : NumPy 2darray
        Undirected binary/weighted connection matrix
    M : integer > 1
        Number of random networks to generate

    Returns:
    --------
    rnws : NumPy 3darray
        Random networks based on input graph nw. Format is
                rnws.shape = (N,N,M)
        s.t.
                rnws[:,:,m] = m-th N x N random network

    Notes:
    ------
    This function requires bctpy to be installed!

    See also:
    ---------
    The docstring of randmio_und_connected in bct.py
    generate_randnws.m for a MATLAB version of this code
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
    try:
        if (round(M) != M) or (M < 1):
            raise ValueError("M has to be a natural number > 1!")
    except: raise TypeError("M has to be a natural number > 1!")

    # Try to import progressbar module
    try: 
        import progressbar as pb
        showbar = True
    except: 
        print "WARNING: progressbar module not found - consider installing it using pip install progressbar"
        showbar = False

    # Allocate space for random networks and convert input network to list
    rnws = np.zeros((N,N,M))

    # If available, initialize progressbar
    if (showbar): 
        widgets = ['Calculating Random Networks: ',pb.Percentage(),' ',pb.Bar(marker='#'),' ',pb.ETA()]
        pbar = pb.ProgressBar(widgets=widgets,maxval=M)

    # Populate tensor
    counter = 0
    if (showbar): pbar.start()
    for m in xrange(M):
        rnws[:,:,m],eff = bct.randmio_und_connected(nw,5)
        counter += eff
        if (showbar): pbar.update(m)
    if (showbar): pbar.finish()

    return rnws

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
    if (min(shc)==1) or (shc[0]!=shc[1]):
        raise ValueError('Input must be a N-by-N-by-k NumPy array!')
    if np.isnan(corrs).max()==True or np.isinf(corrs).max()==True or np.isreal(corrs).min()==False:
        raise ValueError('Input must be a real valued NumPy array without Infs or NaNs!')
