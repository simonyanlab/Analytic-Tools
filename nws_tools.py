# nws_tools.py - Collection of network creation/processing/analysis/plotting routines
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# Created: December 22 2014
# Last modified: <2017-02-02 12:06:21>

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import natsort
import os
import csv
import inspect
import fnmatch
from scipy import weave
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import Normalize, colorConverter, LightSource
import math

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
            and can be installed using `pip`. 
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
            and can be installed using `pip`. 
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
            and can be installed using `pip`. 
    """

    N = CIJ.shape[0]                    # no. of nodes
    K = (np.triu(CIJ,1)!=0).sum()       # no. of edges
    return K/((N**2 - N)/2.0)

##########################################################################################
def get_corr(txtpath,corrtype='pearson',sublist=[],**kwargs):
    """
    Compute pair-wise statistical dependence of time-series

    Parameters
    ----------
    txtpath : string
        Path to directory holding ROI-averaged time-series dumped in `txt` files.
        The following file-naming convention is required

                `sNxy_bla_bla.txt`,

        where `N` is the group id (1,2,3,...), `xy` denotes the subject number 
        (01,02,...,99 or 001,002,...,999) and anything else is separated 
        by underscores. The files will be read in lexicographic order,
        i.e., `s101_1.txt`, `s101_2.txt`,... or `s101_Amygdala.txt`, `s101_Beemygdala`,...
        See Notes for more details. 
    corrtype : string
        Specifier indicating which type of statistical dependence to use to compute 
        pairwise dependence. Currently supported options are 

                `pearson`: the classical zero-lag Pearson correlation coefficient 
                (see NumPy's `corrcoef` for details)

                `mi`: (normalized) mutual information 
                (see the docstring of `mutual_info` in this module for details)
    sublist : list or NumPy 1darray
        List of subject codes to process, e.g., `sublist = ['s101','s102']`. 
        By default all subjects found in `txtpath` will be processed.
    **kwargs : keyword arguments
        Additional keyword arguments to be passed on to the function computing 
        the pairwise dependence (currently either NumPy's `corrcoef` or `mutual_info`
        in this module). 
       
    Returns
    -------
    res : dict
        Dictionary with fields:

        corrs : NumPy 3darray
            `N`-by-`N` matrices of pair-wise regional statistical dependencies 
	    of `numsubs` subjects. Format is 
                    `corrs.shape = (N,N,numsubs)`,
            such that
                    `corrs[:,:,i]` = `N x N` statistical dependence matrix of `i`-th subject 
        bigmat : NumPy 3darray
            Tensor holding unprocessed time series of all subjects. Format is 
                    `bigmat.shape = (tlen,N,numsubs)`,
            where `tlen` is the maximum time-series-length across all subjects 
            (if time-series of different lengths were used in the computation, 
            any unfilled entries in `bigmat` will be NumPy `nan`'s, see Notes 
            for details) and `N` is the number of regions (=nodes in the networks). 
        sublist : list of strings
            List of processed subjects specified by `txtpath`, e.g.,
                    `sublist = ['s101','s103','s110','s111','s112',...]`

    Notes
    -----
    Per-subject time-series do not necessarily have to be of the same length across 
    a subject cohort. However, all ROI-time-courses *within* the same subject must have 
    the same number of entries. 
    For instance, all ROI-time-courses in `s101` can have 140 entries, and time-series 
    of `s102` might have 130 entries. The remaining 10 values "missing" for `s102` are 
    filled with `NaN`'s in `bigmat`. However, if `s101_2.txt` contains 140 data-points while only 
    130 entries are found in `s101_3.txt`, the code will raise a `ValueError`. 

    See also
    --------
    numpy.corrcoef : Pearson product-moment correlation coefficents
    mutual_info : Compute (normalized) mutual information coefficients
    """

    # Make sure `txtpath` doesn't contain nonsense and points to an existing location
    if not isinstance(txtpath,(str,unicode)):
        raise TypeError('Input has to be a string specifying the path to the txt-file directory!')
    txtpath = str(txtpath)
    if txtpath.find("~") == 0:
        txtpath = os.path.expanduser('~') + txtpath[1:]
    if not os.path.isdir(txtpath):
        raise ValueError('Invalid directory: '+txtpath+'!')

    # Check `corrtype`
    if not isinstance(corrtype,(str,unicode)):
        raise TypeError('Correlation type input must be a string, not '+type(corrtype).__name__+'!')
    if corrtype != 'mi' and corrtype != 'pearson':
        raise ValueError("Currently, only Pearson and (N)MI supported!")

    # Check `sublist`
    if not isinstance(sublist,(list,np.ndarray)):
        raise TypeError('Subject codes have to be provided as Python list/NumPy 1darray, not '+type(sublist).__name__+'!')
    if len(np.array(sublist).shape) != 1:
        raise ValueError("Subject codes have to be provided as 1-d list/array!")

    # Get length of `sublist` (to see if a subject list was provided)
    numsubs = len(sublist)

    # Get list of all txt-files in `txtpath` and order them lexicographically
    if txtpath[-1] == ' '  or txtpath[-1] == os.sep: txtpath = txtpath[:-1]
    txtfiles = natsort.natsorted(myglob(txtpath,"s*.[Tt][Xx][Tt]"), key=lambda y: y.lower())
    if len(txtfiles) < 2: raise ValueError('Found fewer than 2 text files in '+txtpath+'!')

    # If no subject-list was provided, take first subject to get the number of ROIs to be processed
    if numsubs == 0:

        # Search from left in file-name for first "s" (naming scheme: sNxy_bla_bla_.txt)
        firstsub  = txtfiles[0]
        firstsub  = firstsub.replace(txtpath+os.sep,'')
        s_in_name = firstsub.find('s')
    
        # The characters right of "s" until the first "_" are the subject identifier
        udrline = firstsub[s_in_name::].find('_')
        subject = firstsub[s_in_name:s_in_name+udrline]

        # Generate list of subjects
        sublist = [subject]
        for fl in txtfiles:
            if fl.count(subject) == 0:
                s_in_name = fl.rfind('s')
                udrline   = fl[s_in_name::].find('_')
                subject   = fl[s_in_name:s_in_name+udrline]
                sublist.append(subject)

        # Update `numsubs`
        numsubs = len(sublist)

        # Prepare output message
        msg = "Found "

    else:

        # Just take the first entry of user-provided subject list
        subject = sublist[0]

        # Prepare output message
        msg = "Processing "

    # Talk to the user
    substr = str(sublist)
    substr = substr.replace('[','')
    substr = substr.replace(']','')
    print msg+str(numsubs)+" subjects: "+substr

    # Get number of regions
    numregs = ''.join(txtfiles).count(subject)
    
    # Get (actual) number of subjects
    numsubs = len(sublist)

    # Scan files to find time-series length
    tlens = np.zeros((numsubs,),dtype=int)
    for k in xrange(numsubs):
        roi = 0
        for fl in txtfiles:
            if fl.count(sublist[k]):
                try:
                    ts_vec = np.loadtxt(fl)
                except:
                    raise ValueError("Cannot read file "+fl)
                if roi == 0:
                    tlens[k] = ts_vec.size     # Subject's first TS sets our reference length
                if ts_vec.size != tlens[k]:
                    raise ValueError("Error reading file: "+fl+\
                                     " Expected a time-series of length "+str(tlens[k])+", "+
                                     "but actual length is "+str(ts_vec.size))
                roi += 1

        # Safeguard: stop if subject is missing, i.e., `roi == 0` still (weirder things have happened...)
        if roi == 0: 
            raise ValueError("Subject "+sublist[k]+" is missing!")

        # Safeguard: stop if subject hast more/fewer ROIs than expected
        elif roi != numregs:
            raise ValueError("Found "+str(int(roi))+" time-series for subject "+sublist[k]+", expected "+str(int(numregs)))

    # Check the lengths of the detected time-series
    if tlens.min() <= 2: 
        raise ValueError('Time-series of Subject '+sublist[tlens.argmin()]+' is empty or has fewer than 2 entries!')

    # Allocate tensor to hold all time series
    bigmat = np.zeros((tlens.max(),numregs,numsubs)) + np.nan

    # Allocate tensor holding correlation matrices of all subjects 
    corrs = np.zeros((numregs,numregs,numsubs))

    # Ready to do this...
    print "Extracting data and calculating "+corrtype.upper()+" coefficients"

    # Cycle through subjects and save per-subject time series data column-wise
    for k in xrange(numsubs):
        col = 0
        for fl in txtfiles:
            if fl.count(sublist[k]):
                ts_vec = np.loadtxt(fl)
                bigmat[:tlens[k],col,k] = ts_vec
                col += 1

        # Compute correlations based on corrtype
        if corrtype == 'pearson':
            corrs[:,:,k] = np.corrcoef(bigmat[:tlens[k],:,k],rowvar=0,**kwargs)
        elif corrtype == 'mi':
            corrs[:,:,k] = mutual_info(bigmat[:tlens[k],:,k],**kwargs)

    # Happy breakdown
    print "Done"
    return {'corrs':corrs, 'bigmat':bigmat, 'sublist':sublist}

##########################################################################################
def corrcheck(*args,**kwargs):
    """
    Sanity checks for statistical dependence matrices (Pearson or NMI)
    
    Parameters
    ----------
    Dynamic : Usage as follows
    corrcheck(A) : input is NumPy 2darray                    
        shows some statistics for the correlation matrix `A`
    corrcheck(A,label) : input is NumPy 2darray and `['string']`
        shows some statistics for the matrix `A` and uses
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

        where `A`, `B`, `C`,... are matrices. 
    corrcheck(T,label) : input is NumPy 3darray and list of strings
        shows some statistics for matrices stored 
        in the tensor `T`. The storage scheme has to be
                `T[:,:,0] = A`

                `T[:,:,1] = B`

                `T[:,:,2] = C`

                etc.

        where `A`, `B`, `C`,... are matrices. The list of strings `label`
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

    # Assign global name for all figures if provided by additional keyword argument `title`
    figtitle = kwargs.get('title',None); nofigname = False
    if figtitle is None: 
        nofigname = True
    else:
        if not isinstance(figtitle,(str,unicode)):
            raise ValueError('Figure title must be a string!')

    # If labels have been provided, extract them now
    if isinstance(args[-1],(list)):
        myin  -= 1
        labels = args[-1]
        usrlbl = 1
    elif isinstance(args[-1],(str,unicode)):
        myin  -= 1
        labels = [args[-1]]
        usrlbl = 1
    else:
        usrlbl = 0

    # Try to get shape of input
    try:
        szin = len(args[0].shape)
    except: raise TypeError("Expected NumPy array(s) as input, found "+type(args[0]).__name__+"!")

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
    if not plt.is_numlike(corrs) or not np.isreal(corrs).all():
        raise TypeError("Input arrays must be real-valued!")
    if np.isfinite(corrs).min() == False:
        raise ValueError("All matrices must be real without NaNs or Infs!")

    # Check if we're dealing with Pearson or NMI matrices (or something completely unexpected)
    cmin = corrs.min(); cmax = corrs.max()
    if cmax > 1 or cmin < -1:
        msg = "WARNING: Input has to have values between -1/+1 or 0/+1. Found "+str(cmin)+" to "+str(cmax)
        print msg
    maxval = 1
    if corrs.min() < 0:
        minval = -1
    else:
        minval = 0

    # If labels have been provided, check if we got enough of'em; if there are no labels, generate defaults
    if (usrlbl):
        if len(labels) != nmat: raise ValueError('Numbers of labels and matrices do not match up!')
        for lb in labels:
            if not isinstance(lb,(str,unicode)):
                raise ValueError('Labels must be provided as list of strings or a single string!')
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
            plt.imshow((corrs[:,:,i]>=0).astype(float),cmap='gray',interpolation='nearest',vmin=0,vmax=1)
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
def get_meannw(nws,percval=0.0):
    """
    Helper function to compute group-averaged networks

    Parameters
    ----------
    nws : NumPy 3darray
        `N`-by-`N` connection matrices of `numsubs` subjects. Format is 
                `nws.shape = (N,N,numsubs)`,
        such that
                `nws[:,:,i] = N x N` connection matrix of `i`-th subject 
    percval : float
        Percentage value, such that connections not present in at least `percval`
        percent of subjects are not considered, thus `0 <= percval <= 1`.
        Default setting is `percval = 0.0`
       
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
    the code increases `percval` in 5% steps to ensure connectedness of the group-averaged graph. 
    The concept of using only a certain percentage of edges present in subjects was taken from 
    M. van den Heuvel, O. Sporns: "Rich-Club Organization of the Human Connectome" (2011), J. Neurosci. 
    Currently available `here <http://www.jneurosci.org/content/31/44/15775.full>`_
    
    See also
    --------
    None
    """

    # Sanity checks
    arrcheck(nws,'tensor','nws')
    scalarcheck(percval,'percval',bounds=[0,1])
    
    # Get shape of input tensor
    N       = nws.shape[0]
    numsubs = nws.shape[-1]

    # Remove self-connections
    nws = rm_selfies(nws)

    # Allocate memory for binary/weighted group averaged networks
    mean_binary = np.zeros((N,N))
    mean_wghted = np.zeros((N,N))

    # Compute mean network and keep increasing `percval` until we get a connected mean network
    docalc = True
    while docalc:

        # Reset matrices 
        mean_binary[:] = 0
        mean_wghted[:] = 0

        # Cycle through subjects to compute average network
        for i in xrange(numsubs):
            mean_binary = mean_binary + (nws[:,:,i]!=0).astype(float)
            mean_wghted = mean_wghted + nws[:,:,i]

        # Kick out connections not present in at least `percval%` of subjects (in binary and weighted NWs)
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
    Remove negative entries from correlation matrices

    Parameters
    ----------
    corrs : NumPy 3darray
        An array of `K` correlation matrices of dimension `N`-by-`N`. Format is 
                `corrs.shape = (N,N,K)`,
        such that
                `corrs[:,:,i]` is the `i`-th `N x N` correlation matrix

    Returns
    -------
    nws : NumPy 3darray
        Same format as input tensor but `corrs >= 0`. 

    Notes
    -----
    None

    See also
    --------
    None
    """

    # Sanity checks
    arrcheck(corrs,'tensor','corrs')

    # See how many matrices are stacked in the array
    K = corrs.shape[-1]

    # Zero diagonals of matrices
    for i in xrange(K):
        np.fill_diagonal(corrs[:,:,i],0)

    # Remove negative correlations
    nws = (corrs > 0)*corrs

    # Check if we lost some nodes...
    ndnum = str(corrs.shape[0])
    for i in xrange(K):
        deg = degrees_und(corrs[:,:,i])
        if deg.min() == 0:
            badidx = np.nonzero(deg==deg.min())[0]
            print "WARNING: In network "+str(i)+" a total of "+str(badidx.size)+" out of "+ndnum+\
                " node(s) got disconnected, namely vertices #"+str(badidx)

    return nws

##########################################################################################
def rm_selfies(conns):
    """
    Remove self-connections from connection matrices

    Parameters
    ----------
    conns : NumPy 3darray
        An array of `K` connection matrices of dimension `N`-by-`N`. Format is 
                `conns.shape = (N,N,K)`,
        such that
                `conns[:,:,i]` is the `i`-th `N x N` connection matrix

    Returns
    -------
    nws : NumPy 3darray
        Same format as input array but `np.diag(conns[:,:,k]).min() = 0.0`. 

    Notes
    -----
    None

    See also
    --------
    None
    """

    # Sanity checks
    arrcheck(conns,'tensor','conns')

    # Create output quantity and zero its diagonals
    nws = conns.copy()
    for i in xrange(nws.shape[-1]):
        np.fill_diagonal(nws[:,:,i],0)

    return nws

##########################################################################################
def thresh_nws(nws,userdens=None,percval=0.0,force_den=False,span_tree=False):
    """
    Threshold networks based on connection density

    Parameters
    ----------
    nws : NumPy 3darray
        Undirected `N`-by-`N` (un)weighted connection matrices of `numsubs` subjects. Format is 
                `corrs.shape = (N,N,numsubs)`,
        such that
                `corrs[:,:,i] = N x N` connection matrix of `i`-th subject 
    userdens : int
        By default, the input networks are thresholded down to the lowest common 
        connection density without disconnecting any nodes in the networks using 
        a relative thresholding strategy (`force_den = False` and `span_tree = False`). 
        If `userdens` is provided and `span_tree = False`, then `userdens`
        is used as target density in the relative thresholding strategy. However, 
        if `userdens` is below the minimum density before networks fragment, 
        it will not be used unless `force_den = True`. 
        If `span_tree = True` and `userdens` is `None`, then maximum spanning 
        trees will be returned for all input networks. If `userdens` is provided, 
        the spanning trees will be populated with the strongest connections 
        found in the original networks up to the desired edge density. 
        For both relative thresholding and maximum spanning tree density reduction, 
        `userdens` should be either `None` or an integer between 0 and 100. 
        See Notes below for more details. 
    percval : float
        Percentage value for computing mean network averaged across all thresholded 
        graphs, such that connections not present in at least `percval`
        percent of subjects are not considered (`0 <= percval <= 1`).
        Default setting is `percval = 0.0`. See `get_meannw` for details. 
    force_den : bool
        If `force_den = True` relative thresholding is applied to the networks 
        until all graphs hit the desired density level defined by the user 
        even if nodes get disconnected in the process. This argument has no 
        effect if `span_tree = True`. By default, `force_den = False`. 
    span_tree : bool
        If `span_tree` is `True` density reduction is performed by constructing maximum 
        spanning trees. If `userdens` is `None`, only spanning trees for all input networks
        will be returned. If `userdens` is provided, spanning trees will be populated 
        with the strongest connections found in the original networks up to the 
        desired edge density. Note that `foce_den` is ignored if `span_tree` is `True`. 
               
    Returns
    -------
    Dictionary holding computed quantities. The fields of the dictionary depend upon 
    the values of the optional keyword arguments `userdens` and `span_tree`. 
    res : dict 
        Dictionary with fields

        th_nws : NumPy 3darray
            Sparse networks. Format is the same as for `nws` 
            (Not returned if `userdens` is `None` and `span_tree = True`). 
        den_values : NumPy 1darray
            Density values of the networks stored in `th_nws`, such that `den_values[i]`
            is the edge density of the graph `th_nws[:,:,i]`
            (not returned if `userdens` is `None` and `span_tree = True`). 
        th_mnw : NumPy 2darray
            Mean network averaged across all sparse networks `th_nws` 
            (not returned if `userdens` is `None` and `span_tree = True`). 
        mnw_percval: float
            Percentage value used to compute `th_mnw` (see documentation of `get_meannw` for
            details, not returned if `userdens` is `None` and `span_tree = True`). 
        tau_levels : NumPy 1darray
            Cutoff values used in the relative thresholding strategy to compute 
            `th_nws`, i.e., `tau_levels[i]` is the threshold that generated 
            network `th_nws[:,:,i]` (only returned if `span_tree = False`). 
        nws_forest : NumPy 3darray
            Maximum spanning trees calculated for all input networks 
            (only returned if `span_tree = True`). 
        mean_tree : NumPy 2darray
            Mean spanning tree averaged across all spanning trees stored in 
            `nws_forest` (only returned if `span_tree = True`). 
        mtree_percval : float
            Percentage value used to compute `mean_tree` (see documentation of `get_meannw` for
            details, only returned if `span_tree = True`). 

    Notes
    -----
    This routine uses either a relative thresholding strategy or a maximum spanning tree 
    approach to decrease the density of a given set of input networks. 

    During relative thresholding (`span_tree = False`) edges are discarded based on their value relative to the 
    maximum edge weight found across all networks beginning with the weakest links. By default, 
    the thresholding algorithm uses the lowest common connection density across all input networks 
    before a node is disconnected as target edge density. That means, if networks `A`, `B` and `C` 
    can be thresholded down to 40%, 50% and 60% density, respectively, without disconnecting any 
    nodes, then the lowest common density for thresholding `A`, `B` and `C` together is 60%. 
    In this case the raw network `A` already has a density of 60% or lower, which is thus excluded 
    from thresholding and the original network is copied into `th_nws`. If a density level 
    is provided by the user, then the code tries to use it unless it violates connectedness 
    of all thresholded networks - in this case the lowest common density of all networks is used, 
    unless `force_den = True` which causes the code to employ the user-provided density level 
    for thresholding, disconnecting nodes from the networks in the process. 

    The maximum spanning tree approach (`span_tree = True`) can be interpreted as the inverse of relative 
    thresholding. Instead of chipping away weak edges in the input networks until a target density 
    is met (or nodes disconnect), a minimal backbone of the network is calculated and then 
    populated with the strongest connections found in the original network until a desired 
    edge density level is reached. The backbone of the network is calculated by computing the graph's maximum
    spanning tree, that connects all nodes with the minimum number of maximum-weight edges. 
    Note, that unless each edge has a distinct unique weight value a graph has numerous different 
    maximum spanning trees. Thus, the spanning trees computed by this routine are usually *not* unique, 
    and consequently the thresholded networks may not be unique either (particularly for low 
    density levels, for which the computed populated networks are very similar to the underlying spanning trees). 
    Thus, in contrast to the more common relative thresholding strategy, this bottom-up approach 
    allows to reduce a given network's density to an almost arbitrary level 
    (>= density of the maximum spanning tree) without disconnecting nodes. However, unlike relative 
    thresholding, the computed sparse networks are not necessarily unique and strongly depend 
    on the intial maximum spanning tree. Note that if `userdens` is `None`, only maximum spanning 
    trees will be computed. 

    The code below relies on the routine `get_meannw` in this module to compute the group-averaged
    network. Futher, maximum spanning trees are calculated using `backbone_wu.m` from the 
    Brain Connectivity Toolbox (BCT) for MATLAB via Octave. Thus, it requires Octave to be installed 
    with the BCT in its search path. Further, `oct2py` is needed to launch an Octave instance 
    from within Python. 

    See also
    --------
    get_meannw : Helper function to compute group-averaged networks
    backbone_wu : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                  `here <https://sites.google.com/site/bctnet/>`_
    """

    # Sanity checks
    arrcheck(nws,'tensor','nws')
    if userdens is not None:
        scalarcheck(userdens,'userdens',kind='int',bounds=[0,100])
    scalarcheck(percval,'percval',bounds=[0,1])
    if not isinstance(force_den,bool):
        raise TypeError("The optional argument `force_den` has to be Boolean!")
    if not isinstance(span_tree,bool):
        raise TypeError("The optional argument `span_tree` has to be Boolean!")
    if force_den and span_tree:
        print "\nWARNING: The flag `foce_den` has no effect if `span_tree == True`!"

    # Try to import `octave` from `oct2py`
    if span_tree:
        try: 
            from oct2py import octave
        except: 
            errmsg = "Could not import octave from oct2py! "+\
                     "To compute the maximum spanning tree octave must be installed and in the search path. "+\
                     "Furthermore, the Brain Connectivity Toolbox (BCT) for MATLAB must be installed "+\
                     "in the octave search path. "
            raise ImportError(errmsg)
        
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
        th_mnw,mnw_percval = get_meannw(nws,percval)
        res_dict = {'th_nws':nws, 'den_values': raw_den, \
                    'th_mnw': th_mnw, 'mnw_percval': mnw_percval}

        # The structure of `backbone_wu.m` requires *exact* symmetry...
        if span_tree:
            nws_forest = np.zeros(nws.shape)
            for i in xrange(numsubs):
                mnw = nws[:,:,i].squeeze()
                mnw = np.triu(mnw,1)                              
                nws_forest[:,:,i] = octave.backbone_wu(mnw + mnw.T,2)
            mean_tree,mtree_percval = get_meannw(nws_forest,percval)
            res_dict['nws_forest'] = nws_forest
            res_dict['mean_tree'] = mean_tree
            res_dict['mtree_percval'] = mtree_percval
        else:
            res_dict['tau_levels'] = None
        return res_dict 

    # Inform user about minimal/maximal density in raw data
    print "\nRaw data has following density values: \n"
    print "\tMinimal density: "+str(min_raw)+"%"
    print "\tMaximal density: "+str(max_raw)+"%"

    # Allocate space for output (needed for both regular thresholding and de-foresting)
    th_nws     = np.zeros(nws.shape)
    den_values = np.zeros((numsubs,))
    th_mnw     = np.zeros((N,N))

    # Maximum spanning tree shenanigans
    if span_tree:

        # Allocate space for the spanning trees
        nws_forest = np.zeros(nws.shape)

        # If no target density was provided, just compute trees and get out of here
        if userdens is None:
            print "\nCalculating maximum spanning trees..."
            for i in xrange(numsubs):
                mnw = nws[:,:,i].squeeze()
                mnw = np.triu(mnw,1)                              
                nws_forest[:,:,i] = octave.backbone_wu(mnw + mnw.T,2)
            mean_tree,mtree_percval = get_meannw(nws_forest,percval)
            return {'nws_forest': nws_forest, 'mean_tree': mean_tree, 'mtree_percval': mtree_percval}
        else:

            # The edge density `d` of an undirected network is given by
            #           (1) `d = 2*K/(N**2 - N)`, 
            # where `K` denotes the number of edges in the network. Thus, `K` can be approximated by
            #           (2) `N*avdg/2`,
            # with `avdg` denoting the average nodal degree in the graph (divide by two
            # to not count links twice (we have undirected links i <-> j, not i -> j and j <- i).
            # Thus, substituting (2) for `K` in (1) and re-arranging terms yields
            # `avdg = d*(N**2 - N)/N`. Thus, for a user-provided density value, we can compute
            # the associated average degree of the wanted target network as
            avdg = np.round(userdens/100*(N**2 - N)/N)
            print "\nReducing network densities to "+str(userdens)+"% by inversely populating maximum spanning trees..."

            # Use this average degree value to cut down input networks to desired density
            for i in xrange(numsubs):
                mnw      = nws[:,:,i].squeeze()
                mnw      = np.triu(mnw,1)                              
                raw_dper = int(np.round(1e2*raw_den[i]))
                if raw_dper <= userdens:
                    print "Density of raw network #"+str(i)+" is "+str(raw_dper)+"%"+\
                        " which is already lower than thresholding density of "+str(userdens)+"%"
                    print "Returning original unthresholded network"
                    th_nws[:,:,i] = nws[:,:,i].copy()
                    den_values[i] = raw_den[i]
                    nws_forest[:,:,i] = octave.backbone_wu(mnw + mnw.T.squeeze(),2)
                else:
                    nws_forest[:,:,i],th_nws[:,:,i] = octave.backbone_wu(mnw + mnw.T,avdg)
                    den_values[i] = density_und(th_nws[:,:,i])
            mean_tree,mtree_percval = get_meannw(nws_forest,percval)

            # Populate results dictionary with method-specific quantities
            res_dict = {'nws_forest': nws_forest, 'mean_tree': mean_tree, 'mtree_percval': mtree_percval}

    # Here the good ol' relative weight thresholding
    else:

        # Allocate space for thresholds
        tau_levels = np.zeros((numsubs,))

        # Create vector of thresholds to iterate over
        dt      = 1e-3
        threshs = np.arange(0,1+2*dt,dt)

        # Cycle through subjects and threshold the connection matrices until a node disconnects
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
                # Thin out connection matrix step by step (that's why we only have to load `nws[:,:,i]` once
                tau = th*maxw
                mnw = mnw*(mnw >= tau).astype(float)

                # Compute density of thinned graph (weight info is discarded)
                den = density_und(mnw)

                # Compute nodal degrees of network 
                deg = degrees_und(mnw)

                # As soon as one node gets disconnected (i.e. `deg[i] == 0`) stop thresholding and save previous dens
                if deg.min() == 0:
                    th_nws[:,:,i] = mnw_old
                    tau_levels[i] = tau_old
                    den_values[i] = den_old
                    break

        # Compute minimal density before fragmentation across all subjects
        densities = np.round(1e2*den_values)
        print "\nMinimal admissible densities of per-subject networks are as follows: "
        for i in xrange(densities.size): print "Subject #"+str(i+1)+": "+str(int(densities[i]))+"%"
        min_den = int(np.round(1e2*den_values.max()))
        print "\nThus, minimal density before fragmentation across all subjects is "+str(min_den)+"%"

        # Assign thresholding density level
        if userdens is None:
            thresh_dens = min_den
        else:
            if userdens < min_den and force_den == False:
                print "\nUser provided density of "+str(int(userdens))+\
                    "% lower than minimal admissible density of "+str(min_den)+"%. "
                print "Using minimal admissible density instead. "
                thresh_dens = min_den
            elif userdens < min_den and force_den == True:
                print "\nWARNING: Provided density of "+str(int(userdens))+\
                    "% leads to disconnected networks - proceed with caution..."
                thresh_dens = int(userdens)
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
                tau_levels[i] = 0
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

        # Populate results dictionary with method-specific quantities
        res_dict = {'tau_levels': tau_levels}

    # Compute group average network
    th_mnw,mnw_percval = get_meannw(th_nws,percval)

    # Fill up results dictionary
    res_dict['th_nws'] = th_nws
    res_dict['den_values'] = den_values
    res_dict['th_mnw'] = th_mnw
    res_dict['mnw_percval'] = mnw_percval
    
    # Be polite and dismiss the user 
    print "\nDone...\n"
    return res_dict

##########################################################################################
def normalize(arr,vmin=0,vmax=1):
    """
    Re-scales a NumPy ndarray

    Parameters
    ----------
    arr : NumPy ndarray
        An array of size > 1 (shape can be arbitrary)
    vmin : float
        Floating point number representing the lower normalization bound. 
        (Note that it has to hold that `vmin < vmax`)
    vmin : float
        Floating point number representing the upper normalization bound. 
        (Note that it has to hold that `vmin < vmax`)
       
    Returns
    -------
    arrn : NumPy ndarray
        Scaled version of the input array `arr`, such that `arrn.min() == vmin` and 
        `arrn.max() == vmax`

    Notes
    -----
    In contrast to Matplotlib's `Normalize`, *all* values of the input array are re-scaled, 
    even if outside the specified bounds. For instance, if `arr.min() == -1` and `arr.max() == 0.5` then
    calling normalize with bounds `vmin = 0` and `vmax = 1` will result in an array `arrn`
    satisfying `arrn.min() == 0` and `arrn.max() == 1`. 

    Examples
    --------
    >>> arr = array([[-1,.2],[100,0]])
    >>> arrn = normalize(arr,vmin=-10,vmax=12)
    >>> arrn 
    array([[-10.        ,  -9.73861386],
           [ 12.        , -10.        ]])

    See also
    --------
    None 
    """

    # Ensure that `arr` is a NumPy-ndarray
    try:
        tmp = arr.size == 1
    except TypeError: 
        raise TypeError('Input `arr` has to be a NumPy ndarray!')
    if (tmp): 
        raise ValueError('Input `arr` has to be a NumPy ndarray of size > 1!')
    if not plt.is_numlike(arr) or not np.isreal(arr).all():
        raise TypeError("Input array hast to be real-valued!")
    if np.isfinite(arr).min() == False: 
        raise ValueError("Input `arr` must be real-valued without Inf's or NaN's!")

    # If normalization bounds are user specified, check them
    scalarcheck(vmin,'vmin')
    scalarcheck(vmax,'vmax')
    if vmax <= vmin: 
        raise ValueError('Lower bound `vmin` has to be strictly smaller than upper bound `vmax`!')
    if np.absolute(vmin - vmax) < np.finfo(float).eps:
            raise ValueError('Bounds too close: `|vmin - vmax| < eps`, no normalization possible')

    # Get min and max of array
    arrmin = arr.min()
    arrmax = arr.max()

    # If min and max values of array are identical do nothing, if they differ close to machine precision abort
    if arrmin == arrmax:
        return arr
    elif np.absolute(arrmin - arrmax) < np.finfo(float).eps:
        raise ValueError('Minimal and maximal values of array too close, no normalization possible')

    # Return normalized array
    return (arr - arrmin)*(vmax - vmin)/(arrmax - arrmin) + vmin

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

    # Make sure `csvfile` makes sense
    if not isinstance(csvfile,(str,unicode)):
        raise TypeError("Name of csv-file has to be a string!")
    if csvfile.find("~") == 0:
        csvfile = os.path.expanduser('~') + csvfile[1:]
    if not os.path.isfile(csvfile):
        raise ValueError('File: '+csvfile+' does not exist!')
    
    # Open `csvfile`
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
def shownet(A,coords,colorvec=None,sizevec=None,labels=None,threshs=[.8,.3,0],lwdths=[5,2,.1],nodecmap='jet',edgecmap='jet',textscale=3):
    """
    Plots a 3d network using Mayavi

    Parameters
    ----------
    A : NumPy 2darray
        Square `N`-by-`N` connection matrix of the network
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
    labels : Python list/NumPy 1darray
        Nodal labels. Format is `['Name1','Name2','Name3',...]` where the ordering HAS to be the same
        as in the `coords` dictionary. Note that the list/array has to have length `N`. 
    threshs : Python list/NumPy 1darray
        Thresholds for visualization. Edges with weights larger than `threshs[0]` are drawn 
        thickest, weights `> threshs[1]` are thinner and so on. Note that if `threshs[-1]>0` not all 
        edges of the network are plotted (since edges with `0 < weight < threshs[-1]` will be ignored). 
    lwdths : Python list/NumPy 1darray
        Line-widths associated to the thresholds provided by `threshs`. Edges with weights larger than 
        `threshs[0]` are drawn with line-width `lwdths[0]`, edges with `weights > threshs[1]` 
        have line-width `lwdths[1]` and so on. Thus `len(lwdths) == len(threshs)`. 
    nodecmap : string 
        Mayavi colormap to be used for plotting nodes. See Notes for details. 
    edgecmap : string 
        Mayavi colormap to be used for plotting edges. See Notes for details. 
    textscale : real number
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

    # Make sure the adjacency/weighting matrix makes sense
    arrcheck(A,'matrix','A')
    (N,M) = A.shape

    # Check the coordinate dictionary
    try:
        bad = (coords.keys() != N)
    except: 
        raise TypeError("The coordinates have to be provided as dictionary!")
    if bad: 
        raise ValueError('The coordinate dictionary has to have N keys!')
    for val in coords.values():
        if not isinstance(val,(list,np.ndarray)):
            raise TypeError('All elements of the `coords` dictionary have to be lists/arrays!')
        arrcheck(np.array(val),'vector','coordinates')
        if len(val) != 3:
            raise ValueError('All elements of the coords dictionary have to be 3-dimensional!')

    # Check `colorvec` if provided, otherwise assign default value
    if colorvec is not None:
        arrcheck(colorvec,'vector','colorvec',bounds=[0,1])
        if colorvec.size != N:
            raise ValueError('`colorvec` has to have length `N`!')
    else: 
        colorvec = np.ones((N,))

    # Same for `sizevec`
    if sizevec is not None:
        arrcheck(sizevec,'vector','sizevec',bounds=[0,np.inf])
        if sizevec.size != N:
            raise ValueError('`sizevec` has to have length `N`!')
    else:
        sizevec = np.ones((N,))

    # Check labels (if any provided)
    if labels is not None:
        try:
            bad = (len(labels) != N)
        except: 
            raise TypeError("Nodal labels have to be provided as list/NumPy 1darray!")
        if bad: 
            raise ValueError("Number of nodes and labels does not match up!")
        for lb in labels:
            if not isinstance(lb,(str,unicode)):
                raise ValueError('Each individual label has to be a string type!')
    else:
        labels = []

    # Check thresholds and linewidhts
    if not isinstance(threshs,(list,np.ndarray)):
        raise TypeError("Visualization thresholds have to be provided as list/NumPy 1darray!")
    threshs = np.array(threshs)
    arrcheck(threshs,'vector','threshs')
    n = threshs.size
    if not isinstance(lwdths,(list,np.ndarray)):
        raise TypeError("Linewidths have to be provided as list/NumPy 1darray!")
    lwdths = np.array(lwdths)
    arrcheck(lwdths,'vector','lwdths')
    m = lwdths.size
    if m != n: 
        raise ValueError("Number of thresholds and linewidths does not match up!")

    # Make sure colormap definitions were given as strings
    if not isinstance(nodecmap,(str,unicode)):
        raise TypeError("Colormap for nodes has to be provided as string!")
    if not isinstance(edgecmap,(str,unicode)):
        raise TypeError("Colormap for edges has to be provided as string!")

    # Check `textscale`
    scalarcheck(textscale,'textscale')

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
        mlab.text3d(coords[i][0]+2,coords[i][1],coords[i][2],labels[i],color=(0,0,0),scale=textscale)

    return

##########################################################################################
def show_nw(A,coords,colorvec=None,sizevec=None,labels=None,nodecmap=plt.get_cmap(name='jet'),edgecmap=plt.get_cmap(name='jet'),linewidths=None,nodes3d=False,viewtype='axial'):
    """
    Matplotlib-based plotting routine for 3d networks

    Parameters
    ----------
    A : NumPy 2darray
        Square `N`-by-`N` connection matrix of the network
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
    labels : Python list/NumPy 1darray
        Nodal labels. Format is `['Name1','Name2','Name3',...]` where the ordering HAS to be the same
        as in the `coords` dictionary. Note that the list/array has to have length `N`. 
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
        Camera position, `viewtype` can be one of the following
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

    # Check the graph's connection matrix
    arrcheck(A,'matrix','A')
    (N,M) = A.shape

    # Check the coordinate dictionary
    try:
        bad = (coords.keys() != N)
    except: 
        raise TypeError("The coordinates have to be provided as dictionary!")
    if bad: 
        raise ValueError('The coordinate dictionary has to have N keys!')
    for val in coords.values():
        if not isinstance(val,(list,np.ndarray)):
            raise TypeError('All elements of the coords dictionary have to be lists/arrays!')
        arrcheck(np.array(val),'vector','coordinates')
        if len(val) != 3:
            raise ValueError('All elements of the coords dictionary have to be 3-dimensional!')

    # Check `colorvec` if provided, otherwise assign default value
    if colorvec is not None:
        arrcheck(colorvec,'vector','colorvec',bounds=[0,1])
        if colorvec.size != N:
            raise ValueError('`colorvec` has to have length `N`!')
    else: 
        colorvec = np.ones((N,))

    # Same for `sizevec`
    if sizevec is not None:
        arrcheck(sizevec,'vector','sizevec',bounds=[0,np.inf])
        if sizevec.size != N:
            raise ValueError('`sizevec` has to have length `N`!')
    else:
        sizevec = np.ones((N,))

    # Check labels (if any provided)
    if labels is not None:
        try:
            bad = (len(labels) != N)
        except: 
            raise TypeError("Nodal labels have to be provided as list/NumPy 1darray!")
        if bad: 
            raise ValueError("Number of nodes and labels does not match up!")
        for lb in labels:
            if not isinstance(lb,(str,unicode)):
                raise ValueError('Each individual label has to be a string type!')
    else:
        labels = []

    # Check the colormaps
    if type(nodecmap).__name__ != 'LinearSegmentedColormap':
        raise TypeError('Nodal colormap has to be a Matplotlib colormap!')
    if type(edgecmap).__name__ != 'LinearSegmentedColormap':
        raise TypeError('Edge colormap has to be a Matplotlib colormap!')

    # If no linewidths were provided, use the entries of `A` as to control edge thickness
    if linewidths is not None:
        arrcheck(linewidths,'matrix','linewidths')
        (ln,lm) = linewidths.shape
        if linewidths.shape != A.shape:
            raise ValueError("Linewidths must be provided as square array of the same dimension as the connection matrix!")
    else:
        linewidths = A

    # Make sure `nodes3d` is Boolean
    if not isinstance(nodes3d,bool):
        raise TypeError('The nodes3d flag has to be a Boolean variable!')

    # Check if `viewtype` is anything strange
    if not isinstance(viewtype,(str,unicode)):
        raise TypeError("Viewtype must be 'axial(_{t/b})', 'sagittal(_{l/r})' or 'coronal(_{f/b})'")

    # Turn on 3d projection
    ax = plt.gcf().gca(projection='3d')
    ax.hold(True)

    # Extract nodal x-, y-, and z-coordinates from the coords-dictionary
    x = np.array([coords[i][0] for i in coords.keys()])
    y = np.array([coords[i][1] for i in coords.keys()])
    z = np.array([coords[i][2] for i in coords.keys()])

    # Order matters here: FIRST plot connections, THEN nodes on top of connections (looks weird otherwise)
    # Cycle through the matrix and plot every single connection line-by-line (this is *really* slow)
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

    # If `viewtype` was specified as 'axial', 'coronal' or 'sagittal' use default (top, front, left) viewtypes
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
        print "Using default viewtype `axial` instead"
        ax.view_init(elev=90,azim=-90)
    plt.axis('scaled')
    plt.axis('off')
    
    return 

##########################################################################################
def generate_randnws(nw,M,method="auto",rwr=5,rwr_max=10):
    """
    Generate random networks given a(n) (un)signed (un)weighted (un)directed input network

    Parameters
    ----------
    nw : NumPy 2darray
        Connection matrix of input network
    M : integer > 1
        Number of random networks to generate
    method : string
        String specifying which method to use to randomize 
        the input network. Currently supported options are 
        `'auto'` (default), `'null_model_und_sign'`, `'randmio_und'`, `'randmio_und_connected'`, 
        `'null_model_dir_sign'`, `'randmio_dir'`, `'randmio_dir_connected'`, 
        `'randmio_und_signed'`, `'randmio_dir_signed'`, 
        If `method = 'auto'` then a randomization strategy is chosen based 
        the the properties of the input network (directedness, edge-density, sign of 
        edge weights). In case of very dense networks (density > 75%) the `null_model`
        routines are used to at least shuffle the input network's edge weights. 
    rwr : integer
        Number of approximate rewirings per edge (default 5). 
    rwr_max : integer
        Maximal number of rewirings per edge to force randomization (default 10). 

    Returns
    -------
    rnws : NumPy 3darray
        Random networks based on input graph `nw`. Format is
                `rnws.shape = (N,N,M)`
        such that
                `rnws[:,:,m] = m-th N x N` random network

    Notes
    -----
    This routine calls functions from the Brain Connectivity Toolbox (BCT) for MATLAB via Octave. 
    Thus, it requires Octave to be installed with the BCT in its search path. Further, 
    `oct2py` is needed to launch an Octave instance from within Python. 

    See also
    --------
    randmio_und_connected : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                            `here <https://sites.google.com/site/bctnet/>`_
    randmio_dir_connected : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                            `here <https://sites.google.com/site/bctnet/>`_
    randmio_und : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                  `here <https://sites.google.com/site/bctnet/>`_
    randmio_dir : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                  `here <https://sites.google.com/site/bctnet/>`_
    randmio_und_signed : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                         `here <https://sites.google.com/site/bctnet/>`_
    randmio_dir_signed : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                         `here <https://sites.google.com/site/bctnet/>`_
    null_model_und_sign : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                          `here <https://sites.google.com/site/bctnet/>`_
    null_model_dir_sign : in the Brain Connectivity Toolbox (BCT) for MATLAB, currently available 
                          `here <https://sites.google.com/site/bctnet/>`_
    """

    # Try to import `octave` from `oct2py`
    try: 
        from oct2py import octave
    except: 
        errmsg = "Could not import octave from oct2py! "+\
                 "To use this routine octave must be installed and in the search path. "+\
                 "Furthermore, the Brain Connectivity Toolbox (BCT) for MATLAB must be installed "+\
                 "in the octave search path. "
        raise ImportError(errmsg)

    # Check the two mandatory inputs
    arrcheck(nw,'matrix','nw')
    N = nw.shape[0]
    scalarcheck(M,'M',kind='int',bounds=[1,np.inf])

    # See if the string `method` is one of the supported randomization algorithms
    supported = ["auto","randmio_und_connected","randmio_und","null_model_und_sign",\
                 "randmio_dir_connected","randmio_dir","null_model_dir_sign",\
                 "randmio_und_signed","randmio_dir_signed"]
    if supported.count(method) == 0:
        sp_str = str(supported)
        sp_str = sp_str.replace('[','')
        sp_str = sp_str.replace(']','')
        msg = 'Network cannot be randomized with '+str(method)+\
              '. Available options are: '+sp_str
        raise ValueError(msg)

    # See if `rwr` makes sense
    scalarcheck(rwr,'rwr',kind='int',bounds=[1,np.inf])

    # Try to import progressbar module
    try: 
        import progressbar as pb
        showbar = True
    except: 
        print "WARNING: progressbar module not found - consider installing it using `pip install progressbar`"
        showbar = False

    # Allocate space for random networks
    rnws = np.empty((N,N,M))
    rnw  = np.empty((N,N))
    rw   = rwr

    # Unless the user explicitly specified a randomization strategy, choose one based on the
    # input network's properties
    if method == "auto":
        min_nw = nw.min()
        sgds = ["unsigned","signed"][min_nw<0]
        if issym(nw):                                   # undirected graphs
            drct = "undirected"
            dns = density_und(nw)
            if dns > 0.75:
                randomizer = octave.null_model_und_sign
            else:
                if min_nw < 0:
                    randomizer = octave.randmio_und_signed
                else:
                    randomizer = octave.randmio_und
        else:                                           # directed graphs
            drct = "directed"
            dns = octave.density_dir(nw)
            if dns > 0.75:           
                randomizer = octave.null_model_dir_sign
            else:
                if min_nw < 0:
                    randomizer = octave.randmio_dir_signed
                else:
                    randomizer = octave.randmio_dir
        print "Input network is "+drct+" and "+sgds+" with an edge-density of "+str(np.round(1e2*dns))+"%. "+\
            "Using `"+randomizer.__name__+"` for randomization..."

    # Depending on whether the chosen randomizer returns effective re-wiring numbers, a slightly different
    # while loop structure is necessary
    use_nm = randomizer.__name__.find('null_model') >= 0

    # If available, initialize progressbar
    if (showbar): 
        widgets = ['Calculating Random Networks: ',pb.Percentage(),' ',pb.Bar(marker='#'),' ',pb.ETA()]
        pbar = pb.ProgressBar(widgets=widgets,maxval=M)

    # Populate tensor
    if (showbar): pbar.start()
    if use_nm:
        for m in xrange(M):
            rwr = rw
            ok = False
            while rwr <= rwr_max and ok == False:
                rnw = randomizer(nw,rwr,1)
                ok = not np.allclose(rnw,nw)
                rwr += 1
            if not ok:
                print "WARNING: network "+str(m)+" has not been randomized!"
            rnws[:,:,m] = rnw.copy()
            if (showbar): pbar.update(m)
    else:
        for m in xrange(M):
            rwr = rw
            eff = 0
            while rwr <= rwr_max and eff == 0:
                rnw,eff = randomizer(nw,rwr)
                rwr += 1
            if eff == 0:
                print "WARNING: network "+str(m)+" has not been randomized!"
            rnws[:,:,m] = rnw.copy()
            if (showbar): pbar.update(m)
    if (showbar): pbar.finish()

    return rnws

##########################################################################################
def hdfburp(f):
    """
    Pump out everything stored in a HDF5 container

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
    local name-space corresponding to the respective dataset-names in the file. 
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

            # If we found a variable, name it following the scheme: `groupname_varname`
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
def mutual_info(tsdata, n_bins=32, normalized=True, fast=True, norm_ts=True):
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
    norm_ts : bool
        If `True` the input time-series is normalized to zero mean and unit variance (default). 

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
    and Jobst Heitzig [1]_. It is currently available 
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

    References
    ----------
    .. [1] Copyright (C) 2008-2015, Jonathan F. Donges (Potsdam-Institute for Climate
           Impact Research), pyunicorn authors. All rights reserved.
           Redistribution and use in source and binary forms, with or without
           modification, are permitted provided that the following conditions are met:
               * Redistributions of source code must retain the above copyright notice, this
                 list of conditions and the following disclaimer.
               * Redistributions in binary form must reproduce the above copyright notice,
                 this list of conditions and the following disclaimer in the documentation
                 and/or other materials provided with the distribution.
               * Neither the name of pyunicorn authors and the Potsdam-Institute for
                 Climate Impact Research nor the names of its contributors may be used to
                 endorse or promote products derived from this software without specific
                 prior written permission.
           THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
           ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
           WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
           DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
           FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
           DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
           SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
           CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
           OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
           OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    # Sanity checks (`tsdata` is probably not square, that's why we don't use `arrcheck` here)
    try:
        shtsdata = tsdata.shape
    except:
        raise TypeError('Input must be a timepoint-by-index NumPy 2d array, not '+type(tsdata).__name__+'!')
    if len(shtsdata) != 2:
        raise ValueError('Input must be a timepoint-by-index NumPy 2d array')
    if (min(shtsdata)==1):
        raise ValueError('At least two time-series/two time-points are required to compute (N)MI!')
    if not plt.is_numlike(tsdata) or not np.isreal(tsdata).all():
        raise TypeError("Input must be real-valued!")
    if np.isfinite(tsdata).min() == False:
        raise ValueError('Input must be a real valued NumPy 2d array without Infs or NaNs!')

    scalarcheck(n_bins,'n_bins',kind='int',bounds=[2,np.inf])
    n_bins = int(n_bins)

    for bvar in [normalized,fast,norm_ts]:
        if not isinstance(bvar,bool):
            raise TypeError('The flags `normalized`, `fast` and `norm_ts` must be Boolean!')
    
    #  Get faster reference to length of time series = number of samples
    #  per grid point.
    (n_samples,N) = tsdata.shape

    #  Normalize `tsdata` time series to zero mean and unit variance
    if norm_ts:
        normalize_time_series(tsdata)

    #  Initialize mutual information array
    mi = np.zeros((N,N), dtype="float32")

    # Execute C++ code
    if (fast):

        # Create local transposed copy of `tsdata`
        tsdata = np.fastCopyAndTranspose(tsdata)
                
        # Get common range for all histograms
        range_min = float(tsdata.min())
        range_max = float(tsdata.max())
        
        # Re-scale all time series to the interval [0,1], 
        # using the maximum range of the whole dataset.
        denom   = range_max - range_min + 1 - (range_max != range_min)
        scaling = float(1. / denom)
        
        # Create array to hold symbolic trajectories
        symbolic = np.empty(tsdata.shape, dtype="int32")
        
        # Initialize array to hold 1d-histograms of individual time series
        hist = np.zeros((N,n_bins), dtype="int32")
        
        # Initialize array to hold 2d-histogram for one pair of time series
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

                //  The case `i = j` is not of interest here!
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

                    //  Reset `hist2d` to zero in all bins
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
                    //  `(i,j)`.
                    for (k = 0; k < n_samples; k++) {
                        symbol_i = symbolic(i,k);
                        symbol_j = symbolic(j,k);
                        hist2d(symbol_i,symbol_j) += 1;
                    }

                    //  Calculate mutual information for one pair of time 
                    //  series `(i,j)`.
                    // Hl = 0;
                    for (l = 0; l < n_bins; l++) {
                        hpl = hist(i,l) * norm;
                        if (hpl > 0.0) {
                            // `Hl += hpl * log(hpl);`
                            // `Hm = 0;`
                            for (m = 0; m < n_bins; m++) {
                                hpm = hist(j,m) * norm;
                                if (hpm > 0.0) {
                                    // `Hm += hpm * log(hpm);`
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

                    //  Reset `hist2d` to zero in all bins
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

        #  Define references to NumPy functions for faster function calls
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
        #  symmetric with respect to `X` and `Y`.
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
def issym(A,tol=1e-9):
    """
    Check for symmetry of a 2d NumPy array A

    Parameters
    ----------
    A : square NumPy 2darray
        A presumably symmetric matrix
    tol : positive real scalar
        Tolerance :math:`\\tau` for checking if :math:`A` is sufficiently close to :math:`A^\\top`.

    Returns
    -------
    is_sym : bool
        True if :math:`A` satisfies :math:`|A - A^\\top| \\leq \\tau |A|`,
        where :math:`|\\cdot|` denotes the Frobenius norm. Thus, if this inequality 
        holds, :math:`A` is approximately symmetric. 

    Notes
    -----
    For further details regarding the Frobenius norm approach used, please refer to the 
    discussion in `this <http://www.mathworks.com/matlabcentral/newsreader/view_thread/252727>`_ 
    thread at MATLAB central
    
    See also
    --------
    isclose : An absolute-value based comparison readily provided by NumPy. 
    """

    # Check if Frobenius norm of `A - A.T` is sufficiently small (respecting round-off errors)
    try:
        is_sym = (norm(A-A.T,ord='fro') <= tol*norm(A,ord='fro'))
    except:
        raise TypeError('Input argument has to be a square matrix/array and a scalar tol (optional)!')

    return is_sym

##########################################################################################
def myglob(flpath,spattern):
    """
    Return a glob-like list of paths matching a path-name pattern BUT support fancy shell syntax

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

    >>> filelist = myglob('Documents/MyHolidayFun','*.[Pp][Nn][Gg]')
    >>> print filelist
    >>> ['Documents/MyHolidayFun/img1.PNG','Documents/MyHolidayFun/img1.png']
        
    See also
    --------
    glob : Unix-style path-name and pattern expansion in Python
    """

    # Make sure provided path is a string and makes sense
    if not isinstance(flpath,(str,unicode)):
        raise TypeError('Filepath has to be a string!')
    flpath = str(flpath)
    if flpath.find("~") == 0:
        flpath = os.path.expanduser('~') + flpath[1:]
    slash = flpath.rfind(os.sep)
    if slash >= 0 and not os.path.isdir(flpath[:flpath.rfind(os.sep)]):
        raise ValueError('Invalid path: '+flpath+'!')
    if not isinstance(spattern,(str,unicode)):
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
        List/array of length `N` or `N+1` providing labels to be printed in the first row of the table
        (strings/numerals or both). See Examples for details
    leadcol : Python list or NumPy 1darray
        List/array of length `M` providing labels to be printed in the first column of the table
        (strings/numerals or both)
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

    # Check dimensions of input
    try:
        ds = data.shape
    except:
        raise TypeError('Input must be a M-by-N NumPy array, not ' + type(data).__name__+'!')
    if len(ds) > 2:
        raise ValueError('Input must be a M-by-N NumPy array!')
    for lvar in [leadcol,leadrow]:
        if not isinstance(lvar,(list,np.ndarray)):
            raise TypeError("The inputs `leadcol` and `leadrow` must by Python lists or Numpy 1d arrays!")
        if len(np.array(lvar).squeeze().shape) != 1:
            raise ValueError("The inputs `leadcol` and `leadrow` must 1-d lists/arrays!")
    m = len(leadcol)
    n = len(leadrow)

    # If a filename was provided make sure it's a string and check if the path exists
    if fname is not None:
        if not isinstance(fname,(str,unicode)):
            raise TypeError('Optional output filename has to be a string!')
        fname = str(fname)
        if fname.find("~") == 0:
            fname = os.path.expanduser('~') + fname[1:]
        slash = fname.rfind(os.sep)
        if slash >= 0 and not os.path.isdir(fname[:fname.rfind(os.sep)]):
            raise ValueError('Invalid path for output file: '+fname+'!')
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
def img2vid(imgpth,imgfmt,outfile,fps,filesize=None,ext='mp4',preset='veryslow'):
    """
    Convert a sequence of image files to a video using ffmpeg

    Parameters
    ----------
    imgpth : str
        Path to image files
    imgfmt : str
        Format specifier for images. All files in the image stack have to follow the same naming 
        convention, e.g., given the sequence `im_01.png`, `im_02.png`, ...,`im_99.png` the correct
        format specifier `imgfmt` is 'im_%02d.png'
    outfile : str
        Filename (including path if not in current directory) for output video. If an extension 
        is provided, e.g., 'animation.mp4' it is passed on to the x264 video encoder in 
        ffmpeg to set the video format of the output. Use `ffmpeg -formats` in a shell 
        to get a list of supported formats (any format labeled 'Muxing supported'). 
    fps : int
        Framerate of the video (number of frames per second)
    filesize : float
        Target size of video file in MB (Megabytes). 
        If provided, a encoding bitrate will be chosen such that the target size 
        `filesize` is not exceeded. If `filesize = None` the default constant rate factor of ffmpeg
        is used (the longer the movie, the larger the generated file). 
    ext : str
        Extension of the video-file. If `outfile` does not have a filetype extension, then 
        the default value of `ext` is used and an mp4 video is generated. Note: if `outfile`
        has an extension, then any value of `ext` will be ignored. Use `ffmpeg -formats` in a shell 
        to get a list of supported formats (any format labeled 'Muxing supported'). 
    preset : str
        Video quality options for ffmpeg's x264 encoder controlling the encoding speed to 
        compression ratio. A slower preset results in better compression (higher quality 
        per filesize) but longer encoding time. Available presets in ffmpeg  are 
        'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow', and 'placebo'. 

    Returns
    -------
    Nothing : None

    Examples
    --------
    Suppose the 600 sequentially numbered tiff files `image_001.tiff`, `image_002.tiff`, ..., `image_600.tiff`
    located in the directory `/path/to/images/ ` have to be converted to Quicktime movie (mov file) of no more than 25MB size. 
    We want the video to show 6 consecutive images per second (i.e., a framerate of 6 frames per second). 
    This can be done using the following command
    
    >>> img2vid('/path/to/images','image_%03d.tiff','image_1_600',6,filesize=25,ext='mov',preset='veryslow')

    Alternatively,

    >>> img2vid('/path/to/images','image_%03d.tiff','image_1_600_loq.mov',6,filesize=25,ext='mkv',preset='ultrafast')

    also generates an mov video of 25MB. The encoding will be faster but the image quality of `image_1_600_loq.mov` 
    will be lower compared to `image_1_600.mov` generated by the first call. Note that the optional keyword argument 
    `ext='mkv'` is ignored since the provided output filename 'image_1_600_loq.mov' already contains an extension. 
    """

    # First and foremost, check if ffmpeg is available, otherwise everything else is irrelevant
    if os.system("which ffmpeg > /dev/null") != 0:
        msg = "Could not find ffmpeg. It seems like ffmpeg is either not installed or not in the search path. "
        raise ValueError(msg)

    # Check if image directory exists and append trailing slash if necessary
    if not isinstance(imgpth,(str,unicode)):
        raise TypeError('Path to image directory has to be a string!')
    imgpth = str(imgpth)
    if imgpth.find("~") == 0:
        imgpth = os.path.expanduser('~') + imgpth[1:]
    if not os.path.isdir(imgpth):
        raise ValueError('Invalid path to image directory: '+imgpth+'!')
    slash = imgpth.rfind(os.sep)
    if slash != len(imgpth)-1: 
        imgpth += os.sep

    # Check if `imgfmt` is a valid string format specifier 
    # (don't use split below in case we have something like im.001.tiff)
    if not isinstance(imgfmt,(str,unicode)):
        raise TypeError('Format specifier for images has to be a string!')
    imgfmt = str(imgfmt)
    dot    = imgfmt.rfind('.')
    fmt    = imgfmt[:dot]
    imtype = imgfmt[dot+1:]
    if fmt.find('%') < 0: raise ValueError('Invalid image format specifier: `'+fmt+'`!')
    
    # Check if image directory actually contains any images of the given type
    imgs     = natsort.natsorted(myglob(imgpth,'*.'+imtype), key=lambda y: y.lower())
    num_imgs = len(imgs)
    if num_imgs < 2: raise ValueError('Directory '+imgpth+' contains fewer than 2 `'+imtype+'` files!')

    # Check validity of `outfile`
    if not isinstance(outfile,(str,unicode)):
        raise TypeError('Output filename has to be a string!')
    outfile = str(outfile)
    if outfile.find("~") == 0:
        outfile = os.path.expanduser('~') + outfile[1:]
    slash = outfile.rfind(os.sep)
    if slash >= 0 and not os.path.isdir(outfile[:outfile.rfind(os.sep)]):
        raise ValueError('Invalid path to save movie: '+outfile+'!')

    # Check format specifier for the movie: the if loop separates filename from extension
    # (use split here to prevent the user from creating abominations like `my.movie.mp4`)
    dot = outfile.rfind('.')
    if dot == 0: raise ValueError(outfile+' is not a valid filename!')          # e.g., outfile = '.name'
    if dot == len(outfile) - 1:                                                 # e.g., outfile = 'name.'
        outfile = outfile[:dot]
        dot     = -1
    if dot > 0:                                                                 # e.g., outfile = 'name.mp4'
        out_split = outfile.split('.')
        if len(out_split) > 2: raise ValueError(outfile+' is not a valid filename!')
        outfile = out_split[0]
        
        # If outfile had an extension but there was an add'l extension provided, warn the user
        if out_split[1] != str(ext) and str(ext) != 'mp4':
            print "WARNING: Using extension `"+out_split[1]+"` of output filename, not `"+str(ext)+"`!"
        ext = out_split[1]
    else:                                                                       # e.g., outfile = 'name'
        if str(ext) != ext:
            raise TypeError('Filename extension for movie has to be a string!')
        exl = str(ext).split('.')
        if len(exl) > 1: raise ValueError(ext+' is not a valid extension for a video file!')
        ext = exl[0]

    # Make sure `fps` is a positive integer
    scalarcheck(fps,'fps',kind='int',bounds=[1,np.inf])
    
    # Check if output filesize makes sense (if provided)
    if filesize is not None:
        scalarcheck(filesize,'filesize',bounds=[0,np.inf])

    # Check if `preset` is valid (if provided)
    if not isinstance(preset,(str,unicode)):
        raise TypeError('Preset specifier for video encoding has to be a string!')
    supported = ['ultrafast','superfast','veryfast','faster','fast','medium','slow','slower','veryslow','placebo']
    if supported.count(preset) == 0:
        opts = str(supported)
        opts = opts.replace('[','')
        opts = opts.replace(']','')
        raise ValueError('Preset `'+preset+'` not supported by ffmpeg. Supported options are: '+opts)

    # Now let's start to actually do something and set the null device based on which platform we're running on
    if os.uname()[0].find('Windows') > 0:
        nulldev = 'NUL'
    else:
        nulldev = '/dev/null'

    # Encode movie respecting provided file-size limit
    if filesize is not None:

        # Calculate movie length based on desired frame-rate and bit-rate such that given filesize is not exceeded (MB->kbit/s uses 8192)
        movie_len = np.ceil(num_imgs/fps)
        brate     = int(np.floor(filesize*8192/movie_len))
        
        # Use two-pass encoding to ensure maximum image quality while keeping the filesize within specified bounds
        os.system("ffmpeg -y -framerate "+str(fps)+" -f image2 -i "+imgpth+imgfmt+" "+\
                  "-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "
                  "-vcodec libx264 -preset "+preset+" -pix_fmt yuv420p -b:v "+str(brate)+"k -b:a 0k -pass 1 "+\
                  "-f "+ext+" "+nulldev+" && "+\
                  "ffmpeg -framerate "+str(fps)+" -f image2 -i "+imgpth+imgfmt+" "+\
                  "-vcodec libx264 -preset "+preset+" -pix_fmt yuv420p -b:v "+str(brate)+"k -b:a 0k -pass 2 "+\
                  "-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "+outfile+"."+ext)

    # Encode movie with no regards given to final size
    else:

        # Use a constant rate factor (incompatible with 2-pass encoding) to render the movie
        os.system("ffmpeg -framerate "+str(fps)+" -f image2 -i "+imgpth+imgfmt+" "+\
                  "-vcodec libx264 -preset "+preset+" -pix_fmt yuv420p "+\
                  "-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "+outfile+"."+ext)


    return

##########################################################################################
def build_hive(ax,branches,connections,node_vals=None,center=(0,0),branch_extent=None,positions=None,labels=None,
               angle=90,branch_colors=None,branch_alpha=1.0,node_cmap=plt.cm.jet,node_alpha=1.0,\
               edge_cmap=plt.cm.jet,edge_alpha=1.0,edge_vrange=[0,1],node_vrange=[0,1],node_sizes=0.01,\
               branch_lw=2,edge_lw=0.5,radians=0.15,labelsize=8,node_lw=0.5,nodes3d=False,sphere_res=40,\
               lightsource=None,full3d=False,viewpoint=None,ethresh=None,show_grid=False):
    """
    By default no threshold is applied to edges, i.e., even edges with zero-weights are drawn using 
    the respective value from the colormap `edge_cmap`. If you want to remove zero-weight edges use 
    the keyword argument `ethresh = 0`. 
    """
    # Define some default values in case the user didn't provide all optional inputs
    branch_beg = 0.05                   # Start of branches as displacement from `center` (if `branch_extent == None`)
    branch_end = 0.95                   # Length of branches (if `branch_extent == None`)
    pos_offset = 0.05                   # Offset percentage for nodes on branches (if `positions == None`)
    x_offset   = 0.1                    # Offset percentage for x-axis limits
    y_offset   = 0.1                    # Offset percentage for y-axis limits
    z_offset   = 0.1                    # Offset percentage for z-axis limits (only relevant if `full3d == True`)

    # Error checking for dictionaries with numeric values
    def check_dict(dct,name):
        try:
            for branch in dct.keys():
                dct[branch] = np.array(dct[branch])
        except:
            raise TypeError('The provided '+name+' have to be a dictionary with the same keys as `branches`!')
        for branch, nodes in dct.items():
            arrcheck(nodes,'vector',name)
            if branches[branch].size != nodes.size:
                raise ValueError("Provided branches and "+name+" don't match up!")

    # Amend `FancyArrowPatch` by 3D capabilities 
    # (taken from http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector)
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

    # Check if `ax` is really an mpl axis object
    try:
        plt.sca(ax)
    except:
        raise TypeError("Could not make axis "+str(ax)+" active!")

    # See if `branches` is a dictionary of branch numbers/labels holding node-numbers 
    if not isinstance(branches,dict):
        raise TypeError('The input `branches` has to be dictionary-like, not '+type(branches).__name__)
    try:
        for branch in branches.keys():
            branches[branch] = np.array(branches[branch]).squeeze()
    except:
        raise TypeError('Branches must be provided as dictionary of node arrays/lists!!')
    for nodes in branches.values():
        arrcheck(nodes,'vector','node indices')
    branch_arr = np.array(branches.keys())
    num_branches = branch_arr.size
    if num_branches == 1:
        raise ValueError('Only one branch found - no bueno')
    if type(branches).__name__ != 'OrderedDict':
        branch_arr = np.sort(branch_arr)                # if we have a regular dict, sort its keys
    node_arr = []
    for nodes in branches.values():
        node_arr += list(nodes)
    node_arr = np.unique(node_arr)
    num_nodes = node_arr.size
    if np.any(np.diff(node_arr) != 1) or node_arr.min() != 0:
        raise ValueError('Node numbers have to be contiguous in ascending order starting with 0!')
    for br in branches.keys():
        branches[br] = np.array(branches[br])

    # See if `connections` is a 2d array that matches the provided branch dictionary
    arrcheck(connections,'matrix','connections',bounds=[0,1])
    if connections.shape[0] != num_nodes:
        raise ValueError('Number of nodes does not match connection matrix!')

    # Let's see if we're going to have fun in 3D
    if not isinstance(nodes3d,bool):
        raise TypeError('Three-dimensional nodes are activated using a binary True/False flag!')
    if not isinstance(full3d,bool):
        raise TypeError('Full 3D plotting is activated using a binary True/False flag!')
    if full3d:
        nodes3d = False         # just internally: turn off this switch to avoid confusion later on
    if not isinstance(show_grid,bool):
        raise TypeError('Grid is drawn or not based on a binary True/False flag!')
    if show_grid and not full3d:
        print "WARNING: 3D grid is only shown for full 3D plots!"

    # Now check resolution parameter for rendering spheres
    scalarcheck(sphere_res,'sphere_res',kind='int',bounds=[2,np.inf])
    if sphere_res >= 100:
        print "WARNING: The resolution parameter for nodal spheres is very large - rendering might take forever..."

    # See if a light-source for illumination was provided
    if lightsource is not None:
        if isinstance(lightsource,bool):
            if lightsource == True:
                lightsource = np.array([90,45])
            else:
                lightsource = None
        if lightsource is not None:
            lightsource = np.array(lightsource)
            arrcheck(lightsource,'vector','lightsource')
            if lightsource.min() < 0 or lightsource[0] > 360 or lightsource[1] > 90:
                raise ValueError("Light-source azimuth/elevation has to be between 0-360 and 0-90 degrees, respectively!")
            if lightsource.size != 2:
                raise ValueError("Light-source has to be provided as azimuth/altitude degrees!")

    # See if a threshold for drawing edges was provided, if not, don't use one
    if ethresh is not None:
        scalarcheck(ethresh,'ethresh',bounds=[0,1])

    # See if a camera position (in azimuth/elevation degrees) was provided, if not use some defaults
    if viewpoint is not None:
        viewpoint = np.array(viewpoint)
        if plt.is_numlike(viewpoint):
            arrcheck(viewpoint,'vector','viewpoint')
            if viewpoint.size != 2:
                raise ValueError("View-point has to be provided as azimuth/altitude degrees!")
        else:
            raise TypeError("View-point for illumination has to [`azdeg`,`altdeg`]!")
    else:
        viewpoint = np.array([-60,30])
        
    # See if nodal values were provided, if not create simple dict
    if node_vals is not None:
        check_dict(node_vals,'nodal values')
        for vals in node_vals.values():
            if vals.min() < 0 or vals.max() > 1:
                raise ValueError('Nodal values must be between zero and one!')
    else:
        node_vals = {}
        for branch, nodes in branches.items():
            node_vals[branch] = np.ones(branches[branch].shape)

    # See if center makes sense, if provided
    try:
        center = np.array(center)
    except:
        raise TypeError('Unsupported type for input `center`: '+type(dict).__name__)
    arrcheck(center,'vector','center')
    if np.all(center) == 0:
        if full3d:
            center = np.zeros((3,))
    else:
        if full3d == False and center.size != 2:
            raise ValueError("Center coordinates have to be two-dimensional!")
        if full3d == True and center.size != 3:
            raise ValueError("For 3D plots center coordinates have to be three-dimensional!")

    # See if branch lengths were provided, otherwise construct'em
    if branch_extent is not None:
        try:
            for branch in branches.keys():
                branch_extent[branch] = np.array(branch_extent[branch])
        except:
            raise TypeError("The provided branch dimensions have to be a dictionary with the same keys as `branches`!")
        for branch in branches.keys():
            arrcheck(branch_extent[branch],'vector','branch dimensions',[0,np.inf])
            if branch_extent[branch].size != 2:
                raise ValueError("Only two values by branch supported for branch dimensions!")
            if branch_extent[branch][0] >= branch_extent[branch][1]:
                raise ValueError("Branch dimensions have to be increasing (beginning -> end)!")
    else:
        branch_extent = {}
        for branch in branches.keys():
            branch_extent[branch] = [branch_beg,branch_end]

    # See if nodal positions were provided, if not create simple dict
    if positions is not None:
        check_dict(positions,'nodal positions')
        for branch in branches.keys():
            if positions[branch].min() < branch_extent[branch][0] or positions[branch].max() > branch_extent[branch][1]:
                raise ValueError('Nodal positions on branches must be within branch extent!')
    else:
        positions = {}
        for branch, extent in branch_extent.items():
            length = extent[1] - extent[0]
            offset = pos_offset*length
            positions[branch] = np.linspace(extent[0]+offset,extent[1]-offset,branches[branch].size)

    # See if labels were provided and make sense, otherwise don't use labels
    if labels is not None:
        try:
            for branch in branches.keys():
                labels[branch] = np.array(labels[branch])
        except:
            raise TypeError("The provided nodal labels have to be a dictionary with the same keys as `branches`!")
        for branch in branches.keys():
            if branches[branch].size != labels[branch].size:
                raise ValueError("Provided branches and nodal labels don't match up!")
            if plt.is_numlike(labels[branch]):
                raise ValueError("The provided nodal labels must be strings!")
        if full3d or nodes3d:
            print "WARNING: Due to limiations in mplot3d the positiong of text in 3d space is somewhat screwed up..."

    # Now make sure label font-size makes sense
    scalarcheck(labelsize,'labelsize',bounds=[0,np.inf])

    # Check branch angle(s) were provided, if not, generate'em
    if isinstance(angle,dict):
        try:
            for branch in branches.keys():
                tmp = np.array(angle[branch])
        except:
            raise TypeError("The provided branch angles have to be a dictionary with the same keys as `branches`!")
        for branch in branches.keys():
            if full3d:
                angle[branch] = np.array(angle[branch])
                arrcheck(angle[branch],'vector','3D branch angles')
                if len(angle[branch]) != 2:
                    raise ValueError("3D branch angles must be provided as two values per branch!")
                if angle[branch][0] < 0 or angle[branch][0] > 360:
                    raise ValueError("Azimuth must be between 0 and 360 degrees!")
                if angle[branch][1] < -90 or angle[branch][1] > 90:
                    raise ValueError("Elevation must be between -90 and +90 degrees!")
                azim = math.radians(angle[branch][0])
                elev = math.radians(angle[branch][1])
                elev = np.pi/2 - (elev > 0)*elev + (elev < 0)*np.abs(elev)
                angle[branch] = np.array([azim,elev])
            else:
                if not np.isscalar(angle[branch]) or not plt.is_numlike(angle[branch]) or not np.isreal(angle[branch]).all():
                    raise TypeError("Branch angles must be real-valued, one value per branch!")
                if np.isfinite(angle[branch]) == False:
                    raise ValueError("Branch angles must not be NaN or Inf!")
                if angle[branch] < 0 or angle[branch] > 360:
                    raise ValueError("Branch angles must be between 0 and 360 degrees!")
                angle[branch] = math.radians(ange[branch])
    elif np.isscalar(angle):
        scalarcheck(angle,'angle',bounds=[0,360])
        if full3d:
            angle = {}
            angle[branch_arr[0]] = np.zeros((2,)) # in spherical coordinates (main branch is vertical line from origin)
            start = 1/4*np.pi 
            degs = np.linspace(start,start+2*np.pi,num_branches)        # these are the "azimuth" angles (well, not really...)
            elev = 3/4*np.pi
            for br, branch in enumerate(branch_arr[1:]):         # Here order is important! Use the generated (sorted) array!
                angle[branch] = np.array([degs[br],elev])
        else:
            angle = math.radians(angle)
            degs = np.linspace(angle,angle+2*np.pi,num_branches+1)
            angle = {}
            for br, branch in enumerate(branch_arr):         # Here order is important! Use the generated (sorted) array!
                angle[branch] = degs[br]
    else:
        raise TypeError("Branch angles have to be provided either as scalar or dictionary!")

    # Check color-values of branches - if not provided, construct'em
    if branch_colors is not None:
        if isinstance(branch_colors,dict):
            for branch in branches.keys():
                if len(branch_colors[branch]) > 1:
                    raise ValueError("Only one color per branch is supported!")
                if plt.is_numlike(branch_colors[branch]):
                    raise ValueError("The provided branch colors must be strings!")
        elif isinstance(branch_colors,str):
            bc = branch_colors
            branch_colors = {}
            for branch in branches.keys():
                branch_colors[branch] = bc
        else:
            raise TypeError("The provided branch colors have to be either a string or "+\
                            "a dictionary with the same keys as `branches`!")
    else:
        branch_colors = {}
        for branch in branches.keys():
            branch_colors[branch] = 'Black'

    # Check node and edge color maps
    for cmap in [node_cmap,edge_cmap]:
        if type(cmap).__name__.find('Colormap') < 0:
            raise TypeError("Node/Edge colormaps have to be matplotlib colormaps!")

    # Check value ranges for nodes and edges
    for vrange in [node_vrange,edge_vrange]:
        try:
            vrange = np.array(vrange)
        except:
            raise TypeError('Unsupported type for node/edge value ranges: '+type(dict).__name__)
        arrcheck(vrange,'vector','node/edge value range',bounds=[0,1])
        if vrange.size != 2:
            raise ValueError("Node/Edge value range has to be two-dimensional!")
        if vrange[0] >= vrange[1]:
            raise ValueError("Node/Edge value range must strictly increasing!")

    # See if nodal sizes have been provided, if not construct dictionary 
    if isinstance(node_sizes,dict):
        check_dict(node_sizes,'nodal sizes')
        for vals in node_sizes.values():
            if vals.min() < 0:
                raise ValueError("Nodal sizes have to be non-negative!")
    elif np.isscalar(node_sizes):
        scalarcheck(node_sizes,'node_sizes',bounds=[0,np.inf])
        ns = node_sizes
        node_sizes = {}
        for branch,nodes in branches.items():
            node_sizes[branch] = ns*np.ones(nodes.shape)
    else:
        raise TypeError("Nodal sizes have to be provided either as scalar or dictionary!")

    # See if nodal alpha values have been provided, if not construct dictionary 
    if isinstance(node_alpha,dict):
        check_dict(node_alpha,'nodal alpha values')
        for vals in node_alpha.values():
            if vals.min() < 0 or vals.max() > 1:
                raise ValueError("Nodal alpha values have to be between zero and one!")
    elif np.isscalar(node_alpha):
        scalarcheck(node_alpha,'node_alpha',bounds=[0,1])
        ns = node_alpha
        node_alpha = {}
        for branch,nodes in branches.items():
            node_alpha[branch] = ns*np.ones(nodes.shape)
    else:
        raise TypeError("Nodal alpha values have to be provided either as scalar or dictionary!")

    # Now make sure node line-width makes sense
    scalarcheck(node_lw,'node_lw',bounds=[0,np.inf])
    if full3d:
        print "WARNING: Line-width specifications for nodes is ignored for full 3D plots!"

    # Check if line-widths for branches have been provided, otherwise assign default values
    if isinstance(branch_lw,dict):
        try:
            for branch in branches.keys():
                tmp = np.array(branch_lw[branch])
        except:
            raise TypeError("The provided branch line-widths have to be a dictionary with the same keys as `branches`!")
        for branch in branches.keys():
            if not np.isscalar(branch_lw[branch]) or not plt.is_numlike(branch_lw[branch]) or not np.isreal(branch_extent[branch]).all():
                raise ValueError("Branch line-widths must be real-valued, one value per branch!")
            if np.isfinite(branch_lw[branch]) == False:
                raise ValueError("Branch line-widths must not be NaN or Inf!")
            if branch_lw[branch] < 0:
                raise ValueError("Branch line-widths have to be non-negative!")
    elif np.isscalar(branch_lw):
        scalarcheck(branch_lw,'branch_lw',bounds=[-0.1,np.inf])
        bw = branch_lw
        branch_lw = {}
        for branch in branches.keys():
            branch_lw[branch] = bw
    else:
        raise TypeError("Branch line-widths have to be provided either as scalar or dictionary!")

    # Check if alpha-values for branches have been provided, otherwise assign default values
    if isinstance(branch_alpha,dict):
        try:
            for branch in branches.keys():
                tmp = np.array(branch_alpha[branch])
        except:
            raise TypeError("The provided branch alpha values have to be a dictionary with the same keys as `branches`!")
        for branch in branches.keys():
            if not np.isscalar(branch_alpha[branch]) or not plt.is_numlike(branch_alpha[branch]) or not np.isreal(branch_extent[branch]).all():
                raise ValueError("Branch alpha values must be real-valued, one value per branch!")
            if np.isfinite(branch_alpha[branch]) == False:
                raise ValueError("Branch alpha values must not be NaN or Inf!")
            if branch_alpha[branch] < 0 or branch_alpha[branch] > 1:
                raise ValueError("Branch alpha values must be between zero and one!")
    elif np.isscalar(branch_alpha):
        scalarcheck(branch_alpha,'branch_alpha',bounds=[0,1])
        bw = branch_alpha
        branch_alpha = {}
        for branch in branches.keys():
            branch_alpha[branch] = bw
    else:
        raise TypeError("Branch alpha values have to be provided either as scalar or dictionary!")

    # Check if line-widths for edges have been provided, otherwise assign default values
    if np.isscalar(edge_lw):
        scalarcheck(edge_lw,'edge_lw',bounds=[0,np.inf])
        edge_lw = np.ones(connections.shape) * edge_lw
    else:
        arrcheck(edge_lw,'matrix','edge_lw',bounds=[0,np.inf])
        if edge_lw.shape != connections.shape:
            raise ValueError("Edge line-widths have to be provided in the same format as connection array!")

    # Check if alpha values for edges have been provided, otherwise assign default values
    if np.isscalar(edge_alpha):
        scalarcheck(edge_alpha,'edge_alpha',bounds=[0,1])
        edge_alpha = np.ones(connections.shape) * edge_alpha
    else:
        arrcheck(edge_alpha,'matrix','edge_alpha',bounds=[0,1])
        if np.any(edge_alpha.shape != connections.shape):
            raise ValueError("Edge alpha values have to be provided in the same format as connection array!")

    # Check if an intial setting for the arch radian was provided, otherwise use the default
    if np.isscalar(radians):
        scalarcheck(radians,'radians')
    else:
        arrcheck(radians,'matrix','radians')
        if rsh[0] != num_branches:
            raise ValueError("Arch radians must be provided as square array matching no. of branches!!")

    # Prepare axis
    ax.set_aspect('equal')
    ax.hold(True)

    # If nodes have to be rendered as spheres, some tuning is required...
    if nodes3d or full3d:

        # Turn on 3d projection if nodes are to be rendered as spheres
        bgcol = ax.get_axis_bgcolor()
        ax = plt.gca(projection='3d',axisbg=bgcol)
        ax.hold(True)
        if not full3d:
            ax.view_init(azim=-90,elev=90)
        else:
            ax.view_init(azim=viewpoint[0],elev=viewpoint[1])

        # Turn off 3D grid and change background of panes (or not)
        if not show_grid:
            ax.grid(False)
            ax.w_xaxis.set_pane_color(colorConverter.to_rgb(bgcol))
            ax.w_yaxis.set_pane_color(colorConverter.to_rgb(bgcol))
            ax.w_zaxis.set_pane_color(colorConverter.to_rgb(bgcol))

        # Turn off all axes highlights
        ax.zaxis.line.set_lw(0)
        ax.set_zticks([])
        ax.xaxis.line.set_lw(0)
        ax.set_xticks([])
        ax.yaxis.line.set_lw(0)
        ax.set_yticks([])

        # Generate surface data for the prototype nodal sphere
        theta  = np.arange(-sphere_res,sphere_res+1,2)/sphere_res*np.pi
        phi    = np.arange(-sphere_res,sphere_res+1,2)/sphere_res*np.pi/2
        cosphi = np.cos(phi); cosphi[0] = 0; cosphi[-1] = 0
        sinth  = np.sin(theta); sinth[0] = 0; sinth[-1] = 0    
        xsurf  = np.outer(cosphi,np.cos(theta))
        ysurf  = np.outer(cosphi,sinth)
        zsurf  = np.outer(np.sin(phi),np.ones((sphere_res+1,)))
        
        # If virtual lighting is wanted, create a light source for illumination
        if lightsource is not None:
            light   = LightSource(*lightsource)
            rgb_arr = np.ones((zsurf.shape[0],zsurf.shape[1],3))

    # Start by truncating color-values based on vrange limits that were provided
    if [0,1] != node_vrange:
        node_cmap = plt.cm.ScalarMappable(norm=Normalize(node_vrange[0],node_vrange[1]),cmap=node_cmap).to_rgba
    if [0,1] != edge_vrange:
        edge_cmap = plt.cm.ScalarMappable(norm=Normalize(edge_vrange[0],edge_vrange[1]),cmap=edge_cmap).to_rgba

    # Plot branches and construct nodal patches (we do this no matter if we're 3-dimensional or not)
    node_patches = {}
    branch_dvecs = {}
    branch_kwargs = {'lw':-1, 'color': -np.ones((3,)), 'alpha': -1, 'zorder':1}
    for branch in branch_arr:

        # Compute normed directional vector of branch
        if full3d:
            azim,elev = angle[branch]
            bdry = branch_extent[branch][1]*np.array([np.sin(elev)*np.cos(azim),np.sin(elev)*np.sin(azim),np.cos(elev)])
        else:
            bdry = branch_extent[branch][1]*np.array([np.cos(angle[branch]),np.sin(angle[branch])])
        vec  = bdry - center
        vec /= np.linalg.norm(vec)
        bstart = center + branch_extent[branch][0]*vec
        branch_dvecs[branch] = vec

        # Plot branch as straight line
        branch_kwargs['lw'] = branch_lw[branch]
        branch_kwargs['color'] = branch_colors[branch]
        branch_kwargs['alpha'] = branch_alpha[branch]
        if full3d:
            plt.plot([bstart[0],bdry[0]],[bstart[1],bdry[1]],zs=[bstart[2],bdry[2]],zdir='z',**branch_kwargs)
        elif nodes3d:
            plt.plot([bstart[0],bdry[0]],[bstart[1],bdry[1]],zs=0,zdir='z',**branch_kwargs)
        else:
            plt.plot([bstart[0],bdry[0]],[bstart[1],bdry[1]],**branch_kwargs)

        # Now construct circular patches for all nodes and save'em in the `patch_list` list (and the `node_patch` dict)
        patch_list = []
        for node in xrange(branches[branch].size):
            pos = center + vec*positions[branch][node]
            patch_list.append(Circle(pos,radius=node_sizes[branch][node],\
                                     facecolor=node_cmap(node_vals[branch][node]),\
                                     alpha=node_alpha[branch][node],\
                                     lw=node_lw,\
                                     zorder=3))
        node_patches[branch] = patch_list

    # Determine if our network is directed or not
    sym = issym(connections)

    # Allocate dicionary for all edge-related parameters
    edge_kwargs = {'connectionstyle':'a string','lw': -1, 'alpha': -1, 'color': -np.ones((3,)), 'zorder': 2}
    if sym:
        edge_kwargs['arrowstyle'] = '-'
    else:
        edge_kwargs['arrowstyle'] = '-|>'

    # 3D is again the special snowflake, so do this nonsense separately...
    if full3d:

        # In a fully three-dimensional environment, we can't go 'round the tree to plot edges - everything may be connected
        seen = []
        for br, branch in enumerate(branch_arr):
            seen.append(branch)
            neighbors = np.setdiff1d(branch_arr,seen)
            for twig in neighbors:
                if np.isscalar(radians):
                    br_vec = branch_dvecs[branch]
                    tw_vec = branch_dvecs[twig]
                    ang_bt = np.arctan2(np.linalg.norm(np.cross(br_vec,tw_vec)),br_vec.dot(tw_vec))
                    ang_bt += 2*np.pi*(ang_bt >= 0)
                    rad    = (-1)**(ang_bt > np.pi)*radians
                else:
                    rad = radians[br,np.where(branch_arr==twig)[0][0]]
                for n1,node1 in enumerate(branches[branch]):
                    for n2,node2 in enumerate(branches[twig]):
                        edge_kwargs['connectionstyle'] = 'arc3,rad=%s'%rad
                        edge_kwargs['lw'] = edge_lw[node1,node2]
                        edge_kwargs['alpha'] = edge_alpha[node1,node2]
                        edge_kwargs['color'] = edge_cmap(connections[node1,node2])
                        xcoords = [node_patches[branch][n1].center[0],node_patches[twig][n2].center[0]]
                        ycoords = [node_patches[branch][n1].center[1],node_patches[twig][n2].center[1]]
                        zcoords = [node_patches[branch][n1].center[2],node_patches[twig][n2].center[2]]
                        if sym:
                            if connections[node1,node2] > ethresh:
                                ax.add_artist(Arrow3D(xcoords,ycoords,zcoords,**edge_kwargs))
                        else:
                            if connections[node1,node2] > ethresh:
                                ax.add_artist(Arrow3D(xcoords,ycoords,zcoords,**edge_kwargs))
                            if connections[node2,node1] > ethresh:
                                rad = - rad
                                edge_kwargs['connectionstyle'] = 'arc3,rad=%s'%rad
                                ax.add_artist(Arrow3D(xcoords[::-1],ycoords[::-1],zcoords[::-1],**edge_kwargs))

    # 2D rendering of edges is a lot easier (just go branch by branch)
    else:
        for br, branch in enumerate(branch_arr):
            if br < branch_arr.size-1:
                twig = branch_arr[br+1]
            else:
                twig = branch_arr[0]
            if np.isscalar(radians):
                br_vec = branch_dvecs[branch]
                tw_vec = branch_dvecs[twig]
                ang_bt = np.arctan2(tw_vec[1],tw_vec[0]) - np.arctan2(br_vec[1],br_vec[0])
                ang_bt += 2*np.pi*(ang_bt < 0)
                rad    = (-1)**(ang_bt > np.pi)*radians
            else:
                rad = radians[br,br+1]
            for n1,node1 in enumerate(branches[branch]):
                for n2,node2 in enumerate(branches[twig]):
                    edge_kwargs['connectionstyle'] = 'arc3,rad=%s'%rad
                    edge_kwargs['lw'] = edge_lw[node1,node2]
                    edge_kwargs['alpha'] = edge_alpha[node1,node2]
                    edge_kwargs['color'] = edge_cmap(connections[node1,node2])
                    xcoords = [node_patches[branch][n1].center[0],node_patches[twig][n2].center[0]]
                    ycoords = [node_patches[branch][n1].center[1],node_patches[twig][n2].center[1]]
                    if sym:
                        if connections[node1,node2] > ethresh:
                            if nodes3d:
                                ax.add_artist(Arrow3D(xcoords,ycoords,[0,0],**edge_kwargs))
                            else:
                                ax.add_patch(FancyArrowPatch(node_patches[branch][n1].center,\
                                                             node_patches[twig][n2].center,\
                                                             **edge_kwargs))
                    else:
                        if connections[node1,node2] > ethresh:
                            if nodes3d:
                                ax.add_artist(Arrow3D(xcoords,ycoords,[0,0],**edge_kwargs))
                            else:
                                ax.add_patch(FancyArrowPatch(node_patches[branch][n1].center,\
                                                             node_patches[twig][n2].center,\
                                                             **edge_kwargs))
                        if connections[node2,node1] > ethresh:
                            rad = - rad
                            edge_kwargs['connectionstyle'] = 'arc3,rad=%s'%rad
                            if nodes3d:
                                ax.add_artist(Arrow3D(xcoords[::-1],ycoords[::-1],[0,0],**edge_kwargs))
                            else:
                                ax.add_patch(FancyArrowPatch(node_patches[twig][n2].center,\
                                                             node_patches[branch][n1].center,\
                                                             **edge_kwargs))

    # Finally, draw nodes and compute maximal extent of branches
    top = -np.inf 
    bot = np.inf 
    lft = np.inf 
    rgt = -np.inf
    up  = -np.inf
    lo  = np.inf
    lbl_kwargs = {'fontsize':labelsize,'ha':'center','va':'center'}
    nd_kwargs = {'cstride':1,'rstride':1,'linewidth':0,'antialiased':False,'alpha':-1,'zorder':-1}
    zcord = 0
    for branch in branch_arr:
        branch_tvec = branch_extent[branch][1]*branch_dvecs[branch]
        top = np.max([top,branch_tvec[1]])
        bot = np.min([bot,branch_tvec[1]])
        lft = np.min([lft,branch_tvec[0]])
        rgt = np.max([rgt,branch_tvec[0]])
        if full3d:
            up = np.max([up,branch_tvec[2]])
            lo = np.min([lo,branch_tvec[2]])
        for node in xrange(branches[branch].size):
            if nodes3d or full3d:
                circ = node_patches[branch][node]
                if full3d:
                    zcord = circ.center[2]
                nd_kwargs['alpha'] = circ.get_alpha()
                nd_kwargs['zorder'] = circ.get_zorder()
                if lightsource is not None:
                    nd_kwargs['facecolors'] = light.shade_rgb(rgb_arr*np.array(circ.get_facecolor()[:-1]),zsurf)
                else:
                    nd_kwargs['color'] = circ.get_facecolor()
                ax.plot_surface(circ.get_radius()*xsurf + circ.center[0],\
                                circ.get_radius()*ysurf + circ.center[1],\
                                circ.get_radius()*zsurf + zcord,\
                                **nd_kwargs)
                if labels is not None:
                    if nodes3d:
                        lcord = 1.5*circ.get_radius()
                    else:
                        lcord = node_patches[branch][node].center[2]
                    ax.text(node_patches[branch][node].center[0],\
                            node_patches[branch][node].center[1],\
                            lcord,\
                            labels[branch][node],**lbl_kwargs)
            else:
                ax.add_patch(node_patches[branch][node])
                if labels is not None:
                    ax.text(node_patches[branch][node].center[0],\
                            node_patches[branch][node].center[1],\
                            labels[branch][node],**lbl_kwargs)

    # Set axes limits based on extent of branches
    x_width = rgt - lft
    y_heght = top - bot
    ax.set_xlim(left=lft-x_offset*x_width,right=rgt+x_offset*x_width)
    ax.set_ylim(bottom=bot-y_offset*y_heght,top=top+y_offset*y_heght)
    if full3d:
        z_len = up - lo
        ax.set_zlim(bottom=lo-z_offset*z_len,top=up+z_offset*z_len)

    # Draw the beauty and get the hell out of here
    plt.draw()
    if full3d or nodes3d: plt.axis('equal')

##########################################################################################
def arrcheck(arr,kind,varname,bounds=None):
    """
    Local helper function performing sanity checks on arrays (1d/2d/3d)
    """
    
    try:
        sha = arr.shape
    except:
        raise TypeError('Input `'+varname+'` must be a NumPy array, not '+type(arr).__name__+'!')

    if kind == 'tensor':
        if len(sha) != 3:
            raise ValueError('Input `'+varname+'` must be a `N`-by-`N`-by-`k` NumPy array')
        if (min(sha[0],sha[1])==1) or (sha[0]!=sha[1]):
            raise ValueError('Input `'+varname+'` must be a `N`-by-`N`-by-`k` NumPy array!')
        dim_msg = '`N`-by-`N`-by-`k`'
    elif kind == 'matrix':
        if len(sha) != 2:
            raise ValueError('Input `'+varname+'` must be a `N`-by-`N` NumPy array')
        if (min(sha)==1) or (sha[0]!=sha[1]):
            raise ValueError('Input `'+varname+'` must be a `N`-by-`N` NumPy array!')
        dim_msg = '`N`-by-`N`'
    elif kind == 'vector':
        sha = arr.squeeze().shape
        if len(sha) != 1:
            raise ValueError('Input `'+varname+'` must be a NumPy 1darray')
        if min(sha)==1:
            raise ValueError('Input `'+varname+'` must be a NumPy 1darray of length `N`!')
        dim_msg = ''
    else:
        print "Error checking could not be performed - something's wrong here..."
    if not plt.is_numlike(arr) or not np.isreal(arr).all():
        raise TypeError('Input `'+varname+'` must be a real-valued '+dim_msg+' NumPy array!')
    if np.isfinite(arr).min() == False:
        raise ValueError('Input `'+varname+'` must be a real valued NumPy array without Infs or NaNs!')
        
    if bounds is not None:
        if arr.min() < bounds[0] or arr.max() > bounds[1]:
            raise ValueError("Values of input array `"+varname+"` must be between "+str(bounds[0])+\
                             " and "+str(bounds[1])+"!")

##########################################################################################
def scalarcheck(val,varname,kind=None,bounds=None):
    """
    Local helper function performing sanity checks on scalars
    """

    if not np.isscalar(val) or not plt.is_numlike(val) or not np.isreal(val).all():
        raise TypeError("Input `"+varname+"` must be a real scalar!")
    if not np.isfinite(val):
        raise TypeError("Input `"+varname+"` must be finite!")

    if kind == 'int':
        if (round(val) != val):
            raise ValueError("Input `"+varname+"` must be an integer!")

    if bounds is not None:
        if val < bounds[0] or val > bounds[1]:
            raise ValueError("Input scalar `"+varname+"` must be between "+str(bounds[0])+" and "+str(bounds[1])+"!")
