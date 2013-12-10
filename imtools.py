# imtools.py - Simple but often used load/save/show shortcuts for images
# 
# Author: Stefan Fuertinger
# Juni 13 2012

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import shutil
from glob import glob
from string import join

##########################################################################################
def imview(Im,interpolation="nearest",vmin=None,vmax=None):
    """
    IMVIEW plots a grey-scale image using "sane" defaults for Matplotlib's imshow. 

    Inputs:
    -------
    Im: NumPy 2darray
        Grey-scale image to plot (has to be a 2D array)
    interpolation: str
        String determining interpolation to be used for plotting. Default 
        value is "nearest". Recommended other values are "bilinear" or "lanczos". 
        See Matplotlib's imshow-documentation for details. 
    vmin: float
        Minimal luminance to be used in plot (if None then vmin = Im.min())
    vmax: float
        Maximal luminance to be used in plot (if None then vmax = Im.max())
       
    Returns:
    --------
    None 

    Notes:
    ------
    None 

    See also:
    ---------
    Matplotlib's imshow

    """

    # Sanity checks
    # Im
    if type(Im).__name__ != "ndarray":
        raise TypeError("Im has to be a NumPy ndarray!")
    else:
        if len(Im.shape) > 2: raise ValueError("Im has to be 2-dimensional!")
        try: Im.shape[1]
        except: raise ValueError("Im has to be an image")
        if np.isnan(Im).max() == True or np.isinf(Im).max() == True:
            raise ValueError("Im must not contain NaNs or Infs!")
        
    # interpolation
    if type(interpolation).__name__ != "str":
        raise TypeError("interpolation has to be a string!")

    # vmin
    if vmin != None:
        try: float(vmin)
        except: raise TypeError("vmin has to be minimal luminance level, i.e. float!")
        if np.isnan(vmin) == True or np.isinf(vmin) == True:
            raise ValueError("vmin must not be NaN or Inf!")

    # vmax
    if vmax != None:
        try: float(vmax)
        except: raise TypeError("vmax has to be maximal luminance level, i.e. float!")
        if np.isnan(vmax) == True or np.isinf(vmax) == True:
            raise ValueError("vmax must not be NaN or Inf!")
        
    # Now do something
    plt.imshow(Im,cmap="gray",interpolation=interpolation,vmin=vmin,vmax=vmax)
    plt.axis("off")
    plt.draw()

    return

##########################################################################################
def imload(fstr,nrm=False):
    """
    IMLOAD loads an image using Matplotlib's imread

    Inputs:
    -------
    fstr : string
        String holding path and filename (including filename extension!) 
        of the image to load. Note that this code only calls imread thus
        only image formats supported by imread will work. 
    nrm : bool
        If nrm = True the loaded image is normalized, such that It.max() = 1
        and It.min() = 0. Default nrm = False. 
       
    Returns:
    --------
    It : NumPy ndarray
        Array representation of the image (2D array for grey-scale images, 
        (:,:,3)-array for RGB images and (:,:,4)-array for RGBA images). 

    Notes:
    ------
    None 

    See also:
    ---------
    Matplotlib's imread
    """

    # Sanity checks
    if type(fstr).__name__ != "str":
        raise TypeError("fstr has to be a string!")
    if type(nrm).__name__ != "bool":
        raise TypeError("nrm has to be True or False!")

    # Get the extension of the image to be loaded
    try:
        ext = fstr.split(".")[1]
    except:
        print "ERROR: No image extension specified! Aborting..."

    # Load the image 
    It = plt.imread(fstr) 

    # If we have a tif-file matplotlib loads the image upside down
    tifvers = ("tif","TIF","tiff","TIFF")
    if tifvers.count(fstr) != 0:
        It = It[::-1,:]

    # Convert it to float and normalize it s.t. It.max() = 1.0
    It = It.astype(float)

    # Normalize the image if wanted
    if nrm: It = normalize(It)

    return It

##########################################################################################
def imwrite(figobj,fstr,dpi=None):
    """
    IMWRITE saves a Matplotlib figure camera-ready using a "tight" bounding box

    Inputs:
    -------
    figobj : matplotlib figure
        Matplotlib figure object to be saved as image. 
    fstr : string
        String holding the filename to be used to save the figure. If 
        a specific file format is wanted, provide it with fstr, e.g., 
        fstr = 'output.tiff'. If fstr does not contain a filename extension
        the matplotlib default (png) will be used. 
    dpi : integer >= 1
        The wanted resolution of the output in dots per inch. If None the 
        matplotlib default will be used. 
       
    Returns:
    --------
    None

    Notes:
    ------
    This is a humble attempt to get rid of the huge white areas around a plot 
    that are generated by Matplotlib's savefig when saving a figure as an 
    image using default values. It tries to mimick export_fig for MATLAB. 
    The result, however, is not perfect yet...

    See also:
    ---------
    Matplotlib's savefig
    """

    # Sanity checks
    if type(figobj).__name__ != "Figure":
        raise TypeError("figobj has to be a valid matplotlib Figure object!")

    if type(fstr).__name__ != "str":
        raise TypeError("fstr has to be a string!")

    # Check if filename extension has been provided
    dt = fstr.rfind('.')
    if dt == -1:
        fname = fstr+'.png'
        ext   = 'png'
    elif len(fstr[dt+1:]) < 2: 
        print "Invalid filename extension: "+fstr[dt:]+" Defaulting to png..."
        fname = fstr[:dt]+'.png'
        ext   = 'png'
    else: 
        fname = fstr
        ext   = fstr[dt+1:]

    # Save the figure using "tight" options for the bounding box
    figobj.savefig(fname,bbox_inches="tight",ppad_inches=0,dpi=dpi,format=ext)

    return

##########################################################################################
def normalize(I,a=0,b=1):
    """
    NORMALIZE rescales a numpy ndarray

    Inputs:
    -------
    I: NumPy ndarray
        Array to be normalized
    a : float
        Floating point number being the lower normalization bound. 
        By default a = 0. (Note that it has to hold that a < b)
    b : float
        Floating point number being the upper normalization bound. 
        By default b = 1. (Note that it has to hold that a < b)
       
    Returns:
    --------
    In : NumPy ndarray
        Scaled version of the input array I, such that a = In.min() and 
        b = In.max()

    Notes:
    ------
    None 

    Examples:
    ---------
    I = array([[-1,.2],[100,0]])
    In = normalize(I,a=-10,b=12)
    In 
    array([[-10.        ,  -9.73861386],
           [ 12.        , -10.        ]])

    See also:
    ---------
    None 
    """

    # Ensure that I is a numpy-ndarray
    try: tmp = I.size == 1
    except TypeError: raise TypeError('I has to be a numpy ndarray!')
    if (tmp): raise ValueError('I has to be a numpy ndarray of size > 1!')

    # If normalization bounds are user specified, check them
    try: tmp = b <= a
    except TypeError: raise TypeError('a and b have to be scalars satisfying a < b!')
    if (tmp):
        raise ValueError('a has to be strictly smaller than b!')
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
def blendedges(Im,chim):
    """
    BLENDEDGES superimposes a (binary) edge set on an grayscale image using Matplotlib's imshow

    Inputs:
    -------
    Im: NumPy 2darray
        Greyscale image (has to be a 2D array)
    chim: NumPy 2darray
        Binary edge map (has to be a 2D array). Note that the edge map must only contain 
        the valus 0 and 1. 
       
    Returns:
    --------
    None 

    Notes:
    ------
    None 

    See also:
    ---------
    Matplotlib's imshow
    .. http://stackoverflow.com/questions/2495656/variable-alpha-blending-in-pylab
    """

    # Sanity checks
    if type(Im).__name__ != "ndarray":
        raise TypeError("Im has to be a NumPy ndarray!")
    else:
        if len(Im.shape) > 2: raise ValueError("Im has to be 2-dimensional!")
        try: Im.shape[1]
        except: raise ValueError("Im has to be an image!")
        if np.isnan(Im).max() == True or np.isinf(Im).max() == True:
            raise ValueError("Im must not contain NaNs or Infs!")

    if type(chim).__name__ != "ndarray":
        raise TypeError("chim has to be a NumPy ndarray!")
    else:
        if len(chim.shape) > 2: raise ValueError("chim has to be 2-dimensional!")
        try: chim.shape[1]
        except: raise ValueError("chim has to be an edge map!")
        if np.isnan(chim).max() == True or np.isinf(chim).max() == True:
            raise ValueError("chim must not contain NaNs or Infs!")
        chim = chim.astype(float)
        chiu = np.unique(chim)
        if chiu.size != 2: raise ValueError("chim has to be binary!")
        if chiu.min() != 0 or chiu.max() != 1: raise ValueError("chim has to be a binary edge map!")

    # Now do something
    plt.imshow(Im,cmap="gray",interpolation="nearest")
    plt.hold(True)
    plt.imshow(mycmap(chim))
    plt.axis("off")
    plt.draw()

    return

##########################################################################################
def mycmap(x):
    """
    Generate a custom color map, setting alpha values to one on edge
    points, and to zero otherwise
    
    Notes:
    ------
    This code is based on the suggestion found at
    .. http://stackoverflow.com/questions/2495656/variable-alpha-blending-in-pylab 
    """

    # Convert edge map to Matplotlib colormap (shape (N,N,4))
    tmp = plt.cm.hsv(x)

    # Set alpha values to one on edge points
    tmp[:,:,3] = x

    return tmp

##########################################################################################
def recmovie(figobj=None, movie=None, savedir=None, fps=None):
    """
    RECMOVIE saves matplotlib figures and generates a movie sequence. 

    Inputs:
    -------
    figobj : matplotlib figure
        Figure that shall be saved. 
    movie : str 
        String determining the filename of the generated movie. 
    savedir : str
        String determining the directory figobj shall be saved
        into or name of directory holding image files that 
        shall be composed into a movie. 
    fps : int
        Integer determining the frames per second in the movie
        that is generated. 

    Returns:
    --------
    None


    Notes:
    ------
    RECMOVIE(FIGOBJ) saves the matplotlib figure-object FIGOBJ in the default directory
    _tmp as png-image. If the default directory is empty the image will be _tmp0000.png, 
    otherwise the highest no. in the png-file-names incremented by one will be
    used as filename. 

    RECMOVIE(FIGOBJ,SAVEDIR="somedir") saves the matplotlib figure-object FIGOBJ in the 
    directory defined by the string SAVEDIR. If the directory does not exist, it will
    be generated. 
    If the directory SAVEDIR is empty the image will be _tmp0000.png, otherwise the 
    highest no. in the png-file-names incremented by one will be used as filename. 

    RECMOVIE() will attempt to use mencoder to generate an avi-movie composed of the 
    png-images found in the default directory _tmp. The movie's default name will be
    composed of the default prefix _tmp and the system's current date and time. 
    After the movie has been generated the default-directory _tmp and its contents 
    will be deleted. 

    RECMOVIE(MOVIE="somename") will attempt to use mencoder to generate an avi-movie 
    composed of the png-images found in the default directory _tmp. The movie's 
    name will be composed of the string MOVIE, hence here somename.avi. 
    After the movie has been generated the default-directory _tmp and its contents 
    will be deleted. 

    RECMOVIE(SAVEDIR="somedir") will attempt to use mencoder to generate an avi-movie 
    composed of the png-images found in the directory specified by the string SAVEDIR. 
    The movie's name will be composed the default prefix _tmp and the system's current 
    date and time. 
    After the movie has been generated it will be moved to the directory SAVEDIR. If 
    a movie-file of the same name exists in SAVEDIR a WARNING is printed and the movie 
    will not be moved. 

    RECMOVIE(MOVIE="somename",SAVEDIR="somedir") will attempt to use mencoder to generate 
    an avi-movie composed of the png-images found in the directory specified by the string 
    SAVEDIR. The movie's name will be composed of the string MOVIE, hence here somename.avi. 
    After the movie has been generated it will be moved to the directory SAVEDIR. If 
    a movie-file of the same name exists in SAVEDIR a WARNING is printed and the movie 
    will not be moved. 

    RECMOVIE(FIGOBJ,MOVIE="somename",SAVEDIR="somedir") will ONLY save the 
    matplotlib-figure-object FIGOBJ in the directory defined by the string SAVEDIR. The 
    optional argument MOVIE will be ignored. 

    Note: the default-directory, image-format and movie-type can be changed in the source code 
    by editing the variables "prefix", "imgtype" and "movtype". 

    See also:
    ---------
    .. http://matplotlib.sourceforge.net/examples/animation/movie_demo.html
    .. http://matplotlib.sourceforge.net/faq/howto_faq.html#make-a-movie 

    """

    # Set default values
    prefix    = "_tmp"
    numdigits = "%04d"
    imgtype   = "png"
    movtype   = "avi"
    fpsno     = 25
    
    now         = datetime.datetime.now()
    savedirname = prefix
    moviename   = "movie"+"_"+repr(now.hour)+repr(now.minute)+repr(now.second)

    # Assign defaults
    if movie == None:
        movie = moviename
    if savedir == None:
        savedir = savedirname
    if fps == None:
        fps = fpsno

    # Sanity checks
    if figobj != None:
        if type(figobj).__name__ != "Figure":
            raise TypeError("figobj has to be a valid matplotlib Figure object!")
        # try:
        #     figobj.number
        # except AttributeError: raise TypeError("figobj has to be a valid matplotlib Figure object!")

    if type(movie).__name__ != "str":
        raise TypeError("movie has to be a string!")

    if type(savedir).__name__ != "str":
        raise TypeError("savedir has to be a string!")

    try: 
        fps = int(fps) # convert possible float argument to integer, if it does not work raise a TypeError
    except:
        raise TypeError("fps has to be an integer (see man mencoder for details)!")

    # try:
    #     movie+"string"
    # except TypeError: raise TypeError("movie has to be a string!")

    # try:
    #     savedir+"string"
    # except TypeError: raise TypeError("savedir has to be a string!")

    # Check if mencoder is available
    if os.system("which mencoder > /dev/null") != 0:
        print "\n\nWARNING: mencoder was not found on your system - movie generation won't work!!!\n\n"

    # Check if movie already exists, if yes abort
    if len(glob(movie)) != 0:
        errormsg = "Movie %s already exists! Aborting..."%savedir
        raise ValueError(errormsg)

    # If not already existent, create directory savedir
    if len(glob(savedir)) == 0:
        os.mkdir(savedir)

    # If a none-default savedir was chosen, automatically keep images
    # and move movie to this non-standard savedir
    if savedir != savedirname:
        keepimgs  = 1
    else:
        keepimgs = 0

    # List all imgtype-files in directory savedir
    filelist = glob(savedir+os.sep+"*."+imgtype)

    # If we have been called with a figobj save it in savedir
    if figobj != None:

        # If there are already imgtype-files in the directory then filelist!=0
        if len(filelist) != 0:

            # Sort filelist, s.t. the file having the highest no. is the last element
            filelist.sort()
            scounter = filelist[-1]

            # Remove the path and prefix from the last elements filename
            scounter = scounter.replace(savedir+os.sep+prefix,"")

            # Split the name further s.t. it is only number+imgtype
            scounter = scounter.split(".")[0]

            # Convert the string holding the file's no. to an integer
            counter  = int(scounter) + 1

        # No files are present in savedir, start with 0
        else:
            counter = 0

        # Generate the name the file is stored under (prefix+numdigits(counter).imgtype, e.g. _tmp0102.png)
        fname = savedir+os.sep+prefix+numdigits+"."+imgtype
        fname = fname%counter

        # Save the figure using the just generated filename
        figobj.savefig(fname)

    # User wants to generate a movie consisting of imgtyp-files in a directory savedir
    else:

        # Check if there are any files to process in savedir, if not raise an error
        if len(filelist) == 0:
            errormsg = "No %s-images found in directory %s! Aborting..."%(imgtype,savedir)
            raise ValueError(errormsg)

        # This is the standard command used to generate the movie
        command = ["mencoder",
                   "mf://*.png",
                   "-mf",
                   "type=png:w=800:h=600:fps=25",
                   "-ovc",
                   "lavc",
                   "-lavcopts",
                   "vcodec=mpeg4",
                   "-oac",
                   "copy",
                   "-o",
                   "output.avi"]

        # Make necessary changes here (like pointing to the right savedir, imgtype, movie,...)
        command[1]  = "mf://"+savedir+os.sep+"*."+imgtype
        command[3]  = "type="+imgtype+":w=800:h=600:fps="+str(fps)
        command[-1] = movie+"."+movtype

        # Call mencoder to generate movie
        os.system(join(command))

        # If we have been called using the default savedir, erase it (and its contents)
        if keepimgs == 0:
            for f in glob(savedir+os.sep+"*"+imgtype):
                os.unlink(f)
            os.rmdir(savedir)

        # If not don't erase it but (try to) move the generated movie into this savedir
        else:
            try:
                shutil.move(movie+"."+movtype,savedir+os.sep)
            except:
               print "\n\n\nWARNING: Movie %s already exists in directory %s. I won't move it there but keep it here. "%(movie,savedir)
