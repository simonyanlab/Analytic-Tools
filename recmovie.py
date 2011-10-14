# recmovie.py - make a movie from given matplotlib-figures

from __future__ import division

import os

import datetime

import shutil

from glob import glob

from string import join

def recmovie(figobj=None, movie=None, savedir=None, fps=None):
    """
    RECMOVIE saves matplotlib figures and generates a movie sequence. 

    Parameters:
    -----------

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


    Returns
    -------

    None


    Notes
    -----

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


    See also
    --------

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
