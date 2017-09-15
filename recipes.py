# recipes.py - Here are some general purpose convenience functions
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: June 25 2013
# Last modified: <2017-09-15 16:07:08>

from __future__ import division
import sys
import re
import fnmatch
import os
from numpy.linalg import norm
import numpy as np
from texttable import Texttable
from datetime import datetime, timedelta
from scipy import ndimage
import matplotlib.pyplot as plt

##########################################################################################
def query_yes_no(question, default=None):
    """
    Ask a yes/no question via `raw_input()` and return the answer.

    Parameters
    ----------
    question : str
        The question to be printed in the prompt
    default : str
        The presumed answer that is used in case the <Return> key is pressed
        (must be either "yes" or "no"). 
        If `default` is `None` then a definitive answer is required 
        (pressing <Return> will re-print `question` in the prompt)

    Returns
    -------
    answer : bool
        Either `True` if the input was "yes"/"y" or `False` otherwise. 

    Notes
    -----
    This code is a slightly modified version of recipe no. 577058 from ActiveState 
    written by Trent Mick. 

    See also
    --------
    ActiveState : ActiveState Code Recipe #577058 currently available 
                  `here <http://code.activestate.com/recipes/577058/>`_
    """

    # Check mandatory input
    if not isinstance(question,(str,unicode)):
        raise TypeError('Input question has to be a string!')

    # Parse optional `default` answer
    valid = {"yes":True,   "y":True,  "ye":True,
             "no":False,     "n":False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("Invalid default answer: '%s'" % default)

    # Do the actual work
    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")

##########################################################################################
def natural_sort(lst): 
    """
    Sort a list/NumPy 1darray in a "natural" way

    Parameters
    ----------
    lst : list or NumPy 1darray
        Python list or 1darray of strings

    Returns
    -------
    lst_sort : list or NumPy 1darray
        Lexicographically sorted version of the input list `lst`

    Notes
    -----
    This function was originally intended to perform a natural sorting of a file-listing
    (see Coding Horror's 
    `note <http://www.codinghorror.com/blog/2007/12/sorting-for-humans-natural-sort-order.html>`_ 
    on this topic for more details). Briefly, an input list `lst` of strings 
    containing digits is sorted such that the actual numerical value of the 
    digits is respected (see Examples for more details). 
    The code below is based on a Stackoverflow submission by Mark Byers, currently available 
    `here <http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort>`_. 

    Examples
    --------
    Calling `glob` in a directory containing files named `Elm` and `elm` plus 
    two-digit suffixes will result in a file listing sorted as follows:

    >>> lst = ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']

    Using `natural_sort` to order `lst` yields

    >>> natural_sort(lst)
    ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']

    See also
    --------
    None
    """

    # Check our single mandatory input argument
    if not isinstance(lst,(list,np.ndarray)):
        raise TypeError('Input has to be a Python list or NumPy 1darray, not '+type(list).__name__+'!')

    # Convert all list entries to strings to avoid any trouble below
    try:
        lst = np.array(lst,dtype=str).flatten()
    except:
        raise ValueError("Input must be a list/NumPy 1darray of strings!")

    # Do the actual sorting
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(lst, key = alphanum_key)

##########################################################################################
def get_numlines(fname):
    """
    Get number of lines of an text file

    Parameters
    ----------
    fname : str
        File to be read

    Returns
    -------
    lineno : int
        Number of lines in the file

    Notes
    -----
    This routine is based on this 
    `Stackoverflow submission <http://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python>`_

    See also
    --------
    None
    """

    # Check if input makes sense
    if not isinstance(fname,(str,unicode)):
        raise TypeError('Filename has to be a string!')
    fname = str(fname)
    if fname.find("~") == 0:
        fname = os.path.expanduser('~') + fname[1:]
    slash = fname.rfind(os.sep)
    if slash >= 0 and not os.path.isdir(fname[:fname.rfind(os.sep)]):
        raise ValueError('Invalid path to file: '+fname+'!')

    # Cycle through lines of the file and do exactly nothing
    with open(fname) as f:
        for lineno, l in enumerate(f):
            pass
    return lineno + 1

##########################################################################################
def myglob(flpath,spattern):
    """
    Return a glob-like list of paths matching a regular expression 

    Parameters
    ----------
    flpath : str
        Path to search (to search current directory use `flpath=''` or `flpath='.'`)
    spattern : str
        Pattern to search for in `flpath`

    Returns
    -------
    flist : list
        A Python list of all files found in `flpath` that match the input pattern `spattern`

    Examples
    --------
    List all png/PNG files in the folder `MyHolidayFun` found under `Documents`

    >>> myglob('Documents/MyHolidayFun','*.[Pp][Nn][Gg]')
    ['Documents/MyHolidayFun/img1.PNG','Documents/MyHolidayFun/img1.png']
        
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
def moveit(fname):
    """
    Check if a file/directory exists, if yes, rename it

    Parameters
    ----------
    fname : str
        A string specifying (the path to) the file/directory to be renamed (if existing)

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

    # Check if input makes sense
    if not isinstance(fname,(str,unicode)):
        raise TypeError("File-/Directory-name has to be a string!")
    fname = str(fname)
    if fname.find("~") == 0:
        fname = os.path.expanduser('~') + fname[1:]

    # If file already exists, rename it
    if os.path.isfile(fname):
        now     = datetime.now()
        dot     = fname.rfind('.')
        idx     = len(fname)
        if dot > 0: idx = dot
        newname = fname[:idx] + "_bak_"+\
                  str(now.year)+"_"+\
                  str(now.month)+"_"+\
                  str(now.day)+"_"+\
                  str(now.hour)+"_"+\
                  str(now.minute)+"_"+\
                  str(now.second)+\
                  fname[idx::]
        print "WARNING: File "+fname+" already exists, renaming it to: "+newname+"!"
        os.rename(fname,newname)

    # If directory already exists, rename it
    elif os.path.isdir(fname):
        now     = datetime.now()
        slash   = fname.rfind(os.sep)
        if slash == (len(fname) - 1): fname = fname[:slash]
        newname = fname + "_bak_"+\
                  str(now.year)+"_"+\
                  str(now.month)+"_"+\
                  str(now.day)+"_"+\
                  str(now.hour)+"_"+\
                  str(now.minute)+"_"+\
                  str(now.second)
        print "WARNING: Directory "+fname+" already exists, renaming it to: "+newname+"!"
        shutil.move(fname,newname)
    
##########################################################################################
def regexfind(arr,expr):
    """
    Find regular expression in a NumPy array

    Parameters
    ----------
    arr : NumPy 1darray
        Array of strings to search 
    expr : str
        Regular expression to search for in the components of `arr`

    Returns
    -------
    ind : NumPy 1darray
        Index array of elements in `arr` that contain expression `expr`. If `expr` was not found
        anywhere in `arr` an empty array is returned

    Examples
    --------
    Suppose the array `arr` is given by

    >>> arr
    array(['L_a', 'L_b', 'R_a', 'R_b'], 
      dtype='|S3')

    If we want to find all elements of `arr` starting with `l_` or `L_` we could use

    >>> regexfind(arr,"[Ll]_*")
    array([0, 1])

    See also
    --------
    None
    """

    # Sanity checks
    try:
        arr = np.array(arr)
    except:
        raise TypeError("Input must be a NumPy array/Python list, not "+type(arr).__name__+"!")
    sha = arr.squeeze().shape
    if len(sha) != 1:
        raise ValueError("Input must be a NumPy 1darray or Python list!")
    for el in arr:
        if not isinstance(el,(str,unicode)):
            raise ValueError("Every element in the input array has to be a string!")
    if not isinstance(expr,(str,unicode)):
        raise TypeError("Input expression has to be a string, not "+type(expr).__name__+"!")

    # Now do something: start by compiling the input expression
    regex = re.compile(expr)

    # Create a generalized function to find matches
    match = np.vectorize(lambda x:bool(regex.match(x)))(arr)

    # Get matching indices and return
    return np.where(match == True)[0]
