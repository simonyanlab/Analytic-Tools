# recipes.py - Here are some convenience functions that are used often
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# June 25 2013

from __future__ import division
import sys
import re
from numpy.linalg import norm

##########################################################################################
def query_yes_no(question, default=None):
    """
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".

    Notes:
    ------
    This is recipe no. 577058 from ActiveState written by Trent Mick

    See also:
    ---------
    .. http://code.activestate.com/recipes/577058/
    """
    valid = {"yes":True,   "y":True,  "ye":True,
             "no":False,     "n":False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

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
def natural_sort(l): 
    """
    Sort a Python list l in a "natural" way

    From the documentation of sort_nat.m:

    "Natural order sorting sorts strings containing digits in a way such that
    the numerical value of the digits is taken into account.  It is
    especially useful for sorting file names containing index numbers with
    different numbers of digits.  Often, people will use leading zeros to get
    the right sort order, but with this function you don't have to do that."

    For instance, a usual glob will give you a file listing sorted in this way 

        ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']

    Calling natural_sort on that list results in 

        ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']

    Inputs:
    -------
    l : Python list 
        Python list of strings

    Returns:
    --------
    l_sort : Python list
        Lexicographically sorted version of the input list l

    Notes:
    ------
    This function does *not* do any error checking and assumes you know what you are doing!
    The code below was written by Mark Byers as part of a Stackoverflow submission, see
    .. http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort

    See also:
    ---------
    MATLAB File Exchange submission sort_nat.m, currently available at 
    .. http://www.mathworks.com/matlabcentral/fileexchange/10959-sortnat-natural-order-sort
    
    Coding Horror's note on natural sorting of file listings
    .. http://www.codinghorror.com/blog/2007/12/sorting-for-humans-natural-sort-order.html
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

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
