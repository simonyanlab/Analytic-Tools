# recipes.py - Here are some convenience functions that are used often
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# June 25 2013

from __future__ import division
import sys
import re
import fnmatch
import os
from numpy.linalg import norm
import numpy as np
from texttable import Texttable
from datetime import datetime, timedelta

##########################################################################################
def find_contiguous_regions(condition):
    """
    Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    See http://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
    """
    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

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
def printstats(variables,pvals,baseline,testset,basestr='baseline',teststr='testset',fname=None):
    """
    Pretty-print previously computed statistical results 

    Parameters
    ----------
    variables : list or NumPy 1darray
        List/array of variables that have been tested
    pvals : Numpy 1darray
        Aray of p-values (has to be same size as `variables`)
    baseline : NumPy 2darray
        An #samples-by-#variables array forming the "baseline" set for the previously 
        computed statistical comparison
    testset : NumPy 2darray
        An #samples-by-#variables array forming the "test" set for the previously 
        computed statistical comparison
    basestr : string
        A string to be used in the generated table to highlight computed mean/std values of 
        the baseline dataset
    teststr : string
        A string to be used in the generated table to highlight computed mean/std values of 
        the test dataset
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

    See also
    --------
    texttable : a module for creating simple ASCII tables (currently available at the 
                `Python Package Index <https://pypi.python.org/pypi/texttable/0.8.1>`_)
    printdata : a function that pretty-prints/-saves data given in an array
    """

    # Sanity checks
    try:
        m = len(variables)
    except: 
        raise TypeError('Input must be a Python list or NumPy 1d array, not '+type(variables).__name__+'!')
    try: 
        M = pvals.size
    except: 
        raise TypeError('Input must be a NumPy 1d array, not '+type(variables).__name__+'!')
    if M != m:
        raise ValueError('No. of labels and p-values do not match up!')

    try:
        N,M = baseline.shape
    except: 
        raise TypeError('Input must be a NumPy 2d array, not '+type(baseline).__name__+'!')
    if M != m:
        raise ValueError('No. of labels and baseline dimension do not match up!')

    try:
        N,M = testset.shape
    except: 
        raise TypeError('Input must be a NumPy 2d array, not '+type(testset).__name__+'!')
    if M != m:
        raise ValueError('No. of labels and testset dimension do not match up!')

    if type(basestr).__name__ != 'str':
        raise TypeError('Input basestr to be a string, not '+type(basestr).__name__+'!')

    if type(teststr).__name__ != 'str':
        raise TypeError('Input teststr to be a string, not '+type(teststr).__name__+'!')

    if fname != None:
        if type(fname).__name__ != 'str':
            raise TypeError('Input fname has to be a string specifying an output filename, not '\
                            +type(fname).__name__+'!')
        if fname[-4::] != '.csv':
            fname = fname + '.csv'
        save = True
    else: save = False

    # Construct table head
    head = [" ","p","mean("+basestr+")"," ","std("+basestr+")","</>",\
            "mean("+teststr+")"," ","std("+teststr+")"]

    # Compute mean/std of input data
    basemean = baseline.mean(axis=0)
    basestd  = baseline.std(axis=0)
    testmean = testset.mean(axis=0)
    teststd  = testset.std(axis=0)

    # Put "<" if mean(base) < mean(test) and vice versa
    gtlt = np.array(['<']*basemean.size)
    gtlt[np.where(basemean > testmean)] = '>'

    # Prettify table
    pmstr = ["+/-"]*basemean.size

    # Assemble data array
    Data = np.column_stack((variables,\
                            pvals.astype('str'),\
                            basemean.astype('str'),\
                            pmstr,\
                            basestd.astype('str'),\
                            gtlt,\
                            testmean.astype('str'),\
                            pmstr,\
                            teststd.astype('str')))

    # Construct texttable object
    table = Texttable()
    table.set_cols_align(["l","l","r","c","l","c","r","c","l"])
    table.set_cols_valign(["c"]*9)
    table.set_cols_dtype(["t"]*9)
    table.set_cols_width([12,18,18,3,18,3,18,3,18])
    table.add_rows([head],header=True)
    table.add_rows(Data.tolist(),header=False)
    table.set_deco(Texttable.HEADER)

    # Pump out table
    print "Summary of statistics:\n"
    print table.draw() + "\n"

    # If wanted, save stuff in a csv file
    if save:
        head = str(head)
        head = head.replace("[","")
        head = head.replace("]","")
        head = head.replace("'","")
        np.savetxt(fname,Data,delimiter=",",fmt="%s",header=head,comments="")

##########################################################################################
def moveit(fname):
    """
    Check if a file exists, if yes, rename it

    Parameters
    ----------
    fname : str
        A string specifying (the path to) the file to be renamed (if existing)

    Returns
    -------
    Nothing : None

    See also
    --------
    None
    """

    # Check if input makes sense
    if type(fname).__name__ != "str":
        raise TypeError("Filename has to be a string!")

    # If file already exists, rename it
    if os.path.isfile(fname):
        newname = fname[:-3] + "_bak_"+\
                  str(datetime.now().year)+"_"+\
                  str(datetime.now().month)+"_"+\
                  str(datetime.now().day)+"_"+\
                  str(datetime.now().hour)+"_"+\
                  str(datetime.now().minute)+"_"+\
                  str(datetime.now().second)+\
                  fname[-3::]
        print "WARNING: file "+fname+" already exists, renaming it to: "+newname+"!"
        os.rename(fname,newname)

