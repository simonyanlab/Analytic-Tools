# recipes.py - Here are some convenience functions that are used often
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# June 25 2013

from __future__ import division
import sys

def query_yes_no(question, default=None):
    """
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".

    Notes
    -----
    This is a recipe No. 577058 from ActiveState written by Trent Mick

    See also
    --------
    http://code.activestate.com/recipes/577058/
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

        ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']

    Calling natural_sort on that list results in 

        ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']

    Notes:
    ------
    This code was written by Mark Byers as part of a Stackoverflow submission, 
    see http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort

    See also:
    ---------
    MATLAB File Exchange submission sort_nat.m, currently available at 
    http://www.mathworks.com/matlabcentral/fileexchange/10959-sortnat-natural-order-sort
    
    Coding Horror's note on natural sorting of file lists
    http://www.codinghorror.com/blog/2007/12/sorting-for-humans-natural-sort-order.html
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
