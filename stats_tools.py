# stats_tools.py - Tools to perform statistical tests
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# Created: December 30 2014
# Last modified: 2015-01-29 11:48:27

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import ipdb; import sys
import h5py
import string
import itertools
from texttable import Texttable
import os

##########################################################################################
def perm_test(X,Y,paired=True,nperms=10000,tail='two',correction=None,get_dist=False,mth=None,\
              verbose=True,fname=None,vars=None,g1str=None,g2str=None):

    # Check manadatory inputs and make sure X and Y are tested for the same no. of variables
    try:
        [nsamples_x,n_testsx] = X.shape
    except: raise TypeError('First input X has to be a NumPy 2darray!')
    try:
        [nsamples_y,n_testsy] = Y.shape
    except: raise TypeError('First input Y has to be a NumPy 2darray!')

    if n_testsx != n_testsy:
        raise ValueError('Number of variables different in X and Y!')
    n_tests = n_testsx

    for arr in [X,Y]:
        if np.isinf(arr).max() == True or np.isnan(arr).max() == True or np.isreal(arr).min() == False:
            raise ValueError('Inputs X and Y must be real NumPy 2darrays without Infs or NaNs!')

    if np.min([nsamples_x,nsamples_y]) < n_tests:
        print "WARNING: Number of variables > number of samples!"

    # Check paired
    msg = 'The switch paired has to be True or False!'
    try:
        bad = (paired == True or paired == False)
    except: raise TypeError(msg)
    if bad == False:
        raise TypeError(msg)

    if nsamples_x != nsamples_y and paired == True:
        raise ValueError('Cannot perform paired independence test for different number of samples!')

    # Check get_dist
    msg = 'The switch get_dist has to be True or False!'
    try:
        bad = (get_dist == True or get_dist == False)
    except: raise TypeError(msg)
    if bad == False:
        raise TypeError(msg)

    # Check nperms
    try:
        bad = (int(nperms) != nperms)
    except: raise TypeError("The number of permutation nperms has to be a positive integer!")
    if bad or nperms <= 0:
        raise ValueError('The number of permutations has to be a positive integer!')

    # Check mth string if provided
    if mth != None:
        if paired:
            print "WARNING: The optional mth argument is only supported "+\
                  "when testing unpaired samples with  R!"
        else:
            supported = ['maxstat']
            if supported.count(mth) == 0:
                sp_str = str(supported)
                sp_str = sp_str.replace('[','')
                sp_str = sp_str.replace(']','')
                msg = 'Unpaired samples cannot be tested with '+str(mth)+\
                      '. Available options are: '+sp_str
                raise ValueError(msg)

    # Check tail if provided
    supported = {'greater':1,'less':-1,'two':0}
    spl       = supported.keys()
    if spl.count(tail) == 0:
        sp_str = str(spl)
        sp_str = sp_str.replace('[','')
        sp_str = sp_str.replace(']','')
        msg = "The alternative hypothesis given by tail = '"+str(tail)+ "' is invalid. "+\
              "Available options are: "+sp_str
        raise ValueError(msg)

    # Depending on `paired`, `tail` is either a str (for coin in R) or -1,0,1 (for mne's t-test)
    if paired == False:
        if tail == 'two':
            tail = 'two.sided'
    else:
        tail = supported[tail]

    # Check the setting for the p-value correction
    if correction != None:
        if paired == False:
            supported = ['step-down','single-step','discrete']
            if supported.count(correction) == 0:
                sp_str = str(supported)
                sp_str = sp_str.replace('[','')
                sp_str = sp_str.replace(']','')
                msg = "The p-value correction method given by correction = '"+str(correction)+\
                      "' is invalid. Available options are: "+sp_str
                raise ValueError(msg)
        else:
            print "WARNING: The stats toolbox in MNE only supports standard Tmax correction of p=values!"
    else:
        correction = 'single-step'

    # Check if the user wants to see what's going on
    msg = 'The switch verbose has to be True or False!'
    try:
        bad = (verbose == True or verbose == False)
    except: raise TypeError(msg)
    if bad == False:
        raise TypeError(msg)
    
    # If a filename was provided make sure it's a string and check if the path exists
    # (unicode chars in filenames are probably a bad idea...)
    if fname != None:
        if str(fname) != fname:
            raise TypeError('Optional output filename has to be a string!')
        fname = str(fname)
        if fname.find("~") == 0:
            fname = os.path.expanduser('~') + fname[1:]
        if not os.path.isdir(fname[:fname.rfind(os.sep)]):
            raise ValueError('Invalid path for output file: '+fname+'!')

    # Warn if output was turn off but labels were provided and assign default values to labels if necessary
    # (Do error checking here to avoid a breakdown at the very end of the code...)
    if verbose == False and fname == None:
        for chk in [vars,g1str,g2str]:
            if chk != None:
                print "WARNING: Output labels were provided but verbose == False and fname == None. "+\
                      "The labels will be ignored and no output will be shown/saved!"
                break
    else:
        if vars == None:
            vars = ['Variable '+str(v) for v in range(1,n_tests+1)]
        else:
            try:
                m = len(vars)
            except: 
                raise TypeError('Input vars must be a Python list or NumPy 1d array of strings, not '+\
                                type(vars).__name__+'!')
            if m != n_tests:
                raise ValueError('Number of variable labels for output and number of tests do not match up!')
            for var in vars:
                if str(var) != var:
                    raise TypeError('All variables in the optional input vars must be strings!')
        
        if g1str == None:
            g1str = 'Group 1'
        else:
            if str(g1str) != g1str:
                raise TypeError('The optional column label `g1str` has to be a string!')
        if g2str == None:
            g2str = 'Group 2'
        else:
            if str(g2str) != g2str:
                raise TypeError('The optional column label `g2str` has to be a string!')

    # Initialize the output dictionary
    stats_dict = {}
            
    # Here we go: in case of paired samples, use Python's mne 
    if paired:

        # Try to import/load everything we need below
        try:
            import mne
        except:
            raise ImportError("The Python module 'mne' is not installed!")

        # Just to double check with user, say what's about to happen
        print "\nTesting statistical symmetry of paired samples using the permuation t-test from mne"

        # Use mne's permutation t-test to check for statistical symmetry of paired samples
        statvals, pvals, dist = mne.stats.permutation_t_test(X-Y,n_permutations=nperms,\
                                                             tail=tail,n_jobs=1,verbose=False)

        # Store result in output dictionary
        stats_dict['pvals']    = pvals
        stats_dict['statvals'] = statvals
        if get_dist:
            stats_dict['dist'] = dist

    # For unpaired samples fire up the coin package to do some stats
    else:

        # Try to import/load everything we need below
        try:
            import pandas as pd
            import pandas.rpy.common as cm 
            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()
            from rpy2.robjects.packages import importr
            from rpy2.robjects import Formula

            # Set up our R namespaces and see if coin is available
            R    = rpy2.robjects.r
            coin = importr('coin')

        except:
            msg = "Either the Python modules 'pandas' and/or 'rpy2' or "+\
                  "the R pacakge 'coin' is/are not installed!"
            raise ImportError(msg)
 
        # Construct a list of strings of the form 
        # ['a','b','c',...,'z','aa','ab','ac',...,'az','ba','bb','bc',...]
        abclist = (list(string.lowercase) + \
                  [''.join(x) for x in itertools.product(string.lowercase, repeat=2)])[:n_tests] + ['group']
        
        # Use that list to build a string of the form 'a + b + c +...+ aa + ab + ... ~ group' 
        # To understand the syntax, see the Examples section of help(mercuryfish) in R
        frm_str = abclist[0]
        for ltr in abclist[1:-1]:
            frm_str += ' + ' + ltr
        frm_str += ' ~ group'
        
        # Construct an array that will be our factor in the R dataframe below: 
        # all rows of X get the factor 1, the rest is 2
        group = 2*np.ones((nsamples_x + nsamples_y,1))
        group[:nsamples_x] = 1

        # Build an R dataframe by stacking X and Y on top of each other, with columsn labeled by abclist
        r_dframe = cm.convert_to_r_dataframe(pd.DataFrame(np.hstack([np.vstack([X,Y]),group]),\
                                                          columns=abclist))

        # Convert the string to an R formula
        r_frm = Formula(frm_str)

        # Just to double check with user, say what's about to happen
        print "\nTesting statistical independence of unpaired samples in R"

        # Depending on the mth string, perform an independence test in R
        if mth == None:
            print "Performing General Independence Test..."
            result = R.independence_test(r_frm,data=r_dframe,alternative=tail,\
                                         distribution=coin.approximate(B=nperms))
        elif mth == 'maxstat':
            print "Performing Maximally Selected Statistics Test..."
            result = R.maxstat_test(r_frm,data=r_dframe,alternative=tail,\
                                    distribution=coin.approximate(B=nperms))

        # The outputs of the pvalue method is converted to a list, use np.array to get a NumPy array
        stats_dict['pvals']    = np.array(R.pvalue(result,method=correction)).squeeze()

        # The output of statistic is converted to a pandas dataframe, 
        # use the dataframe's values method to get NumPy arrays
        stats_dict['statvals'] = cm.convert_robj(R.statistic(result,type='linear')).values.squeeze()

        # R FloatVectors (result of pperm)  are converted to lists, so use np.array to get a NumPy array
        if get_dist:
            print "Getting sampling distribution..."
            stats_dict['dist'] = np.array(cm.convert_robj(R.pperm(result,R.support(result)))).squeeze()

        print "Done"

    # If wanted print/save the results
    if verbose or fname != None:
        printstats(vars,stats_dict['pvals'],X,Y,g1str,g2str,verbose=verbose,fname=fname)

    # Return the stocked dictionary
    return stats_dict

##########################################################################################
def printstats(variables,pvals,group1,group2,g1str='group1',g2str='group2',verbose=True,fname=None):
    """
    Pretty-print previously computed statistical results 

    Parameters
    ----------
    variables : list or NumPy 1darray
        Python list/NumPy array of strings representing variables that have been tested
    pvals : Numpy 1darray
        Aray of `p`-values (floats) of the same size as `variables`
    group1 : NumPy 2darray
        An #samples-by-#variables array forming the "group1" set for the previously 
        computed statistical comparison
    group2 : NumPy 2darray
        An #samples-by-#variables array forming the "group2" set for the previously 
        computed statistical comparison
    g1str : string
        A string to be used in the generated table to highlight computed mean/std values of 
        the group1 dataset
    g2str : string
        A string to be used in the generated table to highlight computed mean/std values of 
        the group2 dataset
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
    printdata : a function that pretty-prints/-saves data given in an array (part of 
                `nws_tools.py <http://research.mssm.edu/simonyanlab/analytical-tools/nws_tools.printdata.html#nws_tools.printdata>`_
    """

    # Make sure that the groups, p-values and tested variables have appropriate dimensions
    try:
        m = len(variables)
    except: 
        raise TypeError('Input variables must be a Python list or NumPy 1d array of strings, not '+\
                        type(variables).__name__+'!')
    for var in variables:
        if str(var) != var:
            raise TypeError('All variables must be strings!')
    try: 
        M = pvals.size
    except: 
        raise TypeError('The p-values must be provided as NumPy 1d array, not '+type(variables).__name__+'!')
    if M != m:
        raise ValueError('No. of variables (=labels) and p-values do not match up!')
    try:
        N,M = group1.shape
    except: 
        raise TypeError('Dataset 1 must be a NumPy 2d array, not '+type(group1).__name__+'!')
    if M != m:
        raise ValueError('No. of variables (=labels) and dimension of group1 do not match up!')
    try:
        N,M = group2.shape
    except: 
        raise TypeError('Dataset 2 must be a NumPy 2d array, not '+type(group2).__name__+'!')
    if M != m:
        raise ValueError('No. of variables (=labels) and dimension of group2 do not match up!')

    # If column labels were provided, make sure they are printable strings
    if str(g1str) != g1str:
        raise TypeError('The optional column label `g1str` has to be a string!')
    if str(g2str) != g2str:
        raise TypeError('The optional column label `g2str` has to be a string!')

    # See if we're supposed to print stuff to the terminal or just save everything to a csv file
    msg = 'The optional switch verbose has to be True or False!'
    try:
        bad = (verbose == True or verbose == False)
    except: raise TypeError(msg)
    if bad == False:
        raise TypeError(msg)

    # If a filename was provided make sure it's a string and check if the path exists
    # (unicode chars in filenames are probably a bad idea...)
    if fname != None:
        if str(fname) != fname:
            raise TypeError('Input fname has to be a string specifying an output filename, not '\
                            +type(fname).__name__+'!')
        fname = str(fname)
        if fname.find("~") == 0:
            fname = os.path.expanduser('~') + fname[1:]
        if not os.path.isdir(fname[:fname.rfind(os.sep)]):
            raise ValueError('Invalid path for output file: '+fname+'!')
        if fname[-4::] != '.csv':
            fname = fname + '.csv'
        save = True
    else: save = False

    # Construct table head
    head = [" ","p","mean("+g1str+")"," ","std("+g1str+")","</>",\
            "mean("+g2str+")"," ","std("+g2str+")"]

    # Compute mean/std of input data
    g1mean = group1.mean(axis=0)
    g1std  = group1.std(axis=0)
    g2mean = group2.mean(axis=0)
    g2std  = group2.std(axis=0)

    # Put "<" if mean(base) < mean(test) and vice versa
    gtlt = np.array(['<']*g1mean.size)
    gtlt[np.where(g1mean > g2mean)] = '>'

    # Prettify table
    pmstr = ["+/-"]*g1mean.size

    # Assemble data array
    Data = np.column_stack((variables,\
                            pvals.astype('str'),\
                            g1mean.astype('str'),\
                            pmstr,\
                            g1std.astype('str'),\
                            gtlt,\
                            g2mean.astype('str'),\
                            pmstr,\
                            g2std.astype('str')))

    # Construct texttable object
    table = Texttable()
    table.set_cols_align(["l","l","r","c","l","c","r","c","l"])
    table.set_cols_valign(["c"]*9)
    table.set_cols_dtype(["t"]*9)
    table.set_cols_width([12,18,18,3,18,3,18,3,18])
    table.add_rows([head],header=True)
    table.add_rows(Data.tolist(),header=False)
    table.set_deco(Texttable.HEADER)

    # Pump out table if wanted
    if verbose:
        print "Summary of statistics:\n"
        print table.draw() + "\n"

    # If wanted, save stuff in a csv file
    if save:
        head = str(head)
        head = head.replace("[","")
        head = head.replace("]","")
        head = head.replace("'","")
        np.savetxt(fname,Data,delimiter=",",fmt="%s",header=head,comments="")

################################################################################
# TESTING
################################################################################

# # Load data
# f     = h5py.File('../Python_tools/stat_data.h5','r')
# X     = f['rest'].value
# Y     = f['seiz'].value
# p_ref = f['pval'].value
# H_ref = f['H0'].value
# t_ref = f['tval'].value
# f.close()

# # Settings that were used to compute the stats
# n_perms = 10000                     # No. of permutations to use
# tail    = 0                         # Two-tailed test
# n_jobs  = 1                         # How many CPUs to use (don't use more than 1!!!)
# verbose = True                      # Be chatty about it

# Paired test
# perm_test(X,Y,paired=True,tail='two')
# sd = perm_test(X,Y,paired=False,tail='two')
# print sd['pvals']
# sd = perm_test(X,Y[:-12],paired=False,tail='two')
# print sd['pvals']
# sd = perm_test(X,Y,paired=True,tail='two',verbose=False,fname='~/Desktop/test.csv')
# print sd['pvals']

# sys.exit()

# supported = ['maxstat']
# # supported = ['wilcox','normal','median','ansari','kruskal',\
# #              'fligner','spearman','oneway','maxstat','chisq','surv','cmh']
# for mth in supported:
#     perm_test(X,Y,paired=False,tail='two',mth=mth)
