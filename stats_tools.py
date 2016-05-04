# stats_tools.py - Tools to perform statistical tests
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# Created: December 30 2014
# Last modified: <2016-05-04 10:27:17>

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import ipdb; import sys
import h5py
import string
import itertools
from texttable import Texttable
from datetime import datetime
import os

##########################################################################################
def perm_test(X,Y,paired=None,nperms=10000,tail='two',correction=None,get_dist=False,mth=None,\
              verbose=True,fname=None,vars=None,g1str=None,g2str=None):
    """
    Perform permutation tests on paired/unpaired uni-/multi-variate data

    Parameters
    ----------
    X : NumPy 2darray
        An #samples-by-#variables array holding the data of the first group
    X : NumPy 2darray
        An #samples-by-#variables array holding the data of the second group
    paired : bool
        Switch to indicate whether the two data-sets `X` and `Y` represent paired 
        (`paired = True`) or unpaired data. 
    nperms : int
        Number of permutations for shuffling the input data
    tail : str
        The alternative hypothesis the data is tested against. If `tail = 'less'', then 
        the null is tested against the alternative that the mean of the first group is 
        less than the mean of the second group ('lower tailed'. Alternatively, 
        `tail = 'greater'` indicates the alternative that the mean of the first group is 
        greater than the mean of the second group ('upper tailed'). For `tail = 'two'` 
        the alternative hypothesis is that the means of the data are different ('two tailed'), 
    correction : str
        Correction method to adjust `p`-values for multiple comparisons. If the two groups 
        are paired (`paired = True`) this option is ignored, since `MNE`'s permutation t-test
        only supports standard `p`-value correction using the maximal test statistic `Tmax` [2]_. 
        For unpaired data (`paired = False`) the `R` library `coin` is used which supports
        'single-step' (default), 'step-down' and 'discrete' `Tmax` approaches for the maximal 
        test statistic to correct for multiple comparisons (see [1]_ for details). 
    get_dist : bool
        Switch that determines whether the sampling distribution used for testing is 
        returned (by default it is not returned). 
    mth : str
        Only supported for unpaired data. If `mth` is not specified, a general independence 
        test is performed (default), for `mth = 'maxstat'` a maximally selected statistics test
        is carried out (see [1]_ for details). 
    verbose : bool
        If `verbose = True` then intermediate results, progression messages and a table 
        holding the final statistical score are printed to the prompt. 
    fname : str
        If provided, testing results are saved to the csv file `fname`. The file-name 
        can be provided with or without the extension '.csv' 
        (WARNING: existing files will be overwritten!). By default, the output is not saved. 
    vars : list or NumPy 1darray
        Names of the variables that are tested for symmetry/independence. Only relevant 
        if `verbose = True` and/or `fname` is not `None`. If `vars = None` and output
        is shown/saved a generic list ['Variable 1','Variable 2',...] will be used 
        in the table showing the final results. 
    g1str : str
        Name of the first group. Only relevant if `verbose = True` and/or `fname` is not `None`. 
        If `g1str = None` and output is shown/saved a generic group name ('Group 1') will be used 
        in the table showing the final results. 
    g2str : str
        Name of the second group. Only relevant if `verbose = True` and/or `fname` is not `None`. 
        If `g2str = None` and output is shown/saved a generic group name ('Group 2') will be used 
        in the table showing the final results. 

    Returns
    -------
    stats_dict : dictionary
        Testing results are saved in a Python dictionary. By default `stats_dict` has 
        the keys 'pvals' (the adjusted `p`-values) and 'statvals' (values of the test statistic
        observed for all variables). If `get_dist = True` then the additional key 'dist' points to
        the used sampling distribution. 

    Notes
    -----
    This routine is merely a wrapper and does not do any heavy computational lifting.  
    For paired data the function `permutation_t_test` of the `MNE` package [2]_ is called. 
    Unpaired data are tested using the `R` library `coin` [1]_. Thus, this routine has a 
    number of dependencies: for paired data the Python package `mne` is required, 
    unpaired samples can only be tested if `pandas` as well as `rpy2` (for `R`/Python conversion)
    and, of course, `R` and the `R`-library `coin` are installed (and in the search path). 
    To show/save results the routine `printstats` (part of this module) is called.

    See also
    --------
    printstats : routine to pretty-print results computed by a symmetry/independence test
    coin : a `R` library for conditional inference procedures in a permutation test framework, 
           currently available `here <http://cran.r-project.org/web/packages/coin/index.html>`_
    mne : a software package for processing magnetoencephalography (MEG) and electroencephalography (EEG) data,
          currently available at the Python Package Index `here <https://pypi.python.org/pypi/mne/0.7.1>`_

    Examples
    --------
    Assume we want to analyze medical data of 200 healthy adult subjects collected before and after 
    physical exercise. For each subject, we have measurements of heart-rate (HR), blood pressure (BP) and
    body temperature (BT) before and after exercise. Thus our data sets contain 200 observations of 
    3 variables. We want to test the data for a statistically significant difference in any of the 
    three observed quantities (HR, BP, BT) after physical exercise compared to the measurements 
    acquired before exercise. 

    Assume all samples are given as Python lists: `HR_before`, `BP_before`, `BT_before`, 
    `HR_after`, `BP_after`, `BT_after`. To be able to use `perm_test`, we collect the 
    data in NumPy arrays:

    >>> import numpy as np
    >>> X = np.zeros((200,3))
    >>> X[:,0] = HR_before
    >>> X[:,1] = BP_before
    >>> X[:,2] = BT_before
    >>> Y = np.zeros((200,3))
    >>> Y[:,0] = HR_after
    >>> Y[:,1] = BP_after
    >>> Y[:,2] = BT_after

    Our null-hypothesis is that physical exercise did not induce a significant change in 
    any of the observed variables. As an alternative hypothesis, we assume that 
    exercise induced an increase in heart rate, blood pressure and body temperature. 
    To test our hypotheses we use the following command

    >>> perm_test(X,Y,paired=True,nperms=20000,tail='less',fname='stats.csv',\
                  vars=['Heart Rate','Blood Pressure','Body Temperature'],g1str='Before Exercise',g2str='After Exercise')

    which performs a lower-tailed paired permutation t-test with 20000 permutations, 
    prints the results to the prompt and also saves them in the file `stats.csv`. 
    
    References
    ----------
    .. [1] T. Hothorn, K. Hornik, M. A. van de Wiel, A. Zeileis. A Lego System for Conditional Inference. 
           The American Statistician 60(3), 257-263, 2014. 
    .. [2] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, L. Parkkonen, 
           M. Haemaelaeinen. MNE software for processing MEG and EEG data. NeuroImage 86, 446-460, 2014
    """

    # Check mandatory inputs and make sure X and Y are tested for the same no. of variables
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
        raise ValueError('Cannot perform paired symmetry test for different number of samples!')

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

    # Just in case: save tail selection for output before we convert it (maybe) to an integer 
    tail_dt1 = {"less":"less than","two":"different from","greater":"greater than"}
    tail_dt2 = {"less":"lower","two":"two","greater":"upper"}
    tail_st1 = tail_dt1[tail]
    tail_st2 = tail_dt2[tail]

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
            print "WARNING: The stats toolbox in MNE only supports standard Tmax correction of p-values!"
    else:
        correction = 'single-step'

    # Check if the user wants to see what's going on
    msg = 'The switch verbose has to be True or False!'
    try:
        bad = (verbose == True or verbose == False)
    except: raise TypeError(msg)
    if bad == False:
        raise TypeError(msg)
    
    # If a file-name was provided make sure it's a string and check if the path exists
    # (unicode chars in file-names are probably a bad idea...)
    if fname != None:
        if str(fname) != fname:
            raise TypeError('Optional output file-name has to be a string!')
        fname = str(fname)
        if fname.find("~") == 0:
            fname = os.path.expanduser('~') + fname[1:]
        slash = fname.rfind(os.sep)
        if slash >= 0 and not os.path.isdir(fname[:fname.rfind(os.sep)]):
            raise ValueError('Invalid path for output file: '+fname+'!')

    # Warn if output was turned off but labels were provided and assign default values to labels if necessary
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
        print "\nTesting statistical symmetry of paired samples using the permutation t-test from mne"

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

            # Set up our R name-spaces and see if coin is available
            R    = rpy2.robjects.r
            coin = importr('coin')

        except:
            msg = "Either the Python modules 'pandas' and/or 'rpy2' or "+\
                  "the R package 'coin' is/are not installed!"
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

        # Build an R dataframe by stacking X and Y on top of each other, with columns labeled by abclist
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

        # Construct string to be used as footer for the output file/last line of command line output
        permstr = "using "+str(nperms)+" permutations under the alternative hypothesis that "+\
                  g1str+" is "+tail_st1+" "+g2str+" ("+tail_st2+"-tailed) "
        if paired:
            ft = "Statistical symmetry of paired samples was assessed using the permutation t-test "+\
                 "from the Python package MNE (see http://martinos.org/mne/stable/mne-python.html)\n"+\
                 permstr+"\n"+\
                 "adjusted for multiple comparisons using the maximal test statistic Tmax. "
        else:
            if mth == None:
                mth_str = "general independence test"
            else:
                mth_str = "maximally selected statistics test"
            mth_dt  = {}
            ft = "Statistical independence of unpaired samples was assessed using the "+mth_str+\
                 " from the R library coin (http://cran.r-project.org/web/packages/coin/index.html)\n"+\
                 permstr+"\n"+\
                 "adjusted for multiple testing by a "+correction+" Tmax approach for the maximal test statistic. \n"

        # Append an auto-gen message and add current date/time to the soon-to-be footer
        ft += "Results were computed by stats_tools.py on "+str(datetime.now())

        # Call printstats to do the heavy lifting
        printstats(vars,stats_dict['pvals'],X,Y,g1str,g2str,foot=ft,verbose=verbose,fname=fname)

    # Return the stocked dictionary
    return stats_dict

##########################################################################################
def printstats(variables,pvals,group1,group2,g1str='group1',g2str='group2',foot='',verbose=True,fname=None):
    """
    Pretty-print previously computed statistical results 

    Parameters
    ----------
    variables : list or NumPy 1darray
        Python list/NumPy array of strings representing variables that have been tested
    pvals : Numpy 1darray
        Aray of `p`-values (floats) of the same size as `variables`
    group1 : NumPy 2darray
        An #samples-by-#variables array holding the data of the first group used in the previously
        performed statistical comparison
    group2 : NumPy 2darray
        An #samples-by-#variables array holding the data of the second group used in the previously
        performed statistical comparison
    g1str : string
        Name of the first group that will be used in the generated table
    g2str : string
        Name of the first group that will be used in the generated table
    fname : string
        Name of a csv-file (with or without extension '.csv') used to save the table 
        (WARNING: existing files will be overwritten!). Can also be a path + file-name 
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
                `nws_tools.py <http://research.mssm.edu/simonyanlab/analytical-tools/nws_tools.printdata.html#nws_tools.printdata>`_)
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
        raise TypeError('Data-set 1 must be a NumPy 2d array, not '+type(group1).__name__+'!')
    if M != m:
        raise ValueError('No. of variables (=labels) and dimension of group1 do not match up!')
    try:
        N,M = group2.shape
    except: 
        raise TypeError('Data-set 2 must be a NumPy 2d array, not '+type(group2).__name__+'!')
    if M != m:
        raise ValueError('No. of variables (=labels) and dimension of group2 do not match up!')

    # If column labels were provided, make sure they are printable strings
    if str(g1str) != g1str:
        raise TypeError('The optional column label `g1str` has to be a string!')
    if str(g2str) != g2str:
        raise TypeError('The optional column label `g2str` has to be a string!')

    # If a footer was provided, make sure it is a printable string
    if str(foot) != foot:
        raise TypeError('The optional footer `foot` has to be a string!')

    # See if we're supposed to print stuff to the terminal or just save everything to a csv file
    msg = 'The optional switch verbose has to be True or False!'
    try:
        bad = (verbose == True or verbose == False)
    except: raise TypeError(msg)
    if bad == False:
        raise TypeError(msg)

    # If a file-name was provided make sure it's a string and check if the path exists
    # (unicode chars in file-names are probably a bad idea...)
    if fname != None:
        if str(fname) != fname:
            raise TypeError('Input fname has to be a string specifying an output file-name, not '\
                            +type(fname).__name__+'!')
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
        print foot + "\n"

    # If wanted, save stuff in a csv file
    if save:
        head = str(head)
        head = head.replace("[","")
        head = head.replace("]","")
        head = head.replace("'","")
        np.savetxt(fname,Data,delimiter=",",fmt="%s",header=head,footer=foot,comments="")
