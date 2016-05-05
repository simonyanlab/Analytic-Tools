# stats_tools.py - Tools to perform statistical tests
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# Created: December 30 2014
# Last modified: <2016-05-05 10:18:15>

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
def perm_test(X,Y,paired=None,useR=False,nperms=10000,tail='two',correction="maxT",get_dist=False,mth="t",\
              verbose=True,fname=None,vars=None,g1str=None,g2str=None):
    """
    Perform permutation tests for paired/unpaired uni-/multi-variate two-sample problems

    Parameters
    ----------
    X : NumPy 2darray
        An #samples-by-#variables array holding the data of the first group
    X : NumPy 2darray
        An #samples-by-#variables array holding the data of the second group
    paired : bool
        Switch to indicate whether the two data-sets `X` and `Y` represent paired 
        (`paired = True`) or unpaired data. 
    useR : bool
        Switch that determines whether the `R` library `flip` is used for testing. 
        Note: unpaired data can only be tested in `R`!
    nperms : int
        Number of permutations for shuffling the input data
    tail : str
        The alternative hypothesis the data is tested against. If `tail = 'less'`, then 
        the null is tested against the alternative that the mean of the first group is 
        less than the mean of the second group ('lower tailed'). Alternatively, 
        `tail = 'greater'` indicates the alternative that the mean of the first group is 
        greater than the mean of the second group ('upper tailed'). For `tail = 'two'` 
        the alternative hypothesis is that the means of the data are different ('two tailed'), 
    correction : str
        Multiplicity correction method. If the `R` package `flip` is not used for testing (`useR = False`)
        this option is ignored, since `MNE`'s permutation t-test only supports `p`-value correction using 
        the maximal test statistic `Tmax` [2]_. 
        Otherwise (either if `paired = False` or `useR = True`) the `R` library `flip` is used which 
        supports the options
        "holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none", "Fisher", "Liptak", 
        "Tippett", "MahalanobisT", "MahalanobisP", "minP", "maxT", "maxTstd", "sumT", "Direct", 
        "sumTstd", "sumT2" (see [1]_ for a detailed explanaition). By default "maxT" 
        is used. 
    get_dist : bool
        Switch that determines whether the sampling distribution used for testing is 
        returned (by default it is not returned). 
    mth : str
        Only relevant if testing is done in `R` (`useR = True` or `paired = False`). If `mth` is 
        not specified a permutation t-test will be performed. Availalbe (but completely untested!)
        options are: "t", "F", "ANOVA","Kruskal-Wallis", "kruskal", "Mann-Whitney", "sum", 
        "Wilcoxon", "rank", "Sign" (see [1]_ for details). Note that by design this wrapper only 
        supports two-sample problems (`X` and `Y`). To analyze `k`-sample data using, e.g., an ANOVA,
        please refer to the `flip` package directly. 
    verbose : bool
        If `verbose = True` then intermediate results, progression messages and a table 
        holding the final statistical evaluation are printed to the prompt. 
    fname : str
        If provided, testing results are saved to the csv file `fname`. The file-name 
        can be provided with or without the extension '.csv' 
        (WARNING: existing files will be overwritten!). By default, the output is not saved. 
    vars : list or NumPy 1darray
        Names of the variables that are being tested. Only relevant 
        if `verbose = True` and/or `fname` is not `None`. If `vars` is `None` and output
        is shown and/or saved, a generic list `['Variable 1','Variable 2',...]` will be used 
        in the table summarizing the final results. 
    g1str : str
        Name of the first sample. Only relevant if `verbose = True` and/or `fname` is not `None`. 
        If `g1str = None` and output is shown/saved a generic group name ('Group 1') will be used 
        in the table showing the final results. 
    g2str : str
        Name of the second sample. Only relevant if `verbose = True` and/or `fname` is not `None`. 
        If `g2str = None` and output is shown/saved a generic group name ('Group 2') will be used 
        in the table showing the final results. 

    Returns
    -------
    stats_dict : dictionary
        Test-results are saved in a Python dictionary. By default `stats_dict` has 
        the keys 'pvals' (the adjusted `p`-values) and 'statvals' (values of the test statistic
        observed for all variables). If `get_dist = True` then an additional entry 'dist' 
        is created for the employed sampling distribution. 

    Notes
    -----
    This routine is merely a wrapper and does not do any heavy computational lifting.  
    In case of paired data and `useR = False` the function `permutation_t_test` of 
    the `MNE` package [2]_ is called. 
    If the samples are independent (`paired = False`) or `useR = True` the `R` 
    library `flip` [1]_ is loaded. Thus, this routine has a 
    number of dependencies: for paired data at least the Python package `mne` is required, 
    unpaired samples can only be tested if `pandas` as well as `rpy2` (for `R`/Python conversion)
    and, of course, `R` and the `R`-library `flip` are installed (and in the search path). 
    To show/save results the routine `printstats` (part of this module) is called.

    See also
    --------
    printstats : routine to pretty-print results computed by a hypothesis test
    flip : a `R` library for univariate and multivariate permutation (and rotation) tests,
           currently available `here <https://cran.r-project.org/web/packages/flip/index.html>`_
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
                  vars=['Heart Rate','Blood Pressure','Body Temperature'],\
                  g1str='Before Exercise',g2str='After Exercise')

    which performs a lower-tailed paired permutation t-test with 20000 permutations, 
    prints the results to the prompt and also saves them in the file `stats.csv`. 

    References
    ----------
    .. [1] F. Pesarin. Multivariate Permutation Tests with Applications in Biostatistics.
           Wiley, New York, 2001. 
    .. [2] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, L. Parkkonen, 
           M. Haemaelaeinen. MNE software for processing MEG and EEG data. NeuroImage 86, 446-460, 2014
    """

    # Check mandatory inputs and make sure `X` and `Y` are tested for the same no. of variables
    try:
        [nsamples_x,n_testsx] = X.shape
    except: raise TypeError('First input `X` has to be a NumPy 2darray!')
    try:
        [nsamples_y,n_testsy] = Y.shape
    except: raise TypeError('First input `Y` has to be a NumPy 2darray!')

    if n_testsx != n_testsy:
        raise ValueError('Number of variables different in `X` and `Y`!')
    n_tests = n_testsx

    for arr in [X,Y]:
        if not plt.is_numlike(arr):
            raise ValueError('Inputs `X` and `Y` must be real-valued NumPy 2darrays')
        if np.isfinite(arr).min() == False:
            raise ValueError('Inputs `X` and `Y` must be real-valued NumPy 2darrays without Infs or NaNs!')

    if np.min([nsamples_x,nsamples_y]) < n_tests:
        print "WARNING: Number of variables > number of samples!"

    # Check `paired` and make sure that input arrays make sense in case we have paired data
    if not isinstance(paired,bool):
        raise TypeError("The switch `paired` has to be Boolean!")
    if nsamples_x != nsamples_y and paired == True:
        raise ValueError('Cannot perform paired test with different number of samples!')
    pairlst = ["unpaired","paired"]

    # Check `useR`
    if not isinstance(useR,bool):
        raise TypeError("The switch `useR` has to be Boolean!")
    if not paired:
        useR = True

    # Check `get_dist`
    if not isinstance(get_dist,bool):
        raise TypeError("The switch `get_dist` has to be Boolean!")

    # Check `nperms`
    if not np.isscalar(nperms) or not plt.is_numlike(nperms):
        raise TypeError("The number of permutations has to be provided as scalar!")
    if not np.isfinite(nperms):
        raise TypeError("The number of permutations must be finite!")
    if (round(nperms) != nperms):
        raise ValueError("The number of permutations must be an integer!")

    # Check `mth` 
    if not isinstance(mth,(str,unicode)):
        raise TypeError("The test-statisic has to be specified using a string, not "+type(mth).__name__+"!")
    if useR:
        msg = ''
        if paired:
            supported = ["t", "Wilcoxon", "rank", "Sign","sum"]
            if mth not in supported:
                msg = 'Unsupported method '+str(mth)+\
                      '. Available options for PAIRED data are: '+sp_str
        else:
            supported = ["t", "F", "ANOVA","Kruskal-Wallis", "kruskal", "Mann-Whitney", "sum"]
            if mth not in supported:
                msg = 'Unsupported method '+str(mth)+\
                      '. Available options for UNPAIRED data are: '+sp_str
        if len(msg) > 0:
            sp_str = str(supported)
            sp_str = sp_str.replace('[','')
            sp_str = sp_str.replace(']','')
            raise ValueError(msg)
    else:
        if mth != "t":
            print "WARNING: The optional argument `mth` will be ignored since R will not be used!"

    # Check `tail` if provided
    if not isinstance(tail,(str,unicode)):
        raise TypeError("The alternative hypothesis has to be specified using a string, not "+\
                        type(tail).__name__+"!")
    supported = {'greater':1,'less':-1,'two':0}
    spl       = supported.keys()
    if tail not in spl:
        sp_str = str(spl)
        sp_str = sp_str.replace('[','')
        sp_str = sp_str.replace(']','')
        msg = "The alternative hypothesis given by tail = '"+str(tail)+ "' is invalid. "+\
              "Available options are: "+sp_str
        raise ValueError(msg)

    # Save tail selection for output before we convert it to an integer 
    tail_dt1 = {"less":"less than","two":"different from","greater":"greater than"}
    tail_dt2 = {"less":"lower","two":"two","greater":"upper"}
    tail_st1 = tail_dt1[tail]
    tail_st2 = tail_dt2[tail]

    # Now convert string-tail to numeric value (lower, two, upper) -> (-1, 0, +1)
    tail = supported[tail]
    
    # Check the setting for the p-value correction
    if not isinstance(correction,(str,unicode)):
        raise TypeError("The multiplicity correction scheme has to be specified using a string, not "+\
                        type(correction).__name__+"!")
    if useR:
        supported = ["holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none", "Fisher",\
                     "Liptak", "Tippett", "MahalanobisT", "MahalanobisP", "minP", "maxT", "maxTstd",\
                     "sumT", "Direct", "sumTstd", "sumT2"]
        if correction not in supported:
            sp_str = str(supported)
            sp_str = sp_str.replace('[','')
            sp_str = sp_str.replace(']','')
            msg = "The multiplicity correction method given by correction = '"+str(correction)+\
                  "' is invalid. Available options are: "+sp_str
            raise ValueError(msg)
    else:
        if correction != "maxT":
            print "WARNING: The stats toolbox in MNE only supports standard Tmax correction of p-values!"
                            
    # Check if the user wants to see what's going on
    if not isinstance(verbose,bool):
        raise TypeError("The switch `verbose` has to be Boolean!")
    
    # If a file-name was provided make sure it's a string and check if the path exists
    if fname != None:
        if not isinstance(fname,(str,unicode)):
            raise TypeError("Filename has to be provided as string, not "+type(fname).__name__+"!")
        fname = str(fname)
        if fname.find("~") == 0:
            fname = os.path.expanduser('~') + fname[1:]
        slash = fname.rfind(os.sep)
        if slash >= 0 and not os.path.isdir(fname[:fname.rfind(os.sep)]):
            raise ValueError('Invalid path for output file: '+fname+'!')

    # Warn if output was turned off but labels were provided and assign default values to labels if necessary
    # (Do error checking here to avoid a breakdown at the very end of the code...)
    if verbose == False and fname is None:
        for chk in [vars,g1str,g2str]:
            if chk != None:
                print "WARNING: Output labels were provided but `verbose == False` and `fname == None`. "+\
                      "The labels will be ignored and no output will be shown/saved!"
                break
    else:
        if vars is None:
            vars = ['Variable '+str(v) for v in range(1,n_tests+1)]
        else:
            if not isinstance(vars,(list,np.ndarray)):
                raise TypeError('Variable names have to be provided as Python list/NumPy 1darray, not '+\
                                type(vars).__name__+'!')
            m = len(vars)
            if m != n_tests:
                raise ValueError('Number of variable labels for output and number of tests do not match up!')
            for var in vars:
                if not isinstance(var,(str,unicode)):
                    raise TypeError('All variables in the optional input `vars` must be strings!')
        
        if g1str is None:
            g1str = 'Group 1'
        else:
            if not isinstance(g1str,(str,unicode)):
                raise TypeError('The optional column label `g1str` has to be a string!')
        if g2str is None:
            g2str = 'Group 2'
        else:
            if not isinstance(g2str,(str,unicode)):
                raise TypeError('The optional column label `g2str` has to be a string!')

    # Initialize the output dictionary
    stats_dict = {}
            
    # Here we go: in case of paired samples and hatred for R, use Python's mne 
    if paired == True and useR == False:

        # Try to import/load everything we need below
        try:
            import mne
        except:
            raise ImportError("The Python module `mne` is not installed!")

        # Just to double check with user, say what's about to happen
        print "\nTesting statistical mean-difference of paired samples using the permutation t-test from `mne`"

        # Perform the actual testing
        statvals, pvals, dist = mne.stats.permutation_t_test(X-Y,n_permutations=nperms,\
                                                             tail=tail,n_jobs=1,verbose=False)

        # Store result in output dictionary
        stats_dict['pvals']    = pvals
        stats_dict['statvals'] = statvals
        if get_dist:
            stats_dict['dist'] = dist

    # Otherwise fire up R and use `flip`
    else:

        # Try to import/load everything we need below
        try:
            import pandas as pd
            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()
            from rpy2.robjects.packages import importr
            from rpy2.robjects import Formula

            # Set up our R name-spaces and see if `flip` is available
            R    = rpy2.robjects.r
            flip = importr('flip')

        except:
            msg = "Either the Python modules `pandas` and/or `rpy2` or "+\
                  "the R package `flip` is/are not installed!"
            raise ImportError(msg)
 
        # Just to double check with user, say what's about to happen
        print "\nPerforming a permutation "+mth+"-test of "+pairlst[paired]+" samples using the `R` package `flip`"

        # Construct a list of strings of the form 
        # ['a','b','c',...,'z','aa','ab','ac',...,'az','ba','bb','bc',...]
        abclist = (list(string.lowercase) + \
                  [''.join(x) for x in itertools.product(string.lowercase, repeat=2)])[:n_tests] + ['group']
        
        # Use that list to build a string of the form 'a + b + c +...+ aa + ab + ... ~ group' 
        frm_str = abclist[0]
        for ltr in abclist[1:-1]:
            frm_str += ' + ' + ltr
        frm_str += ' ~ group'
        
        # Construct an array that will be our factor in the R dataframe below: 
        # all rows of `X` are assigned the factor-level 1, the rest is 2
        group = 2*np.ones((nsamples_x + nsamples_y,1))
        group[:nsamples_x] = 1

        # Stack `X` and `Y` on top of each other, with columns labeled by `abclist`
        # in case of paired data, also append a stratification vector
        dfmat = np.hstack([np.vstack([X,Y]),group])
        stratarg = rpy2.rinterface.R_NilValue
        if paired:
            abclist += ['pairing']
            dfmat = np.hstack([dfmat,np.tile(np.arange(1,nsamples_x+1),(1,2)).T])
            stratarg = Formula("~pairing")
            
        # Create a pandas dataframe with columns labeled by abclist, that we immidiately convert to an R-dataframe
        r_dframe = pandas2ri.py2ri(pd.DataFrame(dfmat,columns=abclist))

        # Convert the string to an R formula
        r_frm = Formula(frm_str)

        # Do the actual testing in R
        result = R.flip(r_frm, data=r_dframe, tail=tail, perms=nperms, statTest=mth,\
                        Strata=stratarg, testType="permutation")
        result = flip.flip_adjust(result,method=correction)
            
        # Extract values from this R nightmare
        stats_dict['statvals'] = pandas2ri.ri2py(result.slots['res'][1])
        stats_dict['pvals']    = pandas2ri.ri2py(result.slots['res'][4])
        if get_dist:
            stats_dict['dist'] = pandas2ri.ri2py(result.slots['permT'])

        print "Done"

    # If wanted print/save the results
    if verbose or fname != None:

        # Construct string to be used as footer for the output file/last line of command line output
        permstr = "using "+str(nperms)+" permutations under the alternative hypothesis that "+\
                  g1str+" is "+tail_st1+" "+g2str+" ("+tail_st2+"-tailed) "
        if not useR:
            ft = "Statistical significance of group differences between paired samples was assessed using the "+\
                 "permutation t-test from the Python package MNE"+\
                 " (see http://martinos.org/mne/stable/mne-python.html)\n"+\
                 permstr+"\n"+\
                 "adjusted for multiple comparisons using the maximal test statistic Tmax. "
        else:
            ft = "Statistical significance of group-differences between "+pairlst[paired]+\
                 " samples was assessed using a "+mth+"-test"\
                 " from the R library flip (https://cran.r-project.org/web/packages/flip/index.html)\n"+\
                 permstr+"\n"+\
                 "adjusted for multiple comparisons based on a "+correction+" approach. \n"

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
