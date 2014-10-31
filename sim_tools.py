# sim_tools.py - Routines necessary to run a model simulation
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# June 23 2014

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import ipdb; import sys

import imp
import h5py
import os
import re
import psutil
from datetime import datetime
from texttable import Texttable
from scipy import ndimage
from nipy.modalities.fmri import hrf, utils
import shutil

try:
    from the_model import par, solve_model
except:
    print "ERROR: Could not import the model. Try running `make all` in a terminal first"
    sys.exit()

##########################################################################################
def run_model(V0, Z0, DA0, task, outfile, \
              seed=None, paramfile='parameters.py', symsyn=True, verbose=True, ram_use=0.2,\
              **kwargs):
    """
    Run a simulation using the neural population model

    Parameters
    ----------
    V0 : NumPy 1darray
        Initial conditions for excitatory neurons. If `N` regions are simulated then `V0` has 
        to have length `N`.
    Z0 : NumPy 1darray
        Initial conditions for inhibitory neurons. If `N` regions are simulated then `Z0` has 
        to have length `N`.
    DA0 : NumPy 1darray
        Initial conditions for dopamine levels. If `N` regions are simulated then `DA0` has 
        to have length `N`.
    task : string
        Specify which task should be simulated. Currently, only 'rest' and 'speech' are supported. 
    outfile : string
        File-name (including path if not in working directory) of HDF5 container that will be created to 
        save simulation results. See Notes for the structure of the generated container. Any existing 
        file will be renamed. The user has to have writing permissions for the given location.
    seed : integer
        Random number generator seed. To make meaningful comparisons between successive simulation
        runs, the random number seed should be fixed so that the solver uses the same Wiener process
        realizations. Also, if synaptic coupling strengths are sampled from a probability distribution,
        simulation results will vary from run to run unless the seed is fixed. 
    paramfile : string
        Parameter file-name (including path if not in working directory) that should be used for simulation. 
        The parameter file has to be a Python file (.py extension). For more details refer to `Examples` 
        below. You should have received a sample parameter file (`parameters.py`) with this copy 
        of `sim_tools.py`. 
    symsyn : bool
        Boolean switch determining whether synaptic coupling strengths should be symmetrized between 
        hemispheres (`symsyn=True`) or not. 
    verbose : bool
        If `True` the code will print a summary of the most important parameters and all used 
        keyword arguments (see below) in the simulation together with a progress bar to (roughly)
        estimate run time (requires the `progressbar` module). 
    ram_use : float
        Fraction of memory to use for caching simulation results before writing to disk 
        (0 < `ram_use` < 1). More available memory means fewer disk-writes and thus better performance, 
        i.e, the larger `ram_use` the faster this code runs. However, if too much RAM is allocated by 
        this routine it may stall the executing computer. By default, `ram_use = 0.2`, i.e., around 20% 
        of available memory is used. 
    kwargs : additional keyword arguments
        Instead of relying solely on a static file to define parameter values, it is also possible
        to pass on parameters to the code using keyword arguments (see `Examples` below). Note: parameters
        given as keyword arguments have higher priority than values set in `paramfile`, i.e., if 
        `p1 = 1` is defined in `paramfile` but `p1 = 2` is a keyword argument, the code will use 
        `p1 = 2` in the simulation. This behavior was intentionally implemented to enable the use 
        of this function within a parameter identification framework. 

    Returns
    -------
    Nothing : None
        Simulation results are saved in the HDF5 container specified by `outfile`. See `Notes` for details. 

    Notes
    -----
    Due to the (usually) high temporal resolution of simulations, results are not kept in memory (and 
    thus returned as variable in the caller's work-space) but saved directly to disk using the HDF5
    container `outfile`. The code uses the HDF library's data chunking feature to save entire
    segments on disk while running. By default the code will allocate around 20% of available 
    memory to cache simulation results. Hence, more memory leads to fewer disk-writes during 
    run-time and thus faster performance. 
    The structure of the generated output container is as follows: all state variables and the dopaminergic 
    gain `Beta` are stored at the top-level of the file. Additionally, the employed coupling 
    matrix `C` and dopamine connection matrix `D` are also saved in the top level group. All used parameters
    are saved in the subgroup `params`. 

    Examples
    --------
    Let `V0`, `Z0`, and `DA0` (NumPy 1darrays of length `N`) be initial conditions of the
    model. Assuming that a valid parameter file (called `parameters.py`) is located in the current 
    working directory, the following call will run a resting state simulation and save the output
    in the HDF5 container `sim_rest.h5`

    >>> run_model(V0,Z0,DA0,'rest','sim_rest.h5')

    Assume another parameter file, say, `par_patho.py` hold parameter settings simulating a 
    certain pathology. Then the command

    >>> run_model(V0,Z0,DA0,'rest','patho/sim_rest_patho.h5',paramfile='par_patho.py')

    runs a resting state simulation with the same initial conditions and saves the result in
    the container `sim_rest_patho.h5` in the sub-directory `patho` (which must already exist, otherwise
    an error is raised). 

    If only one or two parameters should be changed from their values found in a given parameter file, 
    it is probably more handy to change the value of these parameters from the command line, rather
    than to write a separate parameter file (that is identical to the original one except for two 
    values). Thus, assume the values of `VK` and `VL` should be -0.4 and -0.9 respectively, i.e., 
    different than those found in (the otherwise fine) `par_patho.py`. Then the command

    >>> run_model(V0,Z0,DA0,'rest','patho/sim_rest_patho.h5',paramfile='par_patho.py',VK=-0.4,VL=-0.9)
    
    runs the same resting state simulation as above but with `VK=-0.4` and `VL=-0.9`. This feature
    can also be used to efficiently embed `run_model` in a parameter identification framework.

    See also
    --------
    plot_sim : plot simulations generated by run_model

    References
    ----------
    .. [1] S. Fuertinger, J. C. Zinn, and K. Simonyan. A Neural Population Model Incorporating 
           Dopaminergic Neurotransmission during Complex Voluntary Behaviors. PLoS Computational Biology, 
           in press. 
    """

    # Sanity checks
    n = np.zeros((3,)); i = 0
    for vec in [V0, Z0, DA0]:
        try: 
            vs = vec.shape
        except: 
            msg = 'The initial conditions for V, Z, and DA have to be NumPy 1darrays, not '\
                  +type(vec).__name__+'!'
            raise TypeError(msg)
        if (len(vs) >= 2 and min(vs) > 1) or (len(vs) == 1 and min(vs) < 2):
            raise ValueError('The initial conditions for V, Z, and DA have to be NumPy 1darrays!')
        if np.isnan(vec).max()==True or np.isinf(vec).max()==True or np.isreal(vec).min()==False:
            msg = 'The initial conditions for V, Z, and DA have '+\
                  'to be real valued NumPy 1darrays without Infs or NaNs!'
            raise ValueError(msg)
        n[i] = vec.size
        i += 1
    if np.unique(n).size > 1:
        raise ValueError('The initial conditions for V, Z, and DA have to have the same length!')

    if type(task).__name__ != 'str':
        raise TypeError('Task has to be specified as string, not '+type(task).__name__+'!')
    if task != 'rest' and task != 'speech':
        raise ValueError("Task has to be either 'rest' or 'speech'!")
    
    if type(outfile).__name__ != 'str':
        raise TypeError('Output file-name has to be a string specifying the path to an HDF5 container!')

    if seed != None:
        try:
            bad = (np.round(seed) != seed)
        except: raise TypeError("Random number seed has to be an integer!")
        if bad: raise ValueError("Random number seed has to be an integer!")
        seed = int(seed)
    else:
        seed = np.random.get_state()[1][0]

    if type(paramfile).__name__ != 'str':
        raise TypeError('Parameter file has to be specified using a string!')

    if symsyn != True and symsyn != False:
        raise TypeError("The switch `symsyn` has to be Boolean!")

    if verbose != True and verbose != False:
        raise TypeError("The switch `verbose` has to be Boolean!")

    try:
        bad = (ram_use <= 0) or (ram_use >= 1)
    except: 
        raise TypeError('The parameter ram_use has to be a float, not '+type(ram_use).__name__+'!')
    if bad: raise ValueError('The parameter ram_use has to satisfy 0 < ram_use < 1!')

    # Append '.h5' extension to outfile if necessary
    if outfile[-3:] != '.h5':
        outfile = outfile + '.h5'

    # Check if paramfile has an extension, if yes, rip it off
    if paramfile[-3:] == '.py':
        paramfile = paramfile[0:-3]

    # Divide paramfile into file-name and path
    slash = paramfile.rfind(os.sep)
    if slash  < 0:
        pth   = '.'
        fname = paramfile
    else:
        pth   = paramfile[0:slash+1]
        fname = paramfile[slash+1:]

    # Import parameters module and initialize corresponding dictionary (remove __file__, ... )
    param_py = imp.load_module(fname,*imp.find_module(fname,[pth]))
    p_dict   = {}
    for key, value in param_py.__dict__.items():
        if key[0:2] != "__":
            p_dict[key] = value

    # Try to load the coupling matrix
    try:
        f = h5py.File(param_py.matrices,'r')
    except: raise ValueError("Could not open HDF5 file holding the coupling matrix")
    try:
        if kwargs.has_key('C'):
            C = kwargs['C']
        else:
            C = f['C'].value
        if kwargs.has_key('D'):
            D = kwargs['D']
        else:
            D = f['D'].value
        names  = f['names'].value
        labels = None
        if f.keys().count('labels'): labels = f['labels'].value
        f.close()
    except: 
        raise ValueError("HDF5 file "+param_py.matrices+" does not have the required fields!")

    # Do a quick checkup of the matrices
    for mat in [C,D]:
        try:
            shm = mat.shape
        except: raise TypeError("Coupling/dopamine matrix has to be a NumPy 2darray!")
        if len(shm) != 2:
            raise ValueError("Coupling/dopamine matrix has to be a NumPy 2darray!")
        if shm[0] != shm[1]:
            raise ValueError("Coupling/dopamine matrix has to be square!")
        if np.isnan(mat).max()==True or np.isinf(mat).max()==True or np.isreal(mat).min()==False:
            msg = 'Coupling/dopamine matrix must be a real valued NumPy 2darray without Infs or NaNs!'
            raise ValueError(msg)

    # Put ones on the diagonal of the coupling matrix to ensure compatibility with the code
    np.fill_diagonal(C,1.0)

    # Get dimension of matrix and check correspondence
    N = C.shape[0]
    if N != D.shape[0]:
        raise ValueError("Dopamine and coupling matrices don't have the same dimension!")
    if len(names) != N: 
        raise ValueError("Matrix is "+str(N)+"-by-"+str(N)+" but `names` have length "+str(len(names))+"!")
    for nm in names:
        if type(nm).__name__ != 'str' and type(nm).__name__ != 'string_':
            raise ValueError("Names have to be provided as Python list/NumPy array of strings!")

    # Get synaptic couplings from parameter file (and set seed of random number generator)
    np.random.seed(seed)
    aei = eval(param_py.aei)
    aie = eval(param_py.aie)
    ani = eval(param_py.ani)
    ane = eval(param_py.ane)

    # If wanted, make sure left/right hemispheres have balanced coupling strengths
    if symsyn:

        # Get indices of left-hemispheric regions and throw a warning if left/right don't match up
        regex = re.compile("[Ll]_*")
        match = np.vectorize(lambda x:bool(regex.match(x)))(names)
        l_ind = np.where(match == True)[0]
        r_ind = np.where(match == False)[0]
        if l_ind.size != r_ind.size:
            print "WARNING: Number of left-side regions = "+str(l_ind.size)+\
                  " not equal to number of right-side regions = "+str(r_ind.size)
            
        # Equalize coupling strengths
        aei[l_ind] = aei[r_ind]
        aie[l_ind] = aie[r_ind]
        ani[l_ind] = ani[r_ind]
        ane[l_ind] = ane[r_ind]

    # Save updated coupling strengths and random number generator seed in dictionary
    p_dict['aei']  = aei
    p_dict['aie']  = aie
    p_dict['ani']  = ani
    p_dict['ane']  = ane
    p_dict['seed'] = seed

    # If a resting state simulation is done, make sure dopamine doesn't kick in 
    rmin = param_py.rmin
    if task == 'rest':
        rmax = np.ones((N,))*param_py.rmin
    else:
        rmax = eval(param_py.rmax)

    # Save (possibly updated) dopamine bounds and given task in dictionary
    p_dict['rmax'] = rmax
    p_dict['rmin'] = rmin
    p_dict['task'] = task

    # Get ion channel parameters
    p_dict['TCa'] = eval(param_py.TCa)

    # Compute length for simulation and speech on-/offset times
    len_cycle = param_py.stimulus + param_py.production + param_py.acquisition
    speechon  = param_py.stimulus
    speechoff = param_py.stimulus + param_py.production

    # Save that stuff
    p_dict['len_cycle'] = len_cycle
    p_dict['speechon']  = speechon
    p_dict['speechoff'] = speechoff

    # Set/get initial time for simulation
    if p_dict.has_key('tstart'): 
        tstart = param_py.tstart
        if verbose: print "WARNING: Using custom initial time of  "+str(tstart)+" (has to be in ms)!"
    else:
        tstart = 0

    # Set/get step-size for simulation 
    if p_dict.has_key('dt'): 
        dt = param_py.dt
        if verbose: print "WARNING: Using custom step-size of "+str(dt)+" (has to be in ms)!"
    else:
        dt = 1e-1

    # Get sampling step size (in ms) and check if "original" step-size makes sense
    ds = 1/param_py.s_rate*1000
    if dt > ds:
        print "WARNING: Step-size dt = "+str(dt)+\
              " larger than chosen sampling frequency of "+str(s_rate)+"Hz."+\
              " Using dt = "+str(ds)+"ms instead. "
        dt = ds

    # Compute sampling rate (w.r.t dt)
    s_step = int(np.round(ds/dt))

    # Save step-size and sampling rate in dictionary for later reference
    p_dict['dt']     = dt
    p_dict['s_step'] = s_step
        
    # Compute end time for simulation (in ms) and allocate time-step array
    if kwargs.has_key('n_cycles'):
        n_cycles = kwargs['n_cycles']
    else:
        n_cycles = param_py.n_cycles
    tend   = tstart + len_cycle*n_cycles*1000
    tsteps = np.arange(tstart,tend,dt)

    # Get the size of the time-array
    tsize   = tsteps.size

    # Before laying out output HDF5 container, rename existing files to not accidentally overwrite 'em
    moveit(outfile)
    
    # Chunk outifle depending on available memory (eat up ~ 20% of RAM)
    meminfo = psutil.virtual_memory()
    maxmem  = int(meminfo.available*0.2/(5*N)/np.dtype('float64').itemsize)

    # If the whole array fits into memory load it once, otherwise chunk it up
    if tsteps.size <= maxmem:
        blocksize = [tsize]
        csize     = int(np.ceil(tsize/s_step))
        chunksize = [csize]
        chunks    = True
    else:
        bsize     = int(tsize//maxmem)
        rest      = int(np.mod(tsize,maxmem))
        blocksize = [maxmem]*bsize
        if rest > 0: blocksize = blocksize + [rest]
        numblocks = len(blocksize)
        csize     = int(np.ceil(maxmem/s_step))
        restc     = int(np.ceil(blocksize[-1]/s_step))
        chunksize = [csize]*(numblocks - 1) + [restc]
        chunks    = (N,csize)

    # Convert blocksize and chunksize to NumPy arrays
    blocksize = np.array(blocksize)
    chunksize = np.array(chunksize)

    # Get the number of elements that will be actually saved
    n_elems = chunksize.sum()

    # Create output HDF5 container
    f = h5py.File(outfile)

    # Create datasets for numeric variables
    f.create_dataset('C',data=C)
    f.create_dataset('D',data=D)
    f.create_dataset('V',shape=(N,n_elems),chunks=chunks)
    f.create_dataset('Z',shape=(N,n_elems),chunks=chunks)
    f.create_dataset('DA',shape=(N,n_elems),chunks=chunks)
    f.create_dataset('QV',shape=(N,n_elems),chunks=chunks)
    f.create_dataset('Beta',shape=(N,n_elems),chunks=chunks)
    f.create_dataset('t',data=np.linspace(tstart,tend,n_elems))

    # Now copy matrices also to p_dict and initialize the model's C-class with it
    p_dict['C'] = C
    p_dict['D'] = D

    # If user provided some additional parameters as keyword arguments, now's the time to assign them
    for key, value in kwargs.items():
        p_dict[key] = value

    # Save parameters (but exclude stuff imported in the parameter file)
    pg = f.create_group('params')
    for key,value in p_dict.items():
        valuetype  = type(value).__name__
        if valuetype != 'instance' and valuetype != 'module' and valuetype != 'function':
            pg.create_dataset(key,data=value)
    pg.create_dataset('names',data=names)
    
    # If labels were provided in the matrix file too, store them
    if labels != None: pg.create_dataset('labels',data=labels)

    # Close container and write to disk
    f.close()

    # Initialize parameter C-class (struct) for the model
    params = par(p_dict)

    # Concatenate initial conditions
    VZD0 = np.hstack((np.hstack((V0.squeeze(),Z0.squeeze())),DA0.squeeze()))

    # Let the user know what's going to happen...
    pstr = "--"
    if len(kwargs) > 0:
        pstr = str(kwargs.keys())
        pstr = pstr.replace("[","")
        pstr = pstr.replace("]","")
        pstr = pstr.replace("'","")
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l", "l"])
    table.add_rows([["Simulating ",task.upper()],\
                    ["#cycles: ",str(p_dict['n_cycles'])],\
                    ["parameter file:",paramfile+".py"],\
                    ["keyword args:",pstr],\
                    ["matrix file:",p_dict['matrices']],\
                    ["output:",outfile]])
    if verbose: print "\n"+table.draw()+"\n"

    # Finally... Run the actual simulation
    VZD = solve_model(VZD0,tsteps,params,blocksize,chunksize,seed,int(verbose),outfile)

    # Done!
    if verbose: print "\nDone\n"

##########################################################################################
def plot_sim(fname,names="all",raw=True,bold=False,figname=None):
    """
    Plot a simulation generated by run_model

    Parameters
    ----------
    fname : string
        File-name (including path if not in working directory) of HDF5 container that was generated 
        by `run_model`. 
    names : str or Python list/NumPy 1darray
        Specify regions to plot. Either use the region's name as found in the `params` group of
        the HDF5 container given by `fname` (e.g., `names='L_IFG'`) or its index in the `names` list
        (e.g., `names = 3`). Use a list or NumPy 1darray to specify more than one region
        (e.g., `names = ['L_IFG','R_IFG']` or `names = [3,15]`). By default, all regions are plotted. 
    raw : bool
        If `True` then the raw model output will be plotted. Depending on the setting of `names` (see
        above) the simulation length, and the model dimension (i.e., the number of modeled regions) 
        this may result in a very 'busy' plot. 
    bold : bool
        If True then the previously converted simulated BOLD signals will be plotted (if no BOLD
        signal is found in the input container specified by `fname`, an error is raised). 
    figname : string
        String to be used as window title for generated figures.

    Returns
    -------
    Nothing : None

    Notes
    -----
    None

    See also
    --------
    run_model : used to run a simulation
    make_bold : convert raw simulation output to a BOLD signal  
    """

    # Sanity checks
    if type(fname).__name__ != 'str':
        raise TypeError("Name of HDF5 file has to be a string!")

    if raw != True and raw != False:
        raise TypeError("The switch `raw` has to be Boolean!")

    if bold != True and bold != False:
        raise TypeError("The switch `bold` has to be Boolean!")

    # Try to open given HDF5 container
    try:
        f = h5py.File(fname,'r')
    except: raise ValueError("Cannot open "+fname+"!")
    try:
        rois_infile = f['params']['names'].value.tolist()
        f.close()
    except: raise ValueError("HDF5 file "+fname+" does not have the required fields!")

    # Check if list of names to plot was provided. If yes, make sure they make sense
    if type(names).__name__ != "str":
        doleg = True
        try:
            names = list(names)
        except: raise TypeError("Regions to plot have to be provided as Python list or NumPy 1darray!")
        idx = []
        for name in names:
            try:
                idx.append(rois_infile.index(name))
            except: raise ValueError("Region "+name+"not found in file!")
    else:
        if names == "all":
            idx   = range(len(rois_infile))
            doleg = False
        else:
            try:
                idx = rois_infile.index(names)
            except: raise ValueError("Region "+names+"not found in file!")
            doleg = True

    if figname != None:
        if type(figname).__name__ != "str":
            raise TypeError("Figure name has to be a string, not "+type(figname).__name__+"!")

    # Fix sorting of idx so that smart indexing below works
    if doleg:
        idx    = np.array(idx)
        names  = np.array(names)
        sorted = idx.argsort()
        names  = names[sorted]
        idx    = idx[sorted]

    # After all the error checking, reopen the file
    f = h5py.File(fname,'r')

    # Turn on interactive plotting
    plt.ion()

    # Plot raw model output
    if (raw):

        # Compute plotting step size s.t. we plot every 100ms (=10Hz) (if possible)
        s_rate = f['params']['s_rate'].value
        if s_rate > 10:
            p_rate = int(s_rate//10)
        else:
            p_rate = s_rate

        # Get quantities for plotting
        t    = f['t'][::p_rate]
        V    = f['V'][idx,::p_rate].T
        QV   = f['QV'][idx,::p_rate].T
        Beta = f['Beta'][idx,::p_rate].T
        DA   = f['DA'][idx,::p_rate].T
        tmin = t.min() - t[1]
        tmax = t.max() + t[1]

        # Workaround for the time being
        if t.size != V.shape[0]:
            t = np.linspace(t[0],t[-1],V.shape[0])

        # Prepare window and plot stuff
        fig = plt.figure(figsize=(10,7.5))
        if figname != None: fig.canvas.set_window_title(figname)
        plt.suptitle("Raw Model Output from "+fname,fontsize=18)

        sp = plt.subplot(4,1,1)
        plt.plot(t,V)
        if doleg:
            plt.legend(names)
        plt.ylabel('mV',fontsize=16)
        [ymin,ymax] = sp.get_ylim()
        plt.yticks([ymin,0,ymax],fontsize=10)
        sp.set_xlim(left=tmin,right=tmax)
        plt.xticks([],fontsize=8)
        plt.title("V",fontsize=16)
        plt.draw()

        sp = plt.subplot(4,1,2)
        plt.plot(t,QV)
        plt.ylabel('Firing',fontsize=16)
        [ymin,ymax] = sp.get_ylim()
        plt.yticks([0,ymax],fontsize=10)
        sp.set_xlim(left=tmin,right=tmax)
        plt.xticks([],fontsize=8)
        plt.title("QV",fontsize=16)
        plt.draw()

        sp = plt.subplot(4,1,3)
        plt.plot(t,Beta)
        plt.ylabel('Gain Factor',fontsize=16)
        [ymin,ymax] = sp.get_ylim()
        plt.yticks([ymin,ymax],fontsize=10)
        sp.set_xlim(left=tmin,right=tmax)
        plt.xticks([],fontsize=8)
        plt.title(r"$\beta$",fontsize=16)
        plt.draw()

        sp = plt.subplot(4,1,4)
        plt.plot(t,DA)
        plt.ylabel('mM',fontsize=16)
        [ymin,ymax] = sp.get_ylim()
        plt.yticks([ymin,ymax],fontsize=10)
        sp.set_xlim(left=tmin,right=tmax)
        plt.xticks(fontsize=10)
        plt.xlabel("ms")
        plt.title("DA",fontsize=16)
        plt.draw()

    # Plot raw model output
    if (bold):
        
        # Try to load the BOLD data from file
        try:
            BOLD = f['BOLD'][idx,:].T
        except: 
            f.close()
            raise ValueError("No BOLD data found in file "+fname+"!")

        # Get x-extent of data and create x-ticks vector
        xmax = BOLD.shape[0] + 1
        xtv  = np.arange(-1,xmax)

        # Prepare window and plot stuff
        fig = plt.figure(figsize=(10,7.5))
        if figname != None: fig.canvas.set_window_title(figname)
        plt.title("BOLD data "+fname,fontsize=18)
        sp = plt.subplot(111)
        plt.plot(BOLD)
        if doleg:
            plt.legend(names)
        plt.xticks(xtv,fontsize=10)
        [ymin,ymax] = sp.get_ylim()
        plt.yticks([ymin,0,ymax],fontsize=10)

    # Close file and return
    f.close()

##########################################################################################
def make_D(target,source,names,values=None):
    """
    Create matrix of afferent/efferent dopamine regions in the model

    Parameters
    ----------
    target : Python list or NumPy 1darray
        Python list or NumPy 1darray of region names that are affected by dopamine release
        (has to be the same length as `source`).
    source : Python list or NumPy 1darray
        Python list or NumPy 1darray of region names that steer dopamine release. 
        (has to be the same length as `target`).
    names : Python list or NumPy 1darray
        Names of all regions in the model. If `names` has length `N`, the resulting 
        dopamine connection matrix will by `N`-by-`N`. 
    values : Python list or NumPy 1darray
        By default, dopamine links are binary, i.e., all links have unit weight. By passing 
        a `values` array, certain links can be emphasized (weight > 1) or weakened (weight < 1). 
        Entries of the `values` array have to be calibrated based on the values of `b_hi`, `b_lo`, 
        and `a`. 
    
    Returns
    -------
    D : NumPy 2darray
        A `N`-by-`N` array. Every row that has a non-zero entry signifies a dopamine target, 
        and every non-zero column corresponds to a dopamine source. 

    Notes
    -----
    None

    Examples
    --------
    For the sake of simplicity consider a brain "parcellation" consisting of three (bilateral) regions, 
    called `A`, `B`, and `C`. Thus, we define the following `names` array

    >>> names = ['L_A','L_B','L_C','R_A','R_B','R_C']

    Assume that in the left hemisphere dopamine is mainly concentrated in region `B`, its release is 
    steered by neural firing in region `A`. In the right hemisphere, dopamine release in region `C` is 
    depending on neural activity in area `B`. Then the `target` and `source` arrays are given by
    
    >>> target = ['L_B','R_C']
    >>> source = ['L_A','R_B']

    Then the call

    >>> D = make_D(target,source,names)
    >>> D
    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.]])

    generates a 6-by-6 NumPy array `D`, that has non-zero entries in rows 2 and 6 (because
    'L_B' is the second, and 'R_C' the sixth element of the list `names`), and columns 1 and 5 
    (because 'L_A' is the first, and 'R_B' the 5th element of the list `names`). Thus, the row/column
    index of each non-zero entry in `D` has the format target-by-source. 

    See also
    --------
    None
    """

    # Sanity checks
    for tsn in [target,source,names]:
        try:
            tsn = list(tsn)
        except: 
            msg = "Inputs target, source and names have to be NumPy 1darrays or Python lists, not "+\
                  type(tsn).__name__
            TypeError(msg)

    if len(source) != len(target):
        raise ValueError("Length of source and target lists/arrays does not match up!")

    if values != None:
        if len(values) != len(target):
            raise ValueError("Length of values list/array does not match up!")
    else:
        values = np.ones((len(target),))

    # Convert (if we're having a NumPy array) names to a Python list
    names = list(names)

    # Get dimension we're dealing with here
    N = len(names)

    # Create dopamine matrix
    D = np.zeros((N,N))

    # Fill the matrix
    for i in xrange(len(source)):

        # Get row and column indices
        row = names.index(target[i])
        col = names.index(source[i])

        # Matrix is targets-by-sources
        D[row,col] = values[i]

    return D

##########################################################################################
def make_bold(fname, stim_onset=None):
    """
    Convert raw model output to BOLD signal

    Parameters
    ----------
    fname : string
        File-name (including path if not in working directory) of HDF5 container that was generated 
        by `run_model`. 
    stim_onset : float
        Time (in seconds) of stimulus onset. By default, onset/offset timings of 
        stimuli are stored in the HDF5 container generated by `run_model`. Only override 
        this setting, if you know what you are doing. 

    Returns
    -------
    Nothing : None
        The computed BOLD signal is stored as dataset `BOLD` at the top level of the HDF5 container 
        specified by `fname`. 

    Notes
    -----
    Regional neural voltages are converted to BOLD time-series using the linear hemodynamic response 
    function proposed by Glover [1]_. For details consult the supporting information of our 
    paper [2]_. 


    References
    ----------
    .. [1] Glover G (1999) Deconvolution of Impulse Response in Event-Related BOLD FMRI. 
           NeuroImage 9: 416-429.

    .. [2] S. Fuertinger, J. C. Zinn, and K. Simonyan. A Neural Population Model Incorporating 
           Dopaminergic Neurotransmission during Complex Voluntary Behaviors. PLoS Computational Biology, 
           in press. 
    """

    # Sanity checks
    if type(fname).__name__ != 'str':
        raise TypeError("Name of HDF5 file has to be a string!")

    # Try to open given HDF5 container
    try:
        f = h5py.File(fname)
    except: raise ValueError("Cannot open "+fname+"!")
    try:
        V = f['V'].value
    except: 
        f.close()
        raise ValueError("HDF5 file "+fname+" does not have the required fields!")

    # Make sure stim_onset makes sense
    if stim_onset != None:
        try: np.round(stim_onset)
        except: raise TypeError("The stimulus onset time has to be a real scalar!")

    # Get task from file to start sub-sampling procedure
    task = f['params']['task'].value

    # Compute cycle length based on the sampling rate used to generate the file
    N         = f['params']['names'].size
    n_cycles  = f['params']['n_cycles'].value
    s_rate    = f['params']['s_rate'].value
    len_cycle = f['params']['len_cycle'].value
    cycle_idx = int(np.round(s_rate*len_cycle))

    # Compute step size and (if not provided by the user) compute stimulus onset time
    dt = 1/s_rate
    if stim_onset == None: stim_onset = f['params']['stimulus'].value
    
    # Use Glover's Hemodynamic response function as convolution kernel (with default length 32)
    hrft       = utils.lambdify_t(hrf.glover(utils.T))
    hrf_kernel = np.hstack((np.zeros((int(s_rate*stim_onset),)),hrft(np.arange(0,32,dt))))

    # Convolve the de-meaned model time-series with the kernel 
    convV = ndimage.filters.convolve1d((V.T - V.mean(axis=1)).T,hrf_kernel,mode='constant')

    # Allocate space for BOLD signal
    BOLD = np.zeros((N,n_cycles))

    # Sub-sample convoluted data depending on task to get BOLD signal
    if task == 'speech':

        # Get interval to be considered for boldification
        start = int(np.round(f['params']['speechoff'].value*s_rate))
        stop  = start + int(np.round(f['params']['acquisition'].value*s_rate))

    elif task == 'rest':

        # Get interval to be considered for boldification
        start = 0
        stop  = int(np.round(f['params']['stimulus'].value*s_rate))

    else:
        raise ValueError("Don't know what to do for task "+task)

    # Compute BOLD signal for all time points
    for j in xrange(n_cycles):
        BOLD[:,j] = convV[:,start:stop].mean(axis=1)
        start += cycle_idx
        stop  += cycle_idx

    # Re-scale the the signal
    BOLD = BOLD*0.02

    # Save it to the file
    try:
        f.create_dataset('BOLD',data=BOLD)
    except:
        f['BOLD'].write_direct(BOLD)
    f.close()

##########################################################################################
def show_params(fname):
    """
    Pretty-print all parameters used in a simulation

    Parameters
    ----------
    fname : string
        Filename (including path if not in working directory) of HDF5 container that was generated 
        by `run_model`.

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

    # Sanity checks
    if type(fname).__name__ != 'str':
        raise TypeError("Name of HDF5 file has to be a string!")

    # Try to open given HDF5 container
    try:
        f = h5py.File(fname,'r')
    except: raise ValueError("Cannot open "+fname+"!")
    try:
        par_grp = f['params']
        f.close()
    except: raise ValueError("HDF5 file "+fname+" does not have the required fields!")

    # After all the error checking, reopen the file
    f = h5py.File(fname,'r')
    
    # Create a list for Texttable
    tlist = []
    for key in f['params'].keys():
        tlist.append([key,f['params'][key].value])

    # Close file
    f.close()

    # Print table
    print"\nShowing parameters of file "+fname
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l", "l"])
    table.add_rows([["Parameter","Value"]],header=True)
    table.add_rows(tlist,header=False)
    print "\n"+table.draw()+"\n"
    
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
    except: raise TypeError("Input must be a NumPy array/Python list, not "+type(arr).__name__+"!")
    sha = arr.shape
    if len(sha) > 2 or (len(sha) == 2 and min(sha) != 1):
        raise ValueError("Input must be a NumPy 1darray or Python list!")

    if type(expr).__name__ != "str":
        raise TypeError("Input expression has to be a string, not "+type(expr).__name__+"!")

    # Now do something: start by compiling the input expression
    regex = re.compile(expr)

    # Create a generalized function to find matches
    match = np.vectorize(lambda x:bool(regex.match(x)))(arr)

    # Get matching indices and return
    return np.where(match == True)[0]

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
    if type(fname).__name__ != "str":
        raise TypeError("File-/Directory-name has to be a string!")

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
    
