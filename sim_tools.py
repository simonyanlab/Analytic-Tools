# sim_tools.py - Routines for running a model simulation
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: June 23 2014
# Last modified: <2017-09-15 16:46:31>

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
import tempfile

from recipes import moveit

try:
    from the_model import par, solve_model
except:
    print "\n\tWARNING: Could not import the model - `run_model` will not work!"
    print "\tTry running `make all` in a terminal first"

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
    task : str
        Specify which task should be simulated. Currently, only 'rest' and 'speech' are supported. 
    outfile : str
        File-name (including path if not in working directory) of HDF5 container that will be created to 
        save simulation results. See Notes for the structure of the generated container. Any existing 
        file will be renamed. The user has to have writing permissions for the given location.
    seed : int
        Random number generator seed. To make meaningful comparisons between successive simulation
        runs, the random number seed should be fixed so that the solver uses the same Wiener process
        realizations. Also, if synaptic coupling strengths are sampled from a probability distribution,
        simulation results will vary from run to run unless the seed is fixed. 
    paramfile : str
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
           10(11), 2014. 
    """

    # Sanity checks for initial conditions
    n = np.zeros((3,))
    vnames = ['V0','Z0','DA0']
    for i, vec in enumerate([V0, Z0, DA0]):
        arrcheck(vec,'vector',vnames[i])
        n[i] = vec.size
    if np.unique(n).size > 1:
        raise ValueError('The initial conditions for `V`, `Z`, and `DA` have to have the same length!')
        
    # Check the `task` string
    if not isinstance(task,(str,unicode)):
        raise TypeError('Task has to be specified as string, not '+type(task).__name__+'!')
    task = str(task)
    if task != 'rest' and task != 'speech':
        raise ValueError("The value of `task` has to be either 'rest' or 'speech'!")

    # The path to the output file should be a valid
    if not isinstance(outfile,(str,unicode)):
        raise TypeError('Output filename has to be a string!')
    outfile = str(outfile)
    if outfile.find("~") == 0:
        outfile = os.path.expanduser('~') + outfile[1:]
    slash = outfile.rfind(os.sep)
    if slash >= 0 and not os.path.isdir(outfile[:outfile.rfind(os.sep)]):
        raise ValueError('Invalid path for output file: '+outfile+'!')

    # Set or get random number generator seed
    if seed is not None:
        scalarcheck(seed,'seed',kind='int')
    else:
        seed = np.random.get_state()[1][0]
    seed = int(seed)

    # Make sure `paramfile` is a valid path
    if not isinstance(paramfile,(str,unicode)):
        raise TypeError('Parameter file has to be specified using a string!')
    paramfile = str(paramfile)
    if paramfile.find("~") == 0:
        paramfile = os.path.expanduser('~') + paramfile[1:]
    if not os.path.isfile(paramfile):
        raise ValueError('Parameter file: '+paramfile+' does not exist!')

    # Make sure `symsyn` and `verbose` are Boolean
    if not isinstance(symsyn,bool):
        raise TypeError("The switch `symsyn` has to be Boolean!")
    if not isinstance(verbose,bool):
        raise TypeError("The switch `verbose` has to be Boolean!")

    # Finally, check `ram_use`
    scalarcheck(ram_use,'ram_use',bounds=[0,1])

    # Append '.h5' extension to `outfile` if necessary
    if outfile[-3:] != '.h5':
        outfile = outfile + '.h5'

    # Check if `paramfile` has an extension, if yes, rip it off
    if paramfile[-3:] == '.py':
        paramfile = paramfile[0:-3]

    # Divide `paramfile` into file-name and path
    slash = paramfile.rfind(os.sep)
    if slash  < 0:
        pth   = '.'
        fname = paramfile
    else:
        pth   = paramfile[0:slash+1]
        fname = paramfile[slash+1:]

    # Import parameters module and initialize corresponding dictionary (remove `__file__`, etc)
    param_py = imp.load_module(fname,*imp.find_module(fname,[pth]))
    p_dict   = {}
    for key, value in param_py.__dict__.items():
        if key[0:2] != "__":
            p_dict[key] = value

    # Try to load coupling and dopamine pathway matrices
    mfile  = "None"
    vnames = ['C','D']
    for mat_str in vnames:
        if kwargs.has_key(mat_str):
            p_dict[mat_str] = kwargs[mat_str]
        else:
            try:
                p_dict[mat_str] = h5py.File(param_py.matrices,'r')[mat_str].value
                mfile = p_dict['matrices']
            except:
                raise ValueError("Error reading `"+param_py.matrices+"`!")
            arrcheck(p_dict[mat_str],'matrix',mat_str)

    # Try to load ROI names
    try:
        names = h5py.File(param_py.matrices,'r')['names'].value
        mfile = p_dict['matrices']
    except:
        try:
            names = kwargs['names']
        except:
            raise ValueError("A NumPy 1darray or Python list of ROI names has to be either specified "+\
                             "in a matrix container or provided as keyword argument!")
    p_dict['names'] = names
        
    # See if we have an (optional) list/array of ROI-shorthand labels
    try:
        p_dict['labels'] = h5py.File(param_py.matrices,'r')['labels'].value
        mfile = p_dict['matrices']
    except:
        if kwargs.has_key('labels'):
            p_dict['labels'] = kwargs['labels']

    # Put ones on the diagonal of the coupling matrix to ensure compatibility with the code
    np.fill_diagonal(p_dict['C'],1.0)

    # Get dimension of matrix and check correspondence
    N = p_dict['C'].shape[0]
    if N != p_dict['D'].shape[0]:
        raise ValueError("Dopamine and coupling matrices don't have the same dimension!")
    if len(names) != N: 
        raise ValueError("Matrix is "+str(N)+"-by-"+str(N)+" but `names` has length "+str(len(names))+"!")
    for nm in names:
        if not isinstance(nm,(str,unicode)):
            raise ValueError("Names have to be provided as Python list/NumPy array of strings!")

    # If user provided some additional parameters as keyword arguments, copy them to `p_dict`
    for key, value in kwargs.items():
        p_dict[key] = value

    # Get synaptic couplings (and set seed of random number generator)
    np.random.seed(seed)
    if kwargs.has_key('aei'):
        aei = kwargs['aei']
    else:
        aei = eval(param_py.aei)
    if kwargs.has_key('aie'):
        aie = kwargs['aie']
    else:
        aie = eval(param_py.aie)
    if kwargs.has_key('ani'):
        ani = kwargs['ani']
    else:
        ani = eval(param_py.ani)
    if kwargs.has_key('ane'):
        ane = kwargs['ane']
    else:
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

    # If a resting state simulation is done, make sure dopamine doesn't kick in, i.e., enforce `rmax == rmin`
    if task == 'rest':
        rmax = np.ones((N,))*p_dict['rmin']
    else:
        if not kwargs.has_key('rmax'):
            p_dict['rmax'] = eval(param_py.rmax)

    # Save given task in dictionary
    p_dict['task'] = task

    # Get ion channel parameters
    if not kwargs.has_key('TCa'):
        p_dict['TCa'] = eval(param_py.TCa)

    # Compute length for simulation and speech on-/offset times
    len_cycle = p_dict['stimulus'] + p_dict['production'] + p_dict['acquisition']
    speechon  = p_dict['stimulus']
    speechoff = p_dict['stimulus'] + p_dict['production']

    # Save that stuff
    p_dict['len_cycle'] = len_cycle
    p_dict['speechon']  = speechon
    p_dict['speechoff'] = speechoff

    # Set/get initial time for simulation
    if p_dict.has_key('tstart'): 
        tstart = p_dict['tstart']       # Use `p_dict` here, since `tstart` could be a kwarg!
        if verbose: print "WARNING: Using custom initial time of  "+str(tstart)+" (has to be in ms)!"
    else:
        tstart = 0

    # Set/get step-size for simulation 
    if p_dict.has_key('dt'): 
        dt = p_dict['dt']
        if verbose: print "WARNING: Using custom step-size of "+str(dt)+" (has to be in ms)!"
    else:
        dt = 1e-1

    # Get sampling step size (in ms) and check if "original" step-size makes sense
    ds = 1/p_dict['s_rate']*1000
    if dt > ds:
        print "WARNING: Step-size dt = "+str(dt)+\
              " larger than chosen sampling frequency of "+str(s_rate)+"Hz."+\
              " Using dt = "+str(ds)+"ms instead. "
        dt = ds

    # Compute sampling rate (w.r.t `dt`)
    s_step = int(np.round(ds/dt))

    # Save step-size and sampling rate in dictionary for later reference
    p_dict['dt']     = dt
    p_dict['s_step'] = s_step
        
    # Compute end time for simulation (in ms) and allocate time-step array
    tend   = tstart + len_cycle*p_dict['n_cycles']*1000
    tsteps = np.arange(tstart,tend,dt)

    # Get the size of the time-array
    tsize   = tsteps.size

    # Before laying out output HDF5 container, rename existing files to not accidentally overwrite 'em
    moveit(outfile)
    
    # Chunk outifle depending on available memory (eat up ~ 100*`ram_use`% of RAM)
    datype  = np.dtype('float64')
    meminfo = psutil.virtual_memory()
    maxmem  = int(meminfo.available*ram_use/(5*N)/datype.itemsize)
    maxmem  += s_step - np.mod(maxmem,s_step)

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
    f.create_dataset('C',data=p_dict['C'],dtype=datype)
    f.create_dataset('D',data=p_dict['D'],dtype=datype)
    f.create_dataset('V',shape=(N,n_elems),chunks=chunks,dtype=datype)
    f.create_dataset('Z',shape=(N,n_elems),chunks=chunks,dtype=datype)
    f.create_dataset('DA',shape=(N,n_elems),chunks=chunks,dtype=datype)
    f.create_dataset('QV',shape=(N,n_elems),chunks=chunks,dtype=datype)
    f.create_dataset('Beta',shape=(N,n_elems),chunks=chunks,dtype=datype)
    f.create_dataset('t',data=np.linspace(tstart,tend,n_elems),dtype=datype)

    # Save parameters (but exclude stuff imported in the parameter file)
    pg = f.create_group('params')
    for key,value in p_dict.items():
        valuetype  = type(value).__name__
        if valuetype != 'instance' and valuetype != 'module' and valuetype != 'function':
            pg.create_dataset(key,data=value)
    
    # Close container and write to disk
    f.close()

    # Initialize parameter C-class (struct) for the model
    params = par(p_dict)

    # Concatenate initial conditions for the "calibration" run 
    VZD0 = np.hstack([V0.squeeze(),Z0.squeeze(),DA0.squeeze()])

    # Set up parameters for an initial `len_init` (in ms) long resting state simulation to "calibrate" the model
    len_init = 100
    dt       = 0.1
    s_step   = 10 
    rmax     = np.zeros((N,))
    tinit    = np.arange(0,len_init,dt)
    tsize    = tinit.size
    csize    = int(np.ceil(tsize/s_step))

    # Update `p_dict` (we don't use it anymore, so just overwrite stuff) 
    p_dict['dt']     = dt
    p_dict['s_step'] = s_step
    p_dict['rmax']   = rmax
    parinit          = par(p_dict)

    # Create a temporary container for the simulation
    tmpname = tempfile.mktemp() + '.h5'
    tmpfile = h5py.File(tmpname)
    tmpfile.create_dataset('V',shape=(N,csize),dtype=datype)
    tmpfile.create_dataset('Z',shape=(N,csize),dtype=datype)
    tmpfile.create_dataset('DA',shape=(N,csize),dtype=datype)
    tmpfile.create_dataset('QV',shape=(N,csize),dtype=datype)
    tmpfile.create_dataset('Beta',shape=(N,csize),dtype=datype)
    tmpfile.flush()

    # Run 100ms of resting state to get model to a "steady state" for the initial conditions
    solve_model(VZD0,tinit,parinit,np.array([tsize]),np.array([csize]),seed,0,str(tmpfile.filename))

    # Use final values of `V`, `Z` and `DA` as initial conditions for the "real" simulation
    V0   = tmpfile['V'][:,-1]
    Z0   = tmpfile['Z'][:,-1]
    DA0  = tmpfile['DA'][:,-1]
    VZD0 = np.hstack([V0.squeeze(),Z0.squeeze(),DA0.squeeze()])

    # Close and delete the temporary container
    tmpfile.close()
    os.remove(tmpname)

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
                    ["matrix file:",mfile],\
                    ["output:",outfile]])
    if verbose: print "\n"+table.draw()+"\n"

    # Finally... run the actual simulation
    solve_model(VZD0,tsteps,params,blocksize,chunksize,seed,int(verbose),outfile)

    # Done!
    if verbose: print "\nDone\n"

##########################################################################################
def plot_sim(fname,names="all",raw=True,bold=False,figname=None):
    """
    Plot a simulation generated by `run_model`

    Parameters
    ----------
    fname : str
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
    figname : str
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

    # Make sure `fname` is a valid file-path
    f = checkhdf(fname,peek=True)

    # Make sure `raw` and `bold` are either `True` or `False`
    if not isinstance(raw,bool):
        raise TypeError("The flag `raw` has to be Boolean!")
    if not isinstance(bold,bool):
        raise TypeError("The flag `bold` has to be Boolean!")

    # Try to access given HDF5 container
    try:
        rois_infile = f['params']['names'].value.tolist()
        f.close()
    except:
        raise ValueError("HDF5 file "+fname+" does not have the required fields!")

    # Check if list of ROI-names to plot was provided. If yes, make sure they make sense
    if not isinstance(names,(str,unicode)):
        doleg = True
        try:
            names = list(names)
        except:
            raise TypeError("Regions to plot have to be provided as Python list or NumPy 1darray!")
        idx = []
        for name in names:
            try:
                idx.append(rois_infile.index(name))
            except:
                raise ValueError("Region "+name+"not found in file!")
    else:
        if names == "all":
            idx   = range(len(rois_infile))
            doleg = False
        else:
            try:
                idx = rois_infile.index(names)
            except:
                raise ValueError("Region "+names+"not found in file!")
            doleg = True

    # Check if `figname` is actually printable
    if figname != None:
        if not isinstance(figname,(str,unicode)):
            raise TypeError("Figure name has to be a string!"+type(figname).__name__+"!")
        figname = str(figname)

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
    vnames = ["target","source","names"]
    for tk, tsn in enumerate([target,source,names]):
        try:
            tsn = list(tsn)
        except: 
            msg = "Inputs target, source and names have to be NumPy 1darrays or Python lists, not "+\
                  type(tsn).__name__
            TypeError(msg)
        for el in tsn:
            if not isinstance(el,(str,unicode)):
                raise ValueError("All elements of `"+vnames[tk]+"` have to be strings!")

    if len(source) != len(target):
        raise ValueError("Length of source and target lists/arrays does not match up!")
    for tk, tsn in enumerate([target,source]):
        for el in tsn:
            if el not in names:
                raise ValueError("Element `"+el+"` not found in `"+vnames[tk]+"`!")

    if values != None:
        values = np.array(values)
        arrcheck(values,'vector','values')
        if len(values) != len(target):
            raise ValueError("Length of `values` list/array does not match up!")
    else:
        values = np.ones((len(target),))

    # Convert (if we're having a NumPy array) `names` to a Python list
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
    fname : str
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
    .. [1] Glover G. Deconvolution of Impulse Response in Event-Related BOLD FMRI. 
           NeuroImage 9: 416-429, 1999.

    .. [2] S. Fuertinger, J. C. Zinn, and K. Simonyan. A Neural Population Model Incorporating 
           Dopaminergic Neurotransmission during Complex Voluntary Behaviors. PLoS Computational Biology, 
           10(11), 2014. 
    """
    # Make sure container exists and is valid
    f = checkhdf(fname,peek=True)
    try:
        V = f['V'].value
    except: 
        f.close()
        raise ValueError("HDF5 file "+fname+" does not have the required fields!")

    # Compute cycle length based on the sampling rate used to generate the file
    N         = f['params']['names'].size
    s_rate    = f['params']['s_rate'].value
    n_cycles  = f['params']['n_cycles'].value
    len_cycle = f['params']['len_cycle'].value
    cycle_idx = int(np.round(s_rate*len_cycle))

    # Make sure `stim_onset` makes sense
    if stim_onset != None:
        scalarcheck(stim_onset,'stim_onset',bounds=[0,len_cycle])

    # Get task from file to start sub-sampling procedure
    task = f['params']['task'].value

    # Compute cycle length based on the sampling rate used to generate the file
    N         = f['params']['names'].size
    n_cycles  = f['params']['n_cycles'].value
    len_cycle = f['params']['len_cycle'].value
    cycle_idx = int(np.round(s_rate*len_cycle))

    # Compute step size and (if not provided by the user) compute stimulus onset time
    dt = 1/s_rate
    if stim_onset == None:
        stim_onset = f['params']['stimulus'].value
    
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
    fname : str
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

    # Make sure container exists and is valid
    f = checkhdf(fname,peek=True)
    try:
        par_grp = f['params']
        f.close()
    except:
        raise ValueError("HDF5 file "+fname+" does not have the required fields!")

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
def arrcheck(arr,kind,varname,bounds=None):
    """
    Local helper function performing sanity checks on arrays (1d/2d/3d)
    """
    
    try:
        sha = arr.squeeze().shape
    except:
        raise TypeError('Input '+varname+' must be a NumPy array, not '+type(arr).__name__+'!')

    if kind == 'tensor':
        if len(sha) != 3:
            raise ValueError('Input `'+varname+'` must be a `N`-by-`N`-by-`k` NumPy array')
        if (min(sha[0],sha[1])==1) or (sha[0]!=sha[1]):
            raise ValueError('Input `'+varname+'` must be a `N`-by-`N`-by-`k` NumPy array!')
        dim_msg = '`N`-by-`N`-by-`k`'
    elif kind == 'matrix':
        if len(sha) != 2:
            raise ValueError('Input `'+varname+'` must be a `N`-by-`N` NumPy array')
        if (min(sha)==1) or (sha[0]!=sha[1]):
            raise ValueError('Input `'+varname+'` must be a `N`-by-`N` NumPy array!')
        dim_msg = '`N`-by-`N`'
    elif kind == 'vector':
        if len(sha) != 1:
            raise ValueError('Input `'+varname+'` must be a NumPy 1darray')
        if min(sha)==1:
            raise ValueError('Input `'+varname+'` must be a NumPy 1darray of length `N`!')
        dim_msg = ''
    else:
        print "Error checking could not be performed - something's wrong here..."
    if not plt.is_numlike(arr) or not np.isreal(arr).all():
        raise TypeError('Input `'+varname+'` must be a real-valued '+dim_msg+' NumPy array!')
    if np.isfinite(arr).min() == False:
        raise ValueError('Input `'+varname+'` must be a real valued NumPy array without Infs or NaNs!')
        
    if bounds is not None:
        if arr.min() < bounds[0] or arr.max() > bounds[1]:
            raise ValueError("Values of input array `"+varname+"` must be between "+str(bounds[0])+\
                             " and "+str(bounds[1])+"!")

##########################################################################################
def scalarcheck(val,varname,kind=None,bounds=None):
    """
    Local helper function performing sanity checks on scalars
    """

    if not np.isscalar(val) or not plt.is_numlike(val) or not np.isreal(val).all():
        raise TypeError("Input `"+varname+"` must be a real scalar!")
    if not np.isfinite(val):
        raise TypeError("Input `"+varname+"` must be finite!")

    if kind == 'int':
        if (round(val) != val):
            raise ValueError("Input `"+varname+"` must be an integer!")

    if bounds is not None:
        if val < bounds[0] or val > bounds[1]:
            raise ValueError("Input scalar `"+varname+"` must be between "+str(bounds[0])+" and "+str(bounds[1])+"!")

##########################################################################################
def checkhdf(fname,peek=False):
    """
    Local helper function performing sanity checks on file-names
    """

    # Make sure `fname` is a valid file-path
    if not isinstance(fname,(str,unicode)):
        raise TypeError("Name of HDF5 file has to be a string!")
    fname = str(fname)
    if fname.find("~") == 0:
        fname = os.path.expanduser('~') + fname[1:]
    if not os.path.isfile(fname):
        raise ValueError('File: '+fname+' does not exist!')

    # Additionally, try opening the container if wanted
    if peek:
        try:
            f = h5py.File(fname)
        except:
            raise ValueError("Cannot open "+fname+"!")
        return f
