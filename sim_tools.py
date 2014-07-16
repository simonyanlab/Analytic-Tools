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

# # Add the model's directoy to the path to be able to import it
# import sys
# sys.path.append(os.path.expanduser('~')+'/Documents/job/Joel/Model/New/')
from the_model import par, solve_model

def run_model(V0, Z0, DA0, task, outfile, C=None, seed = None, paramfile='parameters.py', symsyn=True):

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
        raise TypeError('Output filename has to be a string specifying the path to an HDF5 container!')

    if seed != None:
        try:
            bad = (np.round(seed) != seed)
        except: raise TypeError("Random number seed has to be an integer!")
        if bad: raise ValueError("Random number seed has to be an integer!")
        seed = int(seed)

    if type(paramfile).__name__ != 'str':
        raise TypeError('Parameter file has to be specified using a string!')

    if symsyn != True and symsyn != False:
        raise TypeError("The switch `symsyn` has to be boolean!")

    # Append '.h5' extension to outfile if necessary
    if outfile[-3:] != '.h5':
        outfile = outfile + '.h5'

    # Check if paramfile has an extension, if yes, rip it off
    if paramfile[-3:] == '.py':
        paramfile = paramfile[0:-3]

    # Divide paramfile into filename and path
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
        f = h5py.File(param_py.matrices)
    except: raise ValueError("Could not open HDF5 file holding the coupling matrix")
    try:
        c_str = "from argument"
        if C == None: C = f['C'].value; c_str = "from file"
        D     = f['D'].value
        names = f['names'].value
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

    # If wanted, make sure lef/right hemispheres have balanced coupling strenghts
    if symsyn:

        # Get inidices of left-hemispheric regions and throw a warning if left/right don't match up
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

    # Save updated coupling strengths in dictionary
    p_dict['aei'] = aei
    p_dict['aie'] = aie
    p_dict['ani'] = ani
    p_dict['ane'] = ane

    # If a resting state simulation is done, make sure dopamine doesn't kick in 
    if task == 'rest':
        b_lo = 1
        b_hi = 1
    else:
        b_lo = param_py.b_lo
        b_hi = param_py.b_hi

    # Save (possibly updated) beta limits and given task in dictionary
    p_dict['b_hi'] = b_hi
    p_dict['b_lo'] = b_lo
    p_dict['task'] = task

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
        print "WARNING: Using custom initial time of  "+str(tstart)+" (has to be in ms)!"
    else:
        tstart = 0

    # Set/get step-size for simulation 
    if p_dict.has_key('dt'): 
        dt = param_py.dt
        print "WARNING: Using custom step-size of "+str(dt)+" (has to be in ms)!"
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
    tend   = tstart + len_cycle*param_py.n_cycles*1000
    tsteps = np.arange(tstart,tend,dt)

    # Get the size of the time-array
    tsize   = tsteps.size

    # Start laying out output HDF5 container
    if os.path.isfile(outfile):
        newname = outfile[:-3] + "_bak_"+\
                  str(datetime.now().year)+"_"+\
                  str(datetime.now().month)+"_"+\
                  str(datetime.now().day)+"_"+\
                  str(datetime.now().hour)+"_"+\
                  str(datetime.now().minute)+"_"+\
                  str(datetime.now().second)+\
                  outfile[-3::]
        print "WARNING: file "+outfile+" already exists, renaming it to: "+newname+"!"
        os.rename(outfile,newname)
    
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
    # f.create_dataset('t',data=tsteps[::s_step])

    # Save parameters (but exclude stuff imported in the parameter file)
    pg = f.create_group('params')
    for key,value in p_dict.items():
        valuetype  = type(value).__name__
        if valuetype != 'instance' and valuetype != 'module' and valuetype != 'function':
            pg.create_dataset(key,data=value)
    pg.create_dataset('names',data=names)

    # Close container and write to disk
    f.close()

    # Now copy matrices also to p_dict and initialize the model's C-class with it
    p_dict['C'] = C
    p_dict['D'] = D

    # Initialize parameter C-class (struct) for the model
    params = par(p_dict)

    # Concatenate intial conditions
    VZD0 = np.hstack((np.hstack((V0.squeeze(),Z0.squeeze())),DA0.squeeze()))

    # Let the user know what's going to happen...
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l", "l"])
    table.add_rows([["Simulating ",task.upper()],\
                    ["#cycles: ",str(param_py.n_cycles)],\
                    ["parameter file:",paramfile+".py"],\
                    ["matrix file:",param_py.matrices],\
                    ["Coupling matrix:",c_str],\
                    ["Output:",outfile]])
    print "\n"+table.draw()+"\n"

    # Finally... Run the actual simulation
    VZD = solve_model(VZD0,tsteps,params,blocksize,chunksize,seed,outfile)

    # Done!
    print "\nDone\n"

def plot_sim(fname,names="all",raw=True,bold=False,figname=None):

    # Sanity checks
    if type(fname).__name__ != 'str':
        raise TypeError("Name of HDF5 file has to be a string!")

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
            idx = range(len(rois_infile))
        else:
            try:
                idx = rois_infile.index(names)
            except: raise ValueError("Region "+names+"not found in file!")

    if figname != None:
        if type(figname).__name__ != "str":
            raise TypeError("Figure name has to be a string, not "+type(figname).__name__+"!")

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
            BOLD = f['BOLD'].value 
        except: 
            f.close()
            raise ValueError("No BOLD data found in file "+fname+"!")

        # Get x-entent of data and create x-ticks vector
        xmax = BOLD.shape[1] + 1
        xtv  = np.arange(-1,xmax)

        # Prepare window and plot stuff
        fig = plt.figure(figsize=(10,7.5))
        if figname != None: fig.canvas.set_window_title(figname)
        plt.title("BOLD data "+fname,fontsize=18)
        sp = plt.subplot(111)
        plt.plot(BOLD.T)
        plt.xticks(xtv,fontsize=10)
        [ymin,ymax] = sp.get_ylim()
        plt.yticks([ymin,0,ymax],fontsize=10)

    # Close file and return
    f.close()

def make_D(target,source,names,values=None):

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

    if values != none:
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
    for i in len(source):

        # Get row and column indices
        row = names.index(target[i])
        col = names.index(source[i])

        # Matrix is targets-by-sources
        D[row,col] = values[i]

    return D

def make_bold(fname, stim_onset=None):

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

    # Get task from file to start subsampling procedure
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

    # t1 = np.arange(0,32+stim_onset,dt)
    # gamma = Psi*np.exp(-t1/theta)*t1**(alpha - 1.)
    # hrf_kernel = gamma.copy()
    # plt.figure();plt.plot(gamma)

    # t1 = np.arange(0,32+stim_onset,dt)
    # plt.ion();plt.figure();plt.plot(t1,hrf_kernel)
    # plt.figure();plt.plot(gamma)

    # V  = (V.T - V.mean(axis=1)).T
    # qq = np.abs(np.diff(V,axis=1))
    # convV = ndimage.filters.convolve1d(V,hrf_kernel,mode='constant')
    # convV = ndimage.filters.convolve1d((qq.T - qq.mean(axis=1)).T,hrf_kernel,mode='constant')

    # Convolve the de-meaned model time-series with the kernel 
    convV = ndimage.filters.convolve1d((V.T - V.mean(axis=1)).T,hrf_kernel,mode='constant')

    # Allocate space for BOLD signal
    BOLD = np.zeros((N,n_cycles))

    # Subsample convoluted data depending on task to get BOLD signal
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
    # BOLD = (BOLD.T - BOLD.mean(axis=1)).T

    # Save it to the file
    try:
        f.create_dataset('BOLD',data=BOLD)
    except:
        f['BOLD'].write_direct(BOLD)
    f.close()


