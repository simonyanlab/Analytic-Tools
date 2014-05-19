# eeg_tools.py - Toolset to read/write EEG data
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# March 19 2014

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
import fnmatch
import os
import calendar
import csv
import h5py
import psutil
from scipy.signal import buttord, butter, lfilter, filtfilt

##########################################################################################
def bandpass_filter(signal,locut,hicut,srate,offset=None,passdB=1.0,stopdB=30.0):
    """
    Band-/Low-/High-pass filter a 1D/2D input signal (based on Hz!!!)

    Lowpass: remove high frequencies
    Highpass: remove low frequencies
    """

    # Sanity checks: signal
    try:
        stu = signal.shape
    except:
        raise TypeError('Signal must be a 1d/2d NumPy array, not '+type(signal).__name__+'!')
    if len(stu) > 2:
        raise ValueError('Signal must be a 1d/2d NumPy array')
    if max(stu) == 1:
        raise ValueError('Signal only consists of one datapoint!')
    if np.isnan(signal).max()==True or np.isinf(signal).max()==True or np.isreal(signal).min()==False:
        raise ValueError('Signal must be a real valued NumPy array without Infs or NaNs!')

    # Both cutoffs undefined
    if (locut == None) and (hicut == None):
        raise ValueError('Both cutoff frequencies are None!')

    # Sampling rate
    try:
        bad = (srate <= 0)
    except: 
        raise TypeError('Sampling rate has to be a strictly positive number, not '+type(srate).__name__+'!')
    if bad: raise ValueError('Sampling rate hast to be > 0!')

    # Compute Nyquist frequency and initialize passfreq
    nyq      = 0.5 * srate 
    passfreq = None

    # Lower cutoff frequency
    if locut != None:
        try:
            bad = (locut < 0)
        except: 
            raise TypeError('Locut has to be None or lower cutoff frequency, not '+type(locut).__name__+'!')
        if bad: raise ValueError('Locut frequency has to be >= 0!')
    else:
        passfreq = hicut/nyq

    # Higher cutoff frequency
    if hicut != None:
        try:
            bad = (hicut < 0)
        except: 
            raise TypeError('Hicut has to be None or higher cutoff frequency, not '+type(hicut).__name__+'!')
        if bad: raise ValueError('Hicut frequency has to be >= 0!')
    else:
        passfreq = locut/nyq

    # Offset frequency for filter
    if offset != None:
        try:
            bad = (offset <= 0)
        except:
            raise TypeError('Frequency offset has to be a strictly positive number, not '+type(offset).__name__+'!')
        if bad: raise ValueError('Frequency offset has to be > 0!')

        # Adjust offset for Nyquist frequency
        offset /= nyq

    else:

        # If no offset frequency was provided, assign default value (for low-/high-pass filters)
        if passfreq != None: offset = 0.2*passfreq

    # Passband decibel value
    if passdB != 1.0:
        try: 
            bad = (passdB <= 0)
        except:
            raise TypeError('Passband dB has to be a strictly positive number, not '+type(passdB).__name__+'!')
        if bad: raise ValueError('Passband dB has to be > 0!')

    # Stopband decibel value
    if stopdB != 0.5:
        try: 
            bad = (stopdB <= 0)
        except:
            raise TypeError('Stopband dB has to be a strictly positive number, not '+type(stopdB).__name__+'!')
        if bad: raise ValueError('Stopband dB has to be > 0!')

    # Determine if we do low-/high-/bandpass-filtering
    if locut == None:
        ftype    = 'highpass'
        stopfreq = passfreq - offset
        if stopfreq > passfreq: raise ValueError('Highpass stopfrequency is higher than passfrequency!')
        if passfreq >= 1: raise ValueError('Highpass frequency >= Nyquist frequency!')
    elif hicut == None:
        ftype    = 'lowpass'
        stopfreq = passfreq + offset
        if stopfreq < passfreq: raise ValueError('Lowpass stopfrequency is lower  than passfrequency!')
        if stopfreq >= 1: raise ValueError('Lowpass stop frequency >= Nyquist frequency!')
    else:
        ftype    = 'bandpass'
        passfreq = [locut/nyq,hicut/nyq]
        if offset == None: offset = 0.2*(passfreq[1] - passfreq[0])
        stopfreq = [passfreq[0] - offset, passfreq[1] + offset]
        if (stopfreq[0] > passfreq[0]) or (stopfreq[1] < passfreq[1]):
            raise ValueError('Stopband is inside the passband!')
        if stopfreq[1] >= 1: raise ValueError('Highpass frequency = Nyquist frequency!')

    # Compute optimal order of Butterworth filter
    order, natfreq = buttord(passfreq, stopfreq, passdB, stopdB)

    # import ipdb;ipdb.set_trace()

    # Compute Butterworth filter coefficients
    b,a = butter(order,natfreq,btype=ftype)

    # Filter data
    filtered = filtfilt(b,a,signal)
    # filtered = lfilter(b,a,signal)
    
    return filtered


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
def bcd(int_in):
    """
    Function used internally by read_eeg to convert unsigned integers to binary format and back again
    """
    int_in = "{0:08b}".format(int(int_in))
    return 10*int(int_in[0:4],2)+int(int_in[4:],2)

##########################################################################################
def read_eeg(eegpath,outfile,electrodelist=None,savemat=True):
    """
    Read raw EEG data from binary *.EEG/*.eeg and *.21E files

    Parameters
    ----------
    eegpath : string
        Path to the directory holding the EEG and 21E files. WARNING: the directory MUST only 
        contain ONE EEG and its corresponding 21E file!
    outfile : string
        Path specifying the HDF5 file to be created. WARNING: File MUST NOT exist!
    electrodelist : list
        Python list of strings holding names of electrodes to be saved (if not the entire 
        EEG file is needed/wanted). By default the entire EEG file is converted to HDF5. 
    savemat : bool
        Specifiy if data should be saved as a NumPy 2darray (format is: number of electrodes by 
        number of samples) or by electrodenames (see Examples for details)

    Returns
    -------
    Nothing : None

    Notes
    -----
    Depending on the value of `savemat` the HDF5 file structure will differ. The HDF5 file always
    contains the groups `EEG` and `info`. In `EEG` the raw data is stored either as NumPy 2darray 
    (`savemat = True`) or sorted by electrode name (`savemat = False`). The `info` group holds 
    metadata of the EEG scan (record date, sampling rate, session length etc.). 
    Note: The code allocates 25% of RAM available on the machine to temporarily hold the EEG data. 
    Thus, reading/writing may take longer on computers with small memory. 

    Examples
    --------
    Suppose the files `test.eeg` and `test.21E` are in the directory `mytest`. Suppose further 
    that the EEG file contains recordings of 84 electrodes and the output HDF5 container 
    should be `Results/test.h5`. If the entire EEG file should be piped into the HDF5 file 
    as a matrix then, `cd` to the parent directory of `mytest` and type

    >>> read_eeg('mytest','Results/test.h5')

    The resulting HDF5 file has groups `EEG` and `info`. The `info` group holds metadata 
    (see Notes for details) while the EEG time-courses can be found in `EEG`:

    >> f = h5py.File('Results/test.h5')
    >> f['EEG'].keys()
    >> ['eeg_mat', 'electrode_list']
    
    The dataset `eeg_mat` holds the entire EEG dataset as matrix (NumPy 2darray),  

    >> f['EEG']['eeg_mat'].value
    >> array([[32079, 32045, 32001, ..., 33607, 33556, 33530],
              [31708, 31712, 31712, ..., 33607, 33597, 33599],
              [31719, 31722, 31704, ..., 33733, 33713, 33708],
              ..., 
              [39749, 34844, 36671, ..., 44616, 43642, 41030],
              [30206, 28126, 30805, ..., 39691, 36586, 34550],
              [31084, 30167, 31580, ..., 38113, 36470, 35205]], dtype=uint16)

    where `electrode_list` is a NumPy array of electrodenames corresponding to the rows of `eeg_mat`, 
    i.e., `f['EEG']['eeg_mat'][23,:]` is the time-series of electrode `f['EEG']['electrode_list'][23]`

    >> f['EEG']['electrode_list'][23]
    >> 'RFC8'
    >> f['EEG']['eeg_mat'][23,:]
    >> array([33602, 33593, 33649, ..., 32626, 32648, 32650], dtype=uint16)

    If only the electrodes 'RFA1' and 'RFA3' are of interest and the read-out should be saved
    by the respective electrode names then the following command could be used

    >> read_eeg('mytest','Results/test.h5',electrodelist=['RFA1','RFA3'],savemat=False)

    In this case the `EEG` group of the resulting HDF5 file looks like this

    >> f = h5py.File('Results/test.h5')
    >> f['EEG'].keys()
    >> ['RFA1', 'RFA3']
    >> f['EEG']['RFA1'].value
    >> array([32079, 32045, 32001, ..., 33607, 33556, 33530], dtype=uint16)

    Thus, the electrode time-courses are saved using the respective electrode names. 

    See also:
    ---------
    None
    """

    # Sanity checks
    if type(eegpath).__name__ != 'str':
        raise TypeError('Input has to be a string specifying the path to the EEG files!')

    if type(outfile).__name__ != 'str':
        raise TypeError('Output filename has to be a string!')
    if os.path.isfile(outfile): 
        raise ValueError("Target HDF5 container already exists!")

    if electrodelist != None: 
        try: le = len(electrodelist)
        except: raise TypeError('Input electrodlist must be a Python list!')
        if le == 0: raise ValueError('Input electrodelist has length 0!')

    if type(savemat).__name__ != 'bool': raise TypeError('Input savemat has to be boolean!')

    # If file extension was provided, remove it to avoid stupid case-sensitive nonsense
    dt = eegpath.rfind('.')
    if eegpath[dt+1:].lower() == 'eeg':
        eegpath = eegpath[0:dt]

    # Extract filename from given path (if just file was provided, path is '')
    slash  = eegpath.rfind(os.sep)
    flpath = eegpath[0:slash+1]
    flname = eegpath[slash+1:]

    # Try to get eeg file and raise an error if it does not exist or an x.eeg and x.EEG file is found
    eegfls = myglob(flpath,flname+'.[Ee][Ee][Gg]')
    if len(eegfls) > 1: 
        raise ValueError('Filename ambiguity: found '+str(eegfls))
    elif len(eegfls) == 0:
        if flpath == '': flpath = 'current directory'
        raise ValueError('File '+flname+'.EEG/eeg not found in '+flpath)

    # Same for (hopefully) corresponding 21E file
    e21fls = myglob(flpath,flname+'.21[Ee]')
    if len(e21fls) > 1: 
        raise ValueError('Filename ambiguity: found '+str(e21fls))
    elif len(e21fls) == 0:
        if flpath == '': flpath = 'current directory'
        raise ValueError('File '+flname+'.21E/21e not found in '+flpath)

    # Open file handles to *.EEG, *.21E and output files
    fh = open(e21fls[0],'rU')
    fh.seek(0)
    fid = open(eegfls[0])
    f   = h5py.File(outfile)

    # Let the user know what's going on
    print "\n Starting reading routine..."
    print "\n Successfully accessed files "+eegfls[0]+", "+e21fls[0]+" and "+f.filename+"\n"

    # Try to import progressbar module
    try: 
        import progressbar as pb
        showbar = True
    except: 
        print "WARNING: progressbar module not found - consider installing it using pip install progressbar"
        showbar = False

    # Skip EEG device block
    deviceBlockLen = 128
    fid.seek(deviceBlockLen)

    # Read EEG1 control Block (contains names and addresses for EEG2 blocks)
    x              = np.fromfile(fid,dtype='uint8',count=1)
    x              = np.fromfile(fid,dtype='S1',count=16)
    numberOfBlocks = np.fromfile(fid,dtype='uint8',count=1)
    blockAddress   = np.fromfile(fid,dtype='int32',count=1)
    x              = np.fromfile(fid,dtype='S1',count=16)

    # Read EEG2m control block (contains names and addresses for waveform blocks)
    fid.seek(int(blockAddress),0)
    x              = np.fromfile(fid,dtype='uint8',count=1)
    x              = np.fromfile(fid,dtype='S1',count=16)
    numberOfBlocks = np.fromfile(fid,dtype='uint8',count=1)
    blockAddress   = np.fromfile(fid,dtype='int32',count=1)
    x              = np.fromfile(fid,dtype='S1',count=16)

    # Read waveform blockA
    fid.seek(int(blockAddress),0)
    x = np.fromfile(fid,dtype='uint8',count=1)
    x = np.fromfile(fid,dtype='S1',count=16)
    x = np.fromfile(fid,dtype='uint8',count=1)

    # Get data byte-length and mark/event flag
    L = np.fromfile(fid,dtype='uint8',count=1)
    M = np.fromfile(fid,dtype='uint8',count=1)

    # Get starting time 
    T_year   = bcd(int(np.fromfile(fid,dtype='uint8',count=1)))
    T_month  = bcd(int(np.fromfile(fid,dtype='uint8',count=1)))
    T_day    = bcd(int(np.fromfile(fid,dtype='uint8',count=1)))
    T_hour   = bcd(int(np.fromfile(fid,dtype='uint8',count=1)))
    T_minute = bcd(int(np.fromfile(fid,dtype='uint8',count=1)))
    T_second = bcd(int(np.fromfile(fid,dtype='uint8',count=1)))

    # Expand T_year to full format and account for millenium (i.e., 13 -> 2013, 96 -> 1996)
    if T_year < 30:
        T_year = 2000 + T_year
    else:
        T_year = 1900 + T_year

    # Print time-stamp info
    weeklist    = [day for day in calendar.day_name]
    monthlist   = [month for month in calendar.month_name]; monthlist.pop(0)
    recordedstr = " Data was recorded on "+weeklist[date(T_year,T_month,T_day).weekday()]+\
        ", "+monthlist[T_month-1]+" "+str(T_day)+" "+str(T_year)
    print recordedstr
    beginstr = " Begin of session: "+str(T_hour)+":"+str(T_minute)+":"+str(T_second)
    print beginstr

    # Get sampling rate
    hexopts = {int('C064',16):100,int('C068',16):200,int('C1F4',16):500,\
               int('C3E8',16):1000,int('C7D0',16):2000,int('D388',16):5000,int('E710',16):10000}
    try: actSamplerate = hexopts[int(np.fromfile(fid,dtype='uint16',count=1))]
    except KeyError: print "ERROR: Unknown Sampling Rate"; sys.exit()
    sratestr = " Sampling rate: "+str(actSamplerate)+" Hz"
    print sratestr

    # Get length of scan
    num100msBlocks = int(np.fromfile(fid,dtype='uint32',count=1))
    lengthstr      =  " Length of session: "+str(num100msBlocks/10/3600)+" hours"
    print lengthstr

    # More scanning parameters
    numSamples  = int(actSamplerate*num100msBlocks/10)
    AD_off      = np.fromfile(fid,dtype='int16',count=1)[0]
    AD_val      = int(np.fromfile(fid,dtype='uint16',count=1))
    bitLen      = int(np.fromfile(fid,dtype='uint8',count=1))
    comFlag     = int(np.fromfile(fid,dtype='uint8',count=1))
    numChannels = int(np.fromfile(fid,dtype='uint8',count=1))

    # Read electrode codes and names using csv module
    reader       = csv.reader(fh, delimiter='=', quotechar='"')
    allCodeNames = {}
    i = 0
    for row in reader:
        if len(row) == 2:
            allCodeNames[int(row[0])] = row[1]
        else:
            if row[0] == '[SD_DEF]':
                break

    # Define good electrode codes and bad electrode names
    goodCodes = list(np.hstack((np.arange(0,37),74,75,np.arange(100,254))))
    badNames  = ['E']

    # Build list of actually used electrodes in this file and their corresponding indices
    actualNames = []
    CALopts     = {0:1000,1:2,2:5,3:10,4:20,5:50,6:100,7:200,8:500,9:1000}
    GAIN        = np.zeros((numChannels,))
    goodElec    = np.zeros((numChannels,),dtype='bool')
    for i in xrange(numChannels):
        x          = np.fromfile(fid,dtype='int16',count=1)[0]
        ActualName = allCodeNames[x]
        if goodCodes.count(x) == 0 or badNames.count(ActualName) == 1:
            goodElec[i] = False
            actualNames.append('###')
        else:
            goodElec[i] = True 
            actualNames.append(ActualName)

        # Skip 6 bits starting at current position
        fid.seek(6,1)

        # Read channel sensitivity and determine CAL in microvolts
        chan_sens = np.fromfile(fid,dtype='uint8',count=1)
        CAL       = CALopts[int(np.fromfile(fid,dtype='uint8',count=1))]
        GAIN[i]   = CAL/AD_val

    # Abort if channels show difference in gain
    if np.unique(GAIN).size != 1: raise ValueError("Channels do not have the same gain!")

    # If user provided list of electrodes to read check it now
    if electrodelist != None:
        idxlist = []
        for el in electrodelist:
            try:
                idx = actualNames.index(el)
            except:
                raise IndexError('Electrode '+el+' not found in file!')
            if goodElec[idx]:
                idxlist.append(idx)
            else: print "WARNING: Electrode "+el+" not in trusted electrode list! Skipping it..."

        # In case the electrodlist was not ordered as the binary file, fix this 
        idxlist.sort()

        # Synchronize goodElec and electrodelist
        goodElec[:]       = False 
        goodElec[idxlist] = True

    # The indexlist is the whole "good" goodElec array
    else:
        idxlist = np.nonzero(goodElec)[0].tolist()

    # The data type of the raw data is unsigned integer. Define that and the bytesize of uint16 here
    dt = np.dtype('uint16')
    ds = dt.itemsize

    # Create a group holding the raw data
    eeg = f.create_group('EEG')

    # Depending on available memory, allocate temporary matrix
    meminfo = psutil.virtual_memory()
    maxmem  = meminfo.available*0.25/(numChannels+1)/ds

    # If the whole array fits into memory load it once, otherwise chunk it up
    if numSamples <= maxmem:
        blocksize = [numSamples]
    else:
        blocksize = [maxmem]*(numSamples//maxmem) + [int(np.mod(numSamples,maxmem))]

    # Count the number of blocks we split up data into
    numblocks = len(blocksize)

    # Allocate matrix to temporarily hold data
    datamat = np.zeros((numChannels+1,blocksize[0]),dtype='int16')

    # Depending on the user wanting to save stuff as matrix, prepare dataset 
    numnodes = goodElec.sum()
    if (savemat): 
        nodelist = []
        for i in xrange(goodElec.size): 
            if goodElec[i]: 
                nodelist.append(actualNames[i])
        eeg.create_dataset('electrode_list',data=nodelist)
        eeg_mat = eeg.create_dataset('eeg_mat',shape=(numnodes,numSamples),chunks=(1,numSamples),dtype='int16')
    else:
        for idx in idxlist:
            eeg.create_dataset(actualNames[idx],shape=(numSamples,),dtype='int16')

    # If available, initialize progressbar
    if (showbar): 
        widgets = ['Processing data block-wise... ',pb.Percentage(),' ',pb.Bar(marker='#'),' ',pb.ETA()]
        pbar    = pb.ProgressBar(widgets=widgets,maxval=numblocks)

    # Here we go...
    print "\n Reading data in "+str(numblocks)+" block(s)...\n "
    if (showbar): pbar.start()

    # Read/write data block by block
    j = 0
    for i in xrange(numblocks):

        # Read data block-wise and save to matrix or row (depending on user choice)
        bsize   = blocksize[i]
        datamat = np.fromfile(fid,dtype='uint16',count=bsize*(numChannels+1)).reshape((numChannels+1,bsize),order='F') + AD_off
        if (savemat):
            eeg_mat[:,j:j+bsize] = datamat[idxlist,0:bsize]
        else:
            for idx in idxlist:
                f['EEG'][actualNames[idx]][j:j+bsize] = datamat[idx,0:bsize]

        # Update index counter 
        j += bsize

        # Update progressbar
        if (showbar): 
            widgets[0] = ' Processing block '+str(i+1)+'/'+str(numblocks)+' '
            pbar.update(i)

    # If progressbar is available, end it now
    if (showbar): pbar.finish()

    # Write meta-data
    info = f.create_group('info')

    # Write a human-readable text block
    info.create_dataset('summary',data=recordedstr+"\n"+beginstr+"\n"+sratestr+"\n"+lengthstr)

    # Write scanning parameters individually 
    info.create_dataset('record_date',data=[T_year,T_month,T_day,T_hour,T_minute,T_second])
    info.create_dataset('sampling_rate',data=actSamplerate)
    info.create_dataset('session_length',data=num100msBlocks/10/3600)
    info.create_dataset('AD_off',data=AD_off)
    info.create_dataset('AD_val',data=AD_val)
    info.create_dataset('CAL',data=CAL)

    # Close and finalize HDF write process
    f.close()
    print " Done. "

    return

##########################################################################################
def catchinput(tklist,wndw):
    """
    Event handler to process user input
    """

    valuelist = [tv.get() for tv in tklist]
    if valuelist.count(1) == 0:
        errormsg = "Please make a choice to proceed!"
        tkMessageBox.showerror(title="Invalid Choice",message=errormsg)
    else:
        wndw.quit()

##########################################################################################
def plot_eeg(h5file=None,electrodelist=None):
    """
    Plots EEG time-courses stored in an HDF5 file that was generated with read_eeg

    Parameters
    ----------
    h5file : string
        Path to the HDF5 file holding the EEG data. If not provided, a GUI window
        will be opened for the user to specify a valid HDF5 file. 
    electrodelist : list
        A Python list holding the names of electrodes to be plotted. If not provided, 
        the user can choose electrodes in a GUI

    Returns
    -------
    Nothing : None

    Notes
    -----
    This routine is a convenience function to visualize data stored in a previously generated
    HDF5 file. This is *not* a general purpose plotting routine. Note that this plotting
    function automatically detects if the HDF5 container was generated with the `savemat`
    flag set or not (see `read_eeg` for details). 

    Examples
    --------
    Suppose we want to plot the time-courses of electrodes `A` and `B` stored in the HDF5 container 
    `myfile.h5`. Then (assuming `myfile.h5` is in the current directory) the following call 
    can be used to visualize the data
    
    >>> plot_eeg(h5file='myfile.h5',electrodelist=['A','B'])

    Alternatively, the call

    >>> plot_eeg()

    opens a GUI where the HDF5 file and corresponding electrodes can be selected. 

    See also
    --------
    read_eeg : routine to convert EEG data from binary to HDF5 format
    """

    # If no or few inputs were given, use TK dialogues 
    if h5file == None or electrodelist == None:
        USEtk = True
    else:
        USEtk = False

    # Try to import TK and exit if it is needed but cannot be imported
    if USEtk == True:
        try:
            import Tkinter as tk
            import tkFileDialog
            import tkMessageBox
        except:
            msg = "ERROR: Tk seems not to be installed - you have to provide the HDF5 filename and" + \
                  "electrodelist in the command line!"
            raise ImportError(msg)

    # Create main Tk-window and hide it immidiately
    if USEtk:
        try:
            root = tk.Tk()
            root.withdraw()
        except:
            print("ERROR: problem opening Tk root window, exiting... ")

    # Sanity checks
    if h5file != None:
        if type(h5file).__name__ != 'str':
            raise TypeError('Input has to be a string specifying the path to the HDF5 file!')
    else:
        h5file = tkFileDialog.askopenfilename(title='Please choose a valid HDF5 file')

    # Try opening the file
    try:
        f = h5py.File(h5file)
    except: 
        errormsg = "Error opening file "+h5file
        if USEtk: 
            tkMessageBox.showerror(title="Invalid File",message=errormsg)
        else:
            print(errormsg)

    # Determine if the given HDF file has the correct structure
    try:
        ismat = (f['EEG'].keys().count('eeg_mat') > 0)
    except: 
        errormsg = 'Invalid input file: '+h5file
        if USEtk:
            tkMessageBox.showerror(title="Invalid File",message=errormsg)
        else:
            raise TypeError(errormsg)

    # Get list of electrodes actually present in file
    if (ismat):
        ec_list = f['EEG']['electrode_list'].value.tolist()
    else:
        ec_list = f['EEG'].keys()

    # Get electrodes for plotting
    if electrodelist != None: 
        try: le = len(electrodelist)
        except: raise TypeError('Input electrodlist must be a Python list!')
        if le == 0: raise ValueError('Input electrodelist has length 0!')
        for el in electrodelist:
            if ec_list.count(el) == 0:
                raise ValueError('Electrode '+el+' not present in file '+h5file)
        plotlist = electrodelist
    else:
        # Make main Tk-window visible again
        root.wm_deiconify()

        # Give it a title and add some text
        root.title("Please choose electrodes for plotting")
        msg = "Choose which electrodes should be plotted"
        tk.Label(root,text=msg).grid(row=0,columnspan=4,sticky=tk.N,padx=15,pady=10)

        # # Create a frame in the main window
        # myframe = tk.Frame(root)
        # myframe.pack(fill=tk.X)

        # Create a list of Tk-integers representing the on/off states of the checkboxes below
        tklist    = []
        i         = 0
        numchecks = np.ceil(len(ec_list)/4)
        collist   = [0]*numchecks + [1]*numchecks + [2]*numchecks + [3]*numchecks
        rowlist   = range(1,int(numchecks)+1)*4
        for el in ec_list:
            tkvar = tk.IntVar(root)
            tk.Checkbutton(root,text=el,variable=tkvar).grid(row=rowlist[i],column=collist[i],sticky=tk.W,padx=5)
            tklist.append(tkvar)
            i += 1

        # Create an "OK" button to "finalize" the choice (if no choice was made warn the user)
        tk.Button(root,text="OK",command=lambda: catchinput(tklist,root)).grid(row=int(numchecks+2),columnspan=4,pady=5)

        # Everything's set up, start the Tk-mainloop that runs until the user presses "OK"
        root.mainloop()
        root.withdraw()

        # Get names of electrodes for plotting
        plotlist = []
        for i in xrange(len(tklist)):
            if tklist[i].get():
                plotlist.append(ec_list[i])

    # Based on session length and sampling rate compute sampling size for plot (nobody needs 9 mio time points)
    sr  = f['info']['sampling_rate'].value 
    sl  = f['info']['session_length'].value
    psr = (sl > 24)*sr*59 + sr

    # Determine subplot layout
    numelec = len(plotlist)
    spcol   = (numelec > 10) + (numelec > 1) + 1
    sprow   = np.ceil(numelec/spcol)

    # Get vector for x-axis
    tmax = sl*3600*sr
    tlog = str(tmax)
    tlog = tlog[0:tlog.find('.')]
    tlog = tlog.count('0')
    tvec = np.arange(0,tmax,psr)
    tunt = '*1e'+str(tlog)
    tnrm = 10**tlog
    
    # The xlabel
    xstring = 'time [msec]'

    # Set up figure
    plt.ion()
    fig = plt.figure(figsize=(12,10))
    fig.canvas.set_window_title('EEG Datafile: '+h5file)

    # Start plotting stuff
    plots = []
    for i in xrange(numelec):
        el = plotlist[i]
        plots.append(plt.subplot(sprow,spcol,i+1))
        if (ismat):
            j = ec_list.index(el)
            plt.plot(tvec,f['EEG']['eeg_mat'][j,::psr],'k')
        else:
            plt.plot(tvec,f['EEG'][el][::psr],'k')
        plt.title(el,fontsize=10)
        if i >= (numelec - spcol):
            plt.xlabel(xstring,fontsize=10)

    # Equalize ordinates and get ticking right
    ymin = 1e15; ymax = 0
    for p in plots:
        ymin = min(ymin,p.get_ylim()[0])
        ymax = max(ymax,p.get_ylim()[1])

    # Get number of decimals on y-axis
    ylog = str(ymin)
    ylog = ylog[0:ylog.find('.')]
    ylog = ylog.count('0')
    ynrm = 10**max(3,ylog)
    yunt = '*1e'+str(ylog)

    # Use scientific notation to shorten x- and y-ticks
    for p in plots:
        p.set_ylim(bottom=ymin,top=ymax)
        yt = p.get_yticks()
        yt = yt/ynrm
        if (np.round(yt)==yt).min() == True:
            yt = yt.astype('int')
        yl     = [str(y) for y in yt]
        yl[-1] = yl[-1]+yunt
        p.set_yticklabels(yl,fontsize=8)
        xt = p.get_xticks()
        p.set_xticks(xt[1::])
        xt     = xt[1::]/tnrm
        xl     = [str(t) for t in xt]
        xl[-1] = xl[-1]+tunt
        p.set_xticklabels(xl,fontsize=8)
        plt.draw()
