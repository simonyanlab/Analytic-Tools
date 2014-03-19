# eeg_tools.py - Toolset to read/write EEG data
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# March 19 2014

from __future__ import division
import numpy as np
import sys
from datetime import datetime, date
import fnmatch
import os
import calendar
import csv
import h5py
import psutil

##########################################################################################
def myglob(flpath,spattern):
    """
    Return a glob-like list of paths matching a pathname pattern BUT support fancy shell syntax

    Parameters
    ----------
    flpath : str
        Path to search
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

    # Append trailing slash to filepath
    if flpath[-1] != os.sep: flpath = flpath + os.sep

    # Return glob-like list
    return [os.path.join(flpath, fnm) for fnm in fnmatch.filter(os.listdir(flpath),spattern)]


##########################################################################################
def bcd(int_in):
    """
    Function used internally by read_eeg to convert unsigned integers to binaries and back again
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
    None

    Notes
    -----
    Depending on the value of `savemat` the HDF5 file structure will differ. The HDF5 file always
    contains the groups `EEG` and `info`. In `EEG` the raw data is stored either as NumPy 2darray 
    (`savemat = True`) or sorted by electrode name (`savemat = False`). The `info` group holds 
    metadata of the EEG scan (record date, sampling rate, session length etc.). 
    Note: The code allocates 25% of RAM available on the machine to temporarily hold the EEG data. 
    Thus, reading/writing may take longer on computers with few memory. 

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
    eegfls = myglob(eegpath,'*.[Ee][Ee][Gg]')
    if len(eegfls) > 1: raise ValueError('Expected 1 EEG file, found '+str(len(eegfls)))
    e21fls = myglob(eegpath,'*.21[Ee]')
    if len(e21fls) > 1: raise ValueError('Expected 1 21E file, found '+str(len(e21fls)))

    if type(outfile).__name__ != 'str':
        raise TypeError('Output filename has to be a string!')
    if os.path.isfile(outfile): 
        raise ValueError("Target HDF5 container already exists!")

    if electrodelist != None: 
        try: le = len(electrodelist)
        except: raise TypeError('Input electrodlist must be a Python list!')
        if le == 0: raise ValueError('Input electrodelist has length 0!')

    if type(savemat).__name__ != 'bool': raise TypeError('Input savemat has to be boolean!')

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
    AD_off      = int(np.fromfile(fid,dtype='uint16',count=1))
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

    # The data type is unsigned integer. Define that and the bytesize of uint16 here (change HERE if necessary!)
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
    datamat = np.zeros((numChannels+1,blocksize[0]),dtype=dt)

    # Depending on the user wanting to save stuff as matrix, prepare dataset 
    numnodes = goodElec.sum()
    if (savemat): 
        nodelist = []
        for i in xrange(goodElec.size): 
            if goodElec[i]: 
                nodelist.append(actualNames[i])
        eeg.create_dataset('electrode_list',data=nodelist)
        eeg_mat = eeg.create_dataset('eeg_mat',shape=(numnodes,numSamples),chunks=(1,numSamples),dtype=dt)
    else:
        for idx in idxlist:
            eeg.create_dataset(actualNames[idx],shape=(numSamples,),dtype=dt)

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
        datamat = np.fromfile(fid,dtype=dt,count=bsize*(numChannels+1)).reshape((numChannels+1,bsize),order='F')
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

    # Close and finalize HDF write process
    f.close()
    print " Done. "

    return
