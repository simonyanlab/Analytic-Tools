# eeg_tools.py - Toolset to read/write EEG data
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# Created: March 19 2014
# Last modified: <2015-05-21 14:29:10>

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date, datetime, timedelta
import fnmatch
import os
import calendar
import csv
import h5py
import psutil
from scipy.signal import buttord, butter, kaiserord, kaiser, lfilter, filtfilt, firwin
from scipy import ndimage

##########################################################################################
def bandpass_filter(signal,locut,hicut,srate,offset=None,passdB=1.0,stopdB=30.0,ftype='IIR'):
    """
    Band-/Low-/High-pass filter a 1D/2D input signal

    Parameters
    ----------
    signal : NumPy 1d/2darray
        Input signal to be filtered. For 2d arrays it is assumed that `signal` has shape `M`-by-`N`, 
        where `M` is the number of 1d signals (e.g., channels), and `N` is the number 
        of samples (measurements etc.). 
    locut : float
        Lower cutoff frequency in Hz. If `locut` is `None` then high-pass filtering 
        is performed. 
    hicut : float
        Upper cutoff frequency in Hz. If `hicut` is `None` then low-pass filtering 
        is performed. 
    srate : float 
        Sampling rate of the signal in Hz.
    offset : float
        Offset frequency in Hz. The frequency shift used to calculate the stopband (see 
        Notes for details). By default, the offset is a fraction of low-/high-cut 
        frequencies. 
    passdB : float
        Maximal frequency loss in the passband (in dB). For `ftype = FIR` (see below) 
        `passdB` has to be equals `stopdB`.
    stopdB : float
        Minimal frequency attentuation in the stopband (in dB). For `ftype = FIR` (see below) 
        `passdB` has to be equals `stopdB`.
    ftype : string
        Type of filter to be used (either `IIR` = infinite impulse response filter, or
        `FIR` = finite impulse response filter). 

    Returns
    -------
    filtered : NumPy 1d/2darray
        Filtered version of input `signal`. 

    Notes
    -----
    This routine uses a Butterworth filter (for `ftype = 'IIR'`) or a Kaiser filter
    (for `ftype = 'FIR'`) to low-/high-/bandpass the input signal. 
    Based on the user's input the optimal (i.e., lowest) order of the filter
    is calculated. Note that depending on the choice of cutoff frequencies and values 
    of `passdB` and `stopdB` the computed filter coefficients might be very large/low 
    causing numerical instability in the filtering routine. The code assumes you know
    what you're doing and does not try to guess whether the combination of 
    cutoff-frequencies, offset and attenuation/amplification values applied to the 
    input signal makes sense. 

    By default the offset frequency is computed as fraction of the input frequencies, 
    i.e., for low-/high-pass filters the offset is 0.5*cutoff-frequency, for band-pass
    filters the offset is calculated as 0.5 times the width of the pass-band. The following
    skteches illustrate the filter's operating modes


    Amplification 
    (dB)
    /\
    ||            Low-pass
    || ---------------------+
    ||                      |\
    ||                      | \
    ||                      |  +---------
    ||               PASS   |OS|   STOP
    || 
    ++===================================> Frequency (Hz)


    Amplification 
    (dB)
    /\
    ||            High-pass
    ||           +------------------------
    ||          /|	    
    ||         / |	    
    || -------+  |	    
    || STOP   |OS|   PASS  
    || 
    ++===================================> Frequency (Hz)


    Amplification 
    (dB)
    /\
    ||            Band-pass
    ||	         +----------+
    ||	        /|	    |\
    ||         / |	    | \
    || -------+  |	    |  +---------
    || STOP   |OS|   PASS   |OS|   STOP
    ||
    ++===================================> Frequency (Hz)


    Where `STOP` = stop-band, `OS` = offset, `PASS` = pass-band. 

    Examples
    --------
    We construct an artifical signal which we want to low-/high-/band-pass filter. 

    >>> import numpy as np
    >>> srate = 5000 # Sampling rate in Hz
    >>> T = 0.05
    >>> nsamples = T*srate
    >>> t = np.linspace(0,T,nsamples,endpoint=False)
    >>> a = 0.02
    >>> f0 = 600.0
    >>> signal = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    >>> signal += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    >>> signal += a * np.cos(2 * np.pi * f0 * t + .11)
    >>> signal += 0.03 * np.cos(2 * np.pi * 2000 * t)

    First, we low-pass filter the signal using the default IIR Butterworth filter (all
    examples given below can be repeated using the FIR Kaiser filter by additionally 
    providing the keyword argument `ftype='FIR'`). 
    As cutoff frequency we choose 50Hz, with an offset of 10Hz. That means frequencies 
    [0-50] Hz "survive", frequencies in the band [50-60] Hz are gradually attenuated, 
    all frequencies >60Hz are maximally attenuated.

    >>> filtered = bandpass_filter(signal,50,None,5000,offset=10)

    Now, construct a high-pass filter that removes all frequencies below 500Hz, using
    the default offset of 0.5*`hicut` (see Notes for details). 

    >>> filtered = bandpass_filter(signal,None,500,5000)

    Finally, we band-pass filter the signal, so that only frequency components between
    500Hz and 1250Hz remain

    >>> filtered = bandpass_filter(signal,500,1250,5000)
    
    Note that ill-chosen values for the offset (e.g., very steep slopes, from the 
    stop- to the pass-band, see Notes for a sketch) and/or attenuation/amplification 
    dB's may lead to very large/small filter coefficients that may cause erratic 
    results due to numerical instability. 

    See also
    --------
    scipy.signal.buttord : routine used to calculate optimal filter order
    scipy.signal.butter : routine used to construct Butterworth filter based on output of buttord. 
    scipy.signal.lfilter : filters the input signal using calculated Butterworth filter design
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

        # Multiplicator for offset
        offmult = 0.5

        # If no offset frequency was provided, assign default value (for low-/high-pass filters)
        if passfreq != None: offset = offmult*passfreq

    # Filter type
    try:
        bad = (ftype != 'IIR') and (ftype != 'FIR')
    except: raise TypeError('Filtertype has to be either FIR or IIR, not '+type(ftype).__name__+'!')
    if bad: raise ValueError('Filtertype has to be either FIR or IIR!')

    # Passband decibel value
    userpass = False
    if passdB != 1.0:
        try: 
            bad = (passdB <= 0)
        except:
            raise TypeError('Passband dB has to be a strictly positive number, not '+type(passdB).__name__+'!')
        if bad: raise ValueError('Passband dB has to be > 0!')
        userpass = True

    # Stopband decibel value
    if stopdB != 30:
        try: 
            bad = (stopdB <= 0)
        except:
            raise TypeError('Stopband dB has to be a strictly positive number, not '+type(stopdB).__name__+'!')
        if bad: raise ValueError('Stopband dB has to be > 0!')
        userpass = True

    # Since the Kaiser filter requires max/min ripple to be equal, make sure that condition is satisfied
    if ftype == 'FIR':
        if passdB != stopdB:

            # Take the maximum of the two dB values
            passdB = np.max([passdB,stopdB])
            stopdB = passdB

            # If the user supplied different dB values, print a warning
            if userpass: 
                msg = "WARNING: FIR filter requires stopdB = passdB, setting stopdB = passdB = "+str(passdB)
                print msg

    # Determine if we do low-/high-/bandpass-filtering
    if locut == None:
        ftype    = 'highpass'
        stopfreq = passfreq - offset
        if stopfreq > passfreq: raise ValueError('Highpass stopfrequency is higher than passfrequency!')
        if passfreq >= 1: raise ValueError('Highpass frequency >= Nyquist frequency!')
    elif hicut == None:
        ftype    = 'lowpass'
        stopfreq = passfreq + offset
        if stopfreq < passfreq: raise ValueError('Lowpass stopfrequency is lower than passfrequency!')
        if stopfreq >= 1: raise ValueError('Lowpass stop frequency >= Nyquist frequency!')
    else:
        ftype    = 'bandpass'
        passfreq = [locut/nyq,hicut/nyq]
        if offset == None: offset = offmult*(passfreq[1] - passfreq[0])
        stopfreq = [passfreq[0] - offset, passfreq[1] + offset]
        if (stopfreq[0] > passfreq[0]) or (stopfreq[1] < passfreq[1]):
            raise ValueError('Stopband is inside the passband!')
        if stopfreq[1] >= 1: raise ValueError('Highpass frequency = Nyquist frequency!')

    # Show input frequencies
    print "Input frequency/frequencies: "+str(locut)+"Hz, "+str(hicut)+"Hz"

    # Compute optimal order of filter
    if ftype == 'IIR':

        # Compute optimal order of Butterworth filter
        order, natfreq = buttord(passfreq, stopfreq, passdB, stopdB)
        
        # Show natural frequencies and optimal order of filter
        print "Natural frequency/frequencies: "+str(natfreq*nyq)+"Hz"
        print "Optimal order for Butterworth filter was found to be: "+str(order)

        # Compute Butterworth filter coefficients
        b,a = butter(order,natfreq,btype=ftype)

        # Filter data
        filtered = lfilter(b,a,signal)

    else:

        # Compute optimal order of Kaiser filter
        order, beta = kaiserord(passdB,offset)

        # Show optimal order
        print "Optimal order for Kaiser filter was found to be: "+str(order)

        # Compute Kaiser filter coefficients
        b = firwin(order,passfreq,window=('kaiser',beta),pass_zero=False)

        # Filter data
        filtered = filtfilt(b,[1.0],signal)
    
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

    # Make sure provided path is a string and does not contain weird unicode characters
    if str(flpath) != flpath:
        raise TypeError('Filepath has to be a string!')
    flpath = str(flpath)
    if flpath.find("~") == 0:
        flpath = os.path.expanduser('~') + flpath[1:]
    if str(spattern) != spattern:
        raise TypeError('Pattern has to be a string!')

    # If user wants to search current directory, make sure that works as expected
    if (flpath == '') or (flpath.count(' ') == len(flpath)):
        flpath = '.'

    # Append trailing slash to filepath
    else:
        if flpath[-1] != os.sep: flpath = flpath + os.sep

    # Now check if the path even exists
    if not os.path.isdir(flpath[:flpath.rfind(os.sep)]):
        raise ValueError('Invalid path: '+flpath+'!')

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
        Path to the *.EEG/*.eeg file (the code assumes that the corresponding *.21E/*.21e file is 
        in the same location)
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
    contain the groups `EEG`. In `EEG` the raw data is stored either as NumPy 2darray 
    (`savemat = True`) or sorted by electrode name (`savemat = False`). Metadata of the EEG scan 
    (record date, sampling rate, session length etc.) are stored as attributes of the `EEG` group. 
    Note: The code allocates 25% of RAM available on the machine to temporarily hold the EEG data. 
    Thus, reading/writing may take longer on computers with little memory. 

    Examples
    --------
    Suppose the files `test.eeg` and `test.21E` are in the directory `mytest`. Suppose further 
    that the EEG file contains recordings of 84 electrodes and the output HDF5 container 
    should be `Results/test.h5`. If the entire EEG file has to be converted to HDF5 
    as a matrix then, `cd` to the parent directory of `mytest` and type

    >>> read_eeg('mytest/test.eeg','Results/test.h5')

    The resulting HDF5 file contains the group `EEG` with attributes holding the corresponding metadata 
    (see Notes for details). The EEG time-courses can be found in `EEG`:

    >>> f = h5py.File('Results/test.h5')
    >>> f['EEG'].keys()
    >>> ['eeg_mat']
    
    The dataset `eeg_mat` holds the entire EEG dataset as matrix (NumPy 2darray),  

    >>> f['EEG']['eeg_mat'].value
    >>> array([[32079, 32045, 32001, ..., 33607, 33556, 33530],
              [31708, 31712, 31712, ..., 33607, 33597, 33599],
              [31719, 31722, 31704, ..., 33733, 33713, 33708],
              ..., 
              [39749, 34844, 36671, ..., 44616, 43642, 41030],
              [30206, 28126, 30805, ..., 39691, 36586, 34550],
              [31084, 30167, 31580, ..., 38113, 36470, 35205]], dtype=uint16)

    The attribute `electrode_list` is a NumPy array of electrodenames corresponding to the rows of `eeg_mat`, 
    i.e., `f['EEG']['eeg_mat'][23,:]` is the time-series of electrode `f['EEG'].attrs['electrode_list'][23]`

    >>> f['EEG'].attrs['electrode_list'][23]
    >>> 'RFC8'
    >>> f['EEG']['eeg_mat'][23,:]
    >>> array([33602, 33593, 33649, ..., 32626, 32648, 32650], dtype=uint16)

    Additional meta-data (like scanning date, session length, etc.) are also saved as group attributes
    where `summary` is a string representation of all meta values, i.e., 

    >>> f['EEG'].attrs['summary']
    >>>  Data was recorded on Friday, April 11 2014
         Begin of session: 10:1:49
         Sampling rate: 1000 Hz
         Length of session: 2.0 hours

    The respective meta values are stored as individual attributes (using numeric values), e.g., 

    >>> f['EEG'].attrs['session_length']
    >>> 2.0

    If only the electrodes 'RFA1' and 'RFA3' are of interest and the read-out should be saved
    by the respective electrode names then the following command could be used

    >> read_eeg('mytest/test.eeg','Results/test.h5',electrodelist=['RFA1','RFA3'],savemat=False)

    In this case the `EEG` group of the resulting HDF5 file looks like this

    >> f = h5py.File('Results/test.h5')
    >> f['EEG'].keys()
    >> ['RFA1', 'RFA3']
    >> f['EEG']['RFA1'].value
    >> array([32079, 32045, 32001, ..., 33607, 33556, 33530], dtype=uint16)

    Thus, the electrode time-courses are saved using the respective electrode names. 

    See also
    --------
    h5py : A Pythonic interface to the HDF5 binary data format
    """

    # Make sure eegpath is a regular string and doesn't contain any unicode weirdness 
    # (and expand "~" if present)
    if str(eegpath) != eegpath:
        raise TypeError('Input has to be a string specifying the path/name of the EEG files!')
    eegpath = str(eegpath)
    if eegpath.find("~") == 0:
        eegpath = os.path.expanduser('~') + eegpath[1:]

    # Same for outfile: no unicode and additionally check if the path is valid
    if str(outfile) != outfile:
        raise TypeError('Output filename has to be a string!')
    outfile = str(outfile)
    if outfile.find("~") == 0:
        outfile = os.path.expanduser('~') + outfile[1:]
    if not os.path.isdir(outfile[:outfile.rfind(os.sep)]):
        raise ValueError('Invalid path for output file: '+outfile+'!')
    if os.path.isfile(outfile): 
        raise ValueError("Target HDF5 container already exists!")

    if electrodelist != None: 
        try: le = len(electrodelist)
        except: raise TypeError('Input electrodlist must be a Python list!')
        if le == 0: raise ValueError('Input electrodelist has length 0!')

    # Make sure savemat and bold are either True or False
    msg = "The switch `savemat` has to be Boolean!"
    try:
        bad = (savemat != True and savemat != False)
    except: raise TypeError(msg)

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

    # If the EEG file is a container of concatenated EEG chunks, throw an error
    if numberOfBlocks > 1:
        raise ValueError('EEG file '+eegfls[0]+' seems to contain more than one recording. Exiting...')

    # Read EEG2m control block (contains names and addresses for waveform blocks)
    fid.seek(int(blockAddress),0)
    x              = np.fromfile(fid,dtype='uint8',count=1)
    x              = np.fromfile(fid,dtype='S1',count=16)
    numberOfBlocks = np.fromfile(fid,dtype='uint8',count=1)
    blockAddress   = np.fromfile(fid,dtype='int32',count=1)
    x              = np.fromfile(fid,dtype='S1',count=16)

    # If the EEG file is a container of concatenated EEG chunks, throw an error
    if numberOfBlocks > 1:
        raise ValueError('EEG file '+eegfls[0]+' seems to contain more than one recording. Exiting...')

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
    maxmem  = int(np.ceil(meminfo.available*0.25/(numChannels+1)/ds))

    # If the whole array fits into memory load it once, otherwise chunk it up
    if numSamples <= maxmem:
        blocksize = [numSamples]
    else:
        blocksize = [maxmem]*int(numSamples//maxmem)
        rest      = int(np.mod(numSamples,maxmem))
        if rest > 0: blocksize = blocksize + [rest]

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
        eeg.attrs.create('electrode_list',data=nodelist)
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

        # Read data block-wise and save to matrix or row (depending on user choice, add offset to get int16)
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
    eeg.attrs.create('summary',data=recordedstr+"\n"+beginstr+"\n"+sratestr+"\n"+lengthstr)
    eeg.attrs.create('record_date',data=[T_year,T_month,T_day,T_hour,T_minute,T_second])
    eeg.attrs.create('sampling_rate',data=actSamplerate)
    eeg.attrs.create('session_length',data=num100msBlocks/10/3600)
    eeg.attrs.create('AD_off',data=AD_off)
    eeg.attrs.create('AD_val',data=AD_val)
    eeg.attrs.create('CAL',data=CAL)
    eeg.attrs.create('comFlag',data=comFlag)
    eeg.attrs.create('bitLen',data=bitLen)
    eeg.attrs.create('numSamples',data=numSamples)

    # Close and finalize HDF write process
    f.close()
    print " Done. "

    return

##########################################################################################
def load_data(h5file,nodes=None):
    """
    Load data from HDF5 container generated with read_eeg

    Parameters
    ----------
    h5file : str or h5py.File instance
        String specifying file name (or path + filename) or `h5py.File` instance of
        HDF5 container to be accessed
    nodes : list or NumPy 1darray
        Python list or NumPy array of electrodes to be read. Can be either an array/list
        of strings or indices. By default `nodes=None` and all electrodes are read 
        from file

    Returns
    -------
    data : NumPy 2darray
        A #nodes-by-#samples array holding the data in float64 format

    Notes
    -----
    The raw iEEG data are stored as int16. This routine normalizes (divides by max(int16))
    and rescales the data based on the original channel sensitivity (read from the HDF5
    container). 

    Examples
    --------
    Suppose we want to read data stored in the file `iEEG.h5`. To access all of its contents
    use
    
    >>> data = load_data('iEEG.h5')
    >>> data.shape
    >>> (84, 9000000)

    If the HDF5 container is already open and only electrodes `RFA1` and `RFB1` should be read
    use

    >>> import h5py
    >>> f = h5py.File('iEEG.h5')
    >>> data = load_data(f,nodes=['RFA1','RFB1'])
    >>> data.shape
    >>> (2, 9000000)

    Alternatively, nodes can be specified using their indices in the file 

    >>> data = load_data(f,nodes=np.array([12,33]))
    >>> data.shape
    >>> (2, 9000000)

    See also
    --------
    read_eeg : Read raw EEG data from binary *.EEG/*.eeg and *.21E files
    """

    # Check if input HDF5 container makes sense
    f, closefile, ismat, ec_list = check_hdf(h5file)

    # Get indices of nodes to be read
    idx = []
    if nodes != None:
        if str(nodes[0]) == nodes[0]:
            for node in nodes:
                try: idx.append(ec_list.index(node))
                except KeyError: raise ValueError("Node "+node+" not found in file "+h5file+"!")
        else:
            try:
                if max(nodes) > len(ec_list)-1 or min(nodes) < 0:
                    raise ValueError("Indices not found in file "+h5file+"!")
            except: 
                errmsg = "Nodes have to be provided as Python list/NumPy 1darray of indices or strings!"
                raise TypeError(errmsg)
            for node in nodes:
                if np.round(node) != node:
                    raise ValueError("Found float "+str(node)+", integer required!")
                idx.append(node)
    else:
        idx = range(len(ec_list))

    # Get channel units and number of samples in file
    CAL = f['EEG'].attrs['CAL']
    N   = f['EEG'].attrs['numSamples']

    # Extract data from HDF5 file and divide by upper bound of dtype (-> values b/w -1/+1), multiply by CAL
    data = np.zeros((len(idx),N))
    if (ismat):
        dt = f['EEG']['eeg_mat'].dtype
        k  = 0
        for i in idx:
            data[k,:] = f['EEG']['eeg_mat'][i,:]
            k += 1
        data = data/np.iinfo(dt).max*CAL
    else:
        dt = f['EEG'][nodes[0]].dtype
        for node in nodes:
            data[i,:] = f['EEG'][node].value
        data = data/np.iinfo(dt).max*CAL

    # Close file if user provided just string
    if closefile: f.close()
            
    return data

##########################################################################################
def MA(signal, window_size, past=True):
    """
    Smooth 1d/2darray using a moving average filter along one axis

    Parameters
    ----------
    signal : NumPy 1d/2darray
        Input signal of shape `M`-by-`N`, where `M` is the number of signal sources (regions, measuring
        devices, etc.) and `N` is the number of observations/measurements. Smoothing is performed along the 
        second axis, i.e., for each source all `N` observations are smoothed independently of each other
        using the same moving average window. 
    window_size : scalar
        Positive scalar defining the size of the window to average over
    past : bool
        If `past = True` then only preceding data is used to calculate the moving average. In addition, 
        the rolling standard deviation is also computed. For `past = False` the input signal is 
        filtered using an MA sliding window that averages across data points between 
        `- window_size/2` and `window_size/2`. If `past = False` no rolling standard deviation is 
        calculated. 

    Returns
    -------
    If `past == True`
    ma_signal : NumPy 1d/2darray
        Moving average of signal (same shape as input)
    sd_signal : NumPy 1d/2darray
        Rolling Standard deviation of signal (same shape as input)

    If `past == False`
    ma_signal : NumPy 1d/2darray
        Smoothed signal (same shape as input)

    See also
    --------
    None

    Notes
    -----
    None
    """

    # Sanity checks
    try:
        shs = signal.shape
    except: raise TypeError("Input `signal` must be a NumPy 1d/2darray, not "+type(signal).__name__+"!")
    if len(shs) > 2:
        raise ValueError("Input `signal` must be a NumPy 1d/2darray!")
    if np.max(shs) < 2:
        raise ValueError("Input `signal` is an array of only one element! ")
    if np.isnan(signal).max() == True or np.isinf(signal).max() == True or np.isreal(signal).min() == False:
        raise ValueError("Input `signal` must be real without NaNs or Infs!")

    try:
        bad = window_size <= 0
    except: raise TypeError("Input window-size must be a positive scalar, not "+type(window_size).__name__+"!")
    if bad: raise ValueError("Input window-size must be a positive scalar!")

    msg = "The siwtch `past` has to be Boolean, not "+type(past).__name__+"!"
    try:
        bad = (past != True and past != False)
    except: raise TypeError(msg)
    if bad: raise TypeError(msg)

    # Use only preceding data points to calculate MA
    if past:

        # If the signal is 1D, reshape it so that the for loops work
        if len(shs) < 2: signal = signal.reshape((1,signal.size))

        # Allocate space for output
        ma_signal = np.zeros(signal.shape)
        sd_signal = np.zeros(signal.shape)

        # The first `window_size` entries are computed using an incremental average/variation computation
        # (update mean/variance by each new element that is added)
        ma_signal[:,0] = signal[:,0]
        for k in xrange(1,window_size):
            ma_signal[:,k] = (signal[:,k] + k*ma_signal[:,k-1])/(k+1)
            sd_signal[:,k] = (k*sd_signal[:,k-1] + (signal[:,k] - ma_signal[:,k-1])*(signal[:,k] - ma_signal[:,k]))/(k+1)
     
        # The remaining entries are the actual moving average/rolling variation
        for k in xrange(window_size,signal.shape[1]):
            ma_signal[:,k] = ma_signal[:,k-1] + (signal[:,k] - signal[:,k-window_size])/window_size
            sd_signal[:,k] = sd_signal[:,k-1] + (signal[:,k] - ma_signal[:,k] + signal[:,k-window_size] - ma_signal[:,k-1])*\
                           (signal[:,k] - signal[:,k-window_size])/window_size
     
        # To get the standard deviation, compute the sqrt of the rolling variation
        sd_signal = np.sqrt(sd_signal*(sd_signal>0))

        # In case we had a 1D signal, remove the unnecessary dimension
        ma_signal = ma_signal.squeeze()
        sd_signal = sd_signal.squeeze()

        return ma_signal, sd_signal

    # Much faster: use a convolution to compute the mean over [-window_size/2,0,window_size/2]
    else:
        
        # Assemble window and compute moving average of signal
        window    = np.ones(int(window_size))/float(window_size)
        ma_signal = ndimage.filters.convolve1d(signal,window,mode='nearest')

        return ma_signal

##########################################################################################
def time2ind(h5file,t_start,t_end):
    """
    Convert human readable 24hr times to indices used in given iEEG file container

    Parameters
    ----------
    h5file : str or h5py.File instance
        String specifying file name (or path + filename) or `h5py.File` instance of
        HDF5 container to be accessed
    t_start : list/NumPy 1darray
        Start time in 24hr format. Syntax is [hh,mm,ss]
    t_end : list/NumPy 1darray
        End time in 24hr format. Syntax is [hh,mm,ss]

    Returns
    -------
    ind_start : integer
        Index of iEEG array corresponding to provided start time `t_start`
    ind_stop : integer
        Index of iEEG array corresponding to provided end time `t_stop`

    See also
    --------
    load_eeg : Load data from HDF5 container generated with read_eeg
    """

    # Check if input HDF5 container makes sense
    f, closefile, ismat, ec_list = check_hdf(h5file)

    # Read session date and sampling rate from file
    sess_start = f['EEG'].attrs['record_date']
    s_rate     = f['EEG'].attrs['sampling_rate']

    # Extract hours of session start, on- and offsets
    sess_hour = sess_start[3]
    ts_hour   = t_start[0]
    te_hour   = t_end[0]

    # Convert session date to datetime object
    sess_begin = datetime(*sess_start)

    # Compute session end
    sess_stop = sess_begin + timedelta(hours=f['EEG'].attrs['session_length'])

    # Combine session date (sess_start[0:3] gives [yr,mnth,day]) with onset time ([hr,min,sec]) 
    ts_date = sess_start[0:3]
    t_begin = datetime(*np.hstack([ts_date,t_start]))

    # If onset hour is less than session hour (01 vs 23), we crossed the 12AM mark, correct t_begin
    if sess_hour > ts_hour:
        t_begin += timedelta(days=1)

    # Same for offset time
    te_date = [t_begin.year,t_begin.month,t_begin.day]
    t_stop  = datetime(*np.hstack([te_date,t_end]))
    if ts_hour > te_hour:
        t_stop += timedelta(days=1)

    # If session begin is later than onset time, raise an error
    if sess_begin > t_begin:
        msg = 'Recording starts at '+str(sess_begin)+' which is after provided start time at '+str(t_begin)
        raise ValueError(msg)

    # Analogously for offset time
    if sess_stop < t_stop:
        msg = 'Recording stops at '+str(sess_stop)+' which is before provided stop time at '+str(t_stop)
        raise ValueError(msg)

    # That's why we use datetime: subtract objects to get time differences
    ts_offset = t_begin - sess_begin 
    te_offset = t_stop - sess_begin

    # Indices are computed as offset seconds * sampling rate
    ind_start = ts_offset.seconds*s_rate
    ind_stop  = te_offset.seconds*s_rate

    # Close file if user provided just string
    if closefile: f.close()

    # Return converted start/stop indices
    return ind_start, ind_stop

##########################################################################################
def check_hdf(h5file):

    # See if we can open provided HDF5 container
    if str(h5file) == h5file:
        try:
            f = h5py.File(h5file)
        except: raise ValueError("Error opening file "+h5file)
        closefile = True
    elif type(h5file).__name__ == "File":
        try:
            h5file.filename 
        except: raise TypeError('Input is not a valid HDF5 file!')
        f         = h5file
        closefile = False
    else: raise TypeError('Input has to be a string specifying an HDF5 file or h5py.File instance!')

    # Check if data is stored as matrix or "tagged list"
    try:
        ismat = (f['EEG'].keys().count('eeg_mat') > 0)
    except: 
        raise TypeError("Input file "+str(h5file)+" does not seem to be an EEG data file...")

    # Get list of electrodes actually present in file
    if (ismat):
        ec_list = f['EEG'].attrs['electrode_list'].tolist()
    else:
        ec_list = f['EEG'].keys()

    # Return HDF5 file object and tell caller if 
    # container uses an array (ismat = True) or named list storage format and
    # if the file needs to be closed at the end(closefile = True)
    return f, closefile, ismat, ec_list
