# parameters.py - Default parameters for the NMM
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# Created: June 23 2014
# Last modified: 2015-01-29 11:50:47

from __future__ import division

# ===============================================================================================
# IMPORTANT NOTE: All variable names in this file have a specific meaning and must not be changed 
#        	  to maintain compatibility with sim_tools.py!
# ===============================================================================================

# File-name (including path if not in working directory) of HDF5 container that holds coupling matrix
# 'C' and dopamine connection matrix 'D'. The container must contain datasets 'C', 'D' (both `N`-by-`N` 
# NumPy 2darrays) and 'names' (NumPy 1darry or Python list of region names, length `N`) in the root group, 
# i.e., 
# 
# >>> h5py.File(matrices).keys() 
# 
# should give 
# 
# >>> [C', 'D', 'names']
# 
# Optionally, a list of regional abbreviations called 'labels' can be included in the root group too, e.g., if
# 
# >>> names = ['L_Inferior_Frontal_Gyrus','R_Inferior_Frontal_Gyrus'] 
# 
# then the corresponding list of abbreviations could be 
# 
# >>> labels = ['L_IFG','R_IFG']
# 
# The routine run_model will extract 'labels' too (if present) and save it together with 'names' in the 
# generated output container
matrices = 'path/to/container.h5'

# Sampling frequency for saving model output (in Hz)
s_rate = 20

# Number of speech cycles to simulate
n_cycles = 150

# Details of the speech cycle: stimulus length, production and image acquisition times (all in seconds)
stimulus    = 3.6
production  = 5
acquisition = 2

# Synaptic coupling strengths - strings are used here to avoid the explicit definition of `N` in this file,
# i.e., the string expressions are evaluated inside run_model where the number of simulated regions `N` 
# is already known. This way, a parameter file can be used with different brain parcellations 
aee   = 0.4
aei   = "2.0*np.ones((N,))"
aie   = "-2.*np.random.normal(loc=1.,scale=0.05,size=(N,))"
ani   = "0.4*np.ones((N,))"
ane   = "2.*np.random.normal(loc=1.,scale=0.45,size=(N,))"

# Dopamine parameters
rmin = 0.0005
Rmax = 0.01 
rmax = str(Rmax)+"*np.ones((N,))"

# More model parameters, for details refer to Fuertinger et al: A Neural Population Model Incorporating 
# Dopaminergic Neurotransmission during Complex Voluntary Behaviors, PLoS Computational Biology, in press
TCa   = -0.01
deCa  = 0.15
gCa   = 1.1
VCa   = 1.
TK    = 0.0
deK   = 0.3
gK    = 2.0
VK    = -0.7
TNa   = 0.3
deNa  = 0.15
gNa   = 6.7
VNa   = 0.53
VL    = -0.5
gL    = 0.5
VT    = 0.54
ZT    = 0.0
deZ   = 0.7
phi   = 0.7
tau   = 1.
rNMDA = 0.25
deV   = 2.
QVmax = 1.
QZmax = 1.
W0    = 0.
b     = 0.1
delta = 0.2
v_m   = 0.004
k_m   = 0.125
rmin  = 0.0005
a     = 0.25
b_hi  = 50. 
b_lo  = 1.  
