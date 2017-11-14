# test_nws_tools.py - Testing module for `nws_tools.py`, run with `pytest --tb=short -v` or `pytest --pdb`
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: October  5 2017
# Last modified: <2017-11-14 17:25:27>

from __future__ import division
import pytest
import os
import sys
import pdb
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import h5py
module_pth = os.path.dirname(os.path.abspath(__file__))
module_pth = module_pth[:module_pth.rfind(os.sep)]
sys.path.insert(0, module_pth)
import nws_tools as nwt

# Check if we're running locallly or on the Travis servers to avoid always going through
# the entire thing while appending additional tests
runninglocal = (os.environ["PWD"] == "/home/travis/analytic_tools")
# runninglocal = False
skiplocal = pytest.mark.skipif(runninglocal, reason="debugging new tests")

# ==========================================================================================
#    Collect some global fixtures that can be re-used throughout the entire module
# ==========================================================================================
# A random collection of some of the most used objects in ``nws_tools.py``
kitchensink = [None, True, 2, [2,3], (2,3), np.empty((2,2)), {"2":2, "3": "three"}, pd.DataFrame([2,3]), "string", plt.cm.jet]

# Fixture to check if everything and the kitchensink are blocked
@pytest.fixture(params=kitchensink)
def check_kitchensink(request):
    return request.param

# Fixture to check if non-array objects and array-likes can get past the gatekeeper
@pytest.fixture(params=[ks for ks in kitchensink if type(ks) != np.ndarray])
def check_notarray(request):
    return request.param

# A fixture that returns anything but a dictionary
@pytest.fixture(params=[ks for ks in kitchensink if type(ks) != dict])
def check_notdict(request):
    return request.param

# A fixture that returns anything but a matplotlib colormap
@pytest.fixture(params=[ks for ks in kitchensink if type(ks) != matplotlib.colors.LinearSegmentedColormap])
def check_notcmap(request):
    return request.param

# A fixture that returns anything but a Boolean value
@pytest.fixture(params=[ks for ks in kitchensink if type(ks) != bool])
def check_notbool(request):
    return request.param

# A fixture that returns anything but a string
@pytest.fixture(params=[ks for ks in kitchensink if type(ks) != str])
def check_notstring(request):
    return request.param
                
# A fixture that returns anything but a list/ndarray
@pytest.fixture(params=[ks for ks in kitchensink if type(ks) != list and type(ks) != np.ndarray])
def check_notlistlike(request):
    return request.param

# A fixture that returns anything but a 2darray
@pytest.fixture(params=[np.empty(()), np.empty((0)), np.empty((1)), np.empty((3,)), np.empty((3,3,2,1))])
def check_not2darray(request):
    return request.param

# A fixture that fills a NumPy 2-by-2 array with nonsense
@pytest.fixture(params=[np.inf, np.nan, "string", plt.cm.jet])
def invalid_mats(request):
    val = np.empty((2,2)).astype(type(request.param))
    val[:] = request.param
    return val

# ==========================================================================================
#                                       arrcheck
# ==========================================================================================
# Assemble a bunch of fixtures to iterate over a plethora of obscure invalid inputs
# Here come three separate fixtures for the tensor-, matrix- and vector-case. 
# Note: defining an one-size-fits-all fixture that uses all of the below akin to
#       ``invalid_dims(tensor_dimerr, matrix_dimerr, vector_dimerr, request)``
# doesn't really work, since it produces all possible (redundant) input combinations -> ~2.5k calls 
@pytest.fixture(scope="class", params=[np.empty(()),
                                       np.empty((0)),
                                       np.empty((1)),
                                       np.empty((3,)),
                                       np.empty((3,1)),
                                       np.empty((1,3)),
                                       np.empty((3,3)),
                                       np.empty((3,1,3)),
                                       np.empty((1,3,3)),
                                       np.empty((2,3,3)),
                                       np.empty((3,2,3)),
                                       np.empty((3,3,2,1))])
def tensor_dimerr(request):
    return request.param

@pytest.fixture(scope="class", params=[np.empty(()),
                                       np.empty((0)),
                                       np.empty((1)),
                                       np.empty((3,)),
                                       np.empty((3,1)),
                                       np.empty((1,3)),
                                       np.empty((2,3)),
                                       np.empty((3,2)),
                                       np.empty((3,3,1))])
def matrix_dimerr(request):
    return request.param

@pytest.fixture(scope="class", params=[np.empty(()),
                                       np.empty((0)),
                                       np.empty((1)),
                                       np.empty((2,3)),
                                       np.empty((3,2)),
                                       np.empty((3,3)),
                                       np.empty((3,3,1))])
def vector_dimerr(request):
    return request.param

# The following two work in concert: `invalid_arrs` uses `array_maker` to fill a tensor/matrix/vector with nonsense
@pytest.fixture(scope="class", params=["tensor","matrix","vector"])
def array_maker(request):
    if request.param == "tensor":
        return np.empty((3,3,2))
    elif request.param == "matrix":
        return np.empty((3,3))
    else:
        return np.empty((3,))
    
@pytest.fixture(scope="class", params=[np.inf, np.nan, "string", plt.cm.jet, complex(2,3)])
def invalid_arrs(array_maker, request):
    val = array_maker.astype(type(request.param))
    val[:] = request.param
    return val

# The following two work in concert: `invalid_range` uses `bounds_maker` to generate input arguments for `arrcheck`
@pytest.fixture(scope="class", params=[[-np.inf,0.5], [2,np.inf], [-1,0.5], [-0.5,0.5]])
def bounds_maker(request):
    return request.param

@pytest.fixture(scope="class", params=["tensor","matrix","vector"])
def invalid_range(bounds_maker, request):
    if request.param == "tensor":
        arr = np.ones((3,3,2))
    elif request.param == "matrix":
        arr = np.ones((3,3))
    else:
        arr = np.ones((3,))
    return {"arr":arr, "kind": request.param, "bounds":bounds_maker}

# Perform the actual error checking
@skiplocal
class Test_arrcheck(object):

    # See if non-NumPy arrays can get past `arrcheck`
    def test_nonarrays(self, capsys, check_notarray):
        with capsys.disabled():
            sys.stdout.write("-> Test if non-NumPy arrays can get through... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.arrcheck(check_notarray, "tensor", "testname")
        assert "Input `testname` must be a NumPy array, not" in str(excinfo.value)
    
    # See if the tensor-specific dimension testing works
    def test_tensor(self, capsys, tensor_dimerr):
        with capsys.disabled():
            sys.stdout.write("-> Trying to sneak in non-tensors... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.arrcheck(tensor_dimerr, "tensor", "testname")
        assert "Input `testname` must be a `N`-by-`N`-by-`k` NumPy array" in str(excinfo.value)
    
    # See if the matrix-specific dimension testing works
    def test_matrix(self, capsys, matrix_dimerr):
        with capsys.disabled():
            sys.stdout.write("-> Trying to sneak in non-matrices... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.arrcheck(matrix_dimerr, "matrix", "testname")
        assert "Input `testname` must be a `N`-by-`N` NumPy array" in str(excinfo.value)
    
    # See if the vector-specific dimension testing works
    def test_vector(self, capsys, vector_dimerr):
        with capsys.disabled():
            sys.stdout.write("-> Trying to sneak in non-vectors... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.arrcheck(vector_dimerr, "vector", "testname")
        assert "Input `testname` must be a NumPy 1darray" in str(excinfo.value)
        
    # Feed `arrcheck` with arrays stuffed with a boatload of invalid values
    def test_valerr(self, capsys, invalid_arrs):
        with capsys.disabled():
            sys.stdout.write("-> Trying to slip by infinite/undefined/non-real values... ")
            sys.stdout.flush()
        if len(invalid_arrs.shape) == 3:
            with pytest.raises(ValueError) as excinfo:
                nwt.arrcheck(invalid_arrs, "tensor", "testname")
            assert "Input `testname` must be a real-valued" in str(excinfo.value)
        elif len(invalid_arrs.shape) == 2:
            with pytest.raises(ValueError) as excinfo:
                nwt.arrcheck(invalid_arrs, "matrix", "testname")
            assert "Input `testname` must be a real-valued" in str(excinfo.value)
        else:
            with pytest.raises(ValueError) as excinfo:
                nwt.arrcheck(invalid_arrs, "vector", "testname")
            assert "Input `testname` must be a real-valued" in str(excinfo.value)

    # Pass arrays that consequently violate the provided bounds
    def test_bounds(self, capsys, invalid_range):
        with capsys.disabled():
            sys.stdout.write("-> Values out of specified bounds... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.arrcheck(invalid_range["arr"],\
                         invalid_range["kind"],\
                         "testname",\
                         bounds=invalid_range["bounds"])
        assert "Values of input array `testname` must be between" in str(excinfo.value)

# ==========================================================================================
#                                       scalarcheck
# ==========================================================================================
@pytest.fixture(scope="class", params=[np.empty(()), np.empty((0)), np.empty((1)), np.empty((2,2)),
                                       pd.DataFrame([2,3]), "s", plt.cm.jet])
def scalarcheck_nonscalar(request):
    return request.param

@pytest.fixture(scope="class", params=[np.inf, np.nan, complex(2,3)])
def invalid_scalars(request):
    return request.param

@pytest.fixture(scope="class", params=[[-np.inf,0.5], [2,np.inf], [-1,0.5], [-0.5,0.5]])
def invalid_bounds(request):
    return request.param

# Perform the actual error checking
@skiplocal
class Test_scalarcheck(object):

    # See if non-scalars can get past `scalarcheck`
    def test_nonscalars(self, capsys, scalarcheck_nonscalar):
        with capsys.disabled():
            sys.stdout.write("-> Test if non-scalars can get through... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.scalarcheck(scalarcheck_nonscalar, "testname")
        assert "Input `testname` must be a scalar!" in str(excinfo.value)

    # Throw some invalid arguments in there
    def test_invalids(self, capsys, invalid_scalars):
        with capsys.disabled():
            sys.stdout.write("-> Test invalid scalars... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.scalarcheck(invalid_scalars, "testname")
        assert "Input `testname` must be real and finite!" in str(excinfo.value)
    
    # See if the integer testing works
    def test_ints(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Test integer filtering... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.scalarcheck(np.pi, "testname", kind="int")
        assert "Input `testname` must be an integer!" in str(excinfo.value)
        with capsys.disabled():
            sys.stdout.write("don't raise dumb errors for integral floats... ")
            sys.stdout.flush()
        assert nwt.scalarcheck(2.0, "testname", kind="int") is None
        
    # Test bounds-checking
    def test_bounds(self, capsys, invalid_bounds):
        with capsys.disabled():
            sys.stdout.write("-> Test bounds checking... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.scalarcheck(1.0, "testname", bounds=invalid_bounds)
        assert "Input scalar `testname` must be between" in str(excinfo.value)

# ==========================================================================================
#                                       get_corr
# ==========================================================================================
# A fixture that uses `pytest`'s builtin `tmpdir` fixture to assemble some dummy time-series txt-files
@pytest.fixture(scope="class")
def txt_files(tmpdir_factory):

    # Set parameters for our dummy time-series (WARNING: decreasing `tlen` might severely affect
    # the numerical accuracy of NumPy's ``corrcoef``!)
    tlen = 500
    nsubs = 10
    nroi = 12

    # Construct time-series data based on out-of-phase sine-waves whose correlations can be calculated
    # analytically (cosine of their phase differences)
    arr = np.zeros((tlen,nroi,nsubs))
    res = np.zeros((nroi,nroi,nsubs))
    xvec = np.linspace(0,2*np.pi,tlen)
    phi = np.linspace(0,np.pi,nroi)

    # Assemble per-subject data
    sub_fls = []
    sublist = []
    txtpath = tmpdir_factory.mktemp("ts_data")
    for ns in xrange(nsubs):
        fls = []
        phi_ns = np.random.choice(phi,size=(phi.shape))
        sub = "s"+str(ns+1)
        for nr in xrange(nroi):
            arr[:,nr,ns] = np.sin(xvec + phi_ns[nr])
            res[nr,:,ns] = np.cos(np.abs(phi_ns - phi_ns[nr]))
            txtfile = txtpath.join(sub+"_"+str(nr+1)+".txt")
            txtfile.write("\n".join(str(val) for val in arr[:,nr,ns]))
            fls.append(txtfile)
        sublist.append(sub)
        sub_fls.append(fls)

    return {"txtpath":txtpath, "sub_fls":sub_fls, "sublist":sublist, "arr":arr, "res":res}

# This class performs the actual testing
@skiplocal
class Test_get_corr(object):

    # Test error-checking of `txtpath`
    def test_txtpath(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Testing error handling of `txtpath`... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.get_corr(2)
        assert "Input has to be a string" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr('   ')
        assert "Invalid directory" in str(excinfo.value)

    # Test error-checking of `corrtype`
    def test_corrtype(self, capsys, txt_files):
        with capsys.disabled():
            sys.stdout.write("-> Testing error handling of `corrtype`... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']), corrtype=3)
        assert "Statistical dependence type" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']), corrtype='neither_pearson_nor_mi')
        assert "Currently, only Pearson" in str(excinfo.value)

    # Test error-checking of `sublist`
    def test_sublist(self, capsys, txt_files):
        with capsys.disabled():
            sys.stdout.write("-> Testing error handling of `sublist`... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']), sublist=3)
        assert "Subject codes have to be provided as Python list/NumPy 1darray, not int!" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']), sublist=np.ones((2,2)))
        assert "Subject codes have to be provided as 1-d list" in str(excinfo.value) 

    # Here comes the actual torture-testing of the routine.... 
    def test_torture(self, capsys, txt_files):
        with capsys.disabled():
            sys.stdout.write("-> Start torture-testing ``get_corr``: ")
            sys.stdout.flush()

        # Let ``get_corr`` loose on the artifical data.
        # Note: we're only testing Pearson correlations here, (N)MI computation is
        # checked when testing ``nwt.mutual_info``
        res_dict = nwt.get_corr(str(txt_files['txtpath']), corrtype='pearson')

        # Make sure `bigmat` was assembled correctly
        with capsys.disabled():
            sys.stdout.write("\n\t-> data extraction... ")
            sys.stdout.flush()
        msg = "Data not correctly read from disk!"
        assert np.allclose(res_dict['bigmat'], txt_files['arr']), msg

        # Check if `txt_files["sublist"]` and `res_dict["sublist"]` are identical (preserving order!)
        with capsys.disabled():
            sys.stdout.write("\n\t-> subject file detection... ")
            sys.stdout.flush()
        s_tst = [s_ref for s_ref, s_test in zip(txt_files["sublist"],res_dict["sublist"]) if s_ref == s_test]
        msg = "Expected artifical subjects "+\
              "".join(sub+', ' for sub in txt_files["sublist"])[:-2]+\
              " but found "+"".join(sub+', ' for sub in res_dict["sublist"])[:-2]+"!"
        assert len(s_tst) == len(txt_files["sublist"]), msg

        # Finally, make sure pair-wise inter-regional correlations were computed correctly
        with capsys.disabled():
            sys.stdout.write("\n\t-> numerical accuracy of inter-regional correlations... ")
            sys.stdout.flush()
        err = np.linalg.norm(res_dict['corrs'] - txt_files['res'])
        tol = 0.05
        msg = "Deviation between expected and actual correlations is "+str(err)+" > "+str(tol)+"!"
        assert err < tol, msg
    
        # Randomly select one of the bogus ROIs of one of the dummy subjects
        tlen, nroi, nsubs = txt_files["arr"].shape
        rand_sub = np.random.choice(nsubs,1)[0]
        rand_fle = np.random.choice(nroi,1)[0]
        target_txt = txt_files['sub_fls'][rand_sub][rand_fle]
        backup_val = txt_files["arr"][:,rand_fle,rand_sub]

        # Try to extract non-existent subjects
        with capsys.disabled():
            sys.stdout.write("\n\t-> invalid subjects... ")
            sys.stdout.flush()
        sub_addon = ["s"+str(nsubs+k) for k in xrange(1,3)]
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files["txtpath"]), sublist=txt_files["sublist"]+sub_addon)
        msg = "No data found for Subject(s) "+"".join(sub+', ' for sub in sub_addon)[:-2]
        assert msg in str(excinfo.value)
        
        # Eliminate one time-point
        with capsys.disabled():
            sys.stdout.write("\n\t-> variable time-series-length... ")
            sys.stdout.flush()
        target_txt.write("\n".join(str(val) for val in backup_val[:-1]))
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']))
        assert "Expected a time-series of length" in str(excinfo.value)
        
        # Non-numeric file-contents
        with capsys.disabled():
            sys.stdout.write("\n\t-> non-numeric data... ")
            sys.stdout.flush()
        target_txt.write("\n".join(str_val for str_val in ["invalid"]*tlen))
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']))
        assert "Cannot read file" in str(excinfo.value)

        # Unequal no. of time-series per subject
        with capsys.disabled():
            sys.stdout.write("\n\t-> varying no. of time-series per subject... ")
            sys.stdout.flush()
        target_txt.remove()
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']))
        msg = "Inconsisten number of time-series across subjects! "+\
              "Found "+str(int(nroi-1))+" regions in Subject(s) s"+str(rand_sub+1)
        assert msg in str(excinfo.value)

        # Inconsistent no. of time-series + missing subjects
        with capsys.disabled():
            sys.stdout.write("\n\t-> inconsistent no. of regions and invalid subjects... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']), sublist=txt_files["sublist"]+sub_addon)
        msg = "Inconsisten number of time-series across subjects! "+\
              "No data found for Subject(s) "+"".join(sub+', ' for sub in sub_addon)[:-2]+\
              ", "+str(int(nroi-1))+" regions in Subject(s) s"+str(rand_sub+1)
        assert msg in str(excinfo.value)
        
        # Fewer than two volumes in a subject
        with capsys.disabled():
            sys.stdout.write("\n\t-> too few time-points in one subject... ")
            sys.stdout.flush()
        for fl in txt_files['sub_fls'][rand_sub]:
            fl.write("2\n3")
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']))
        msg = "Time-series of Subject s"+str(rand_sub+1)+\
              " is empty or has fewer than 2 entries!"
        assert msg in str(excinfo.value)
    
        # Fewer than two txt-files found
        with capsys.disabled():
            sys.stdout.write("\n\t-> not enough txt-files... ")
            sys.stdout.flush()
        for ns in xrange(nsubs):
            for fl in txt_files['sub_fls'][ns]:
                fl.remove()
        txt_files['sub_fls'][rand_sub][0].write("nonsense")
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']))
        msg = "Found fewer than 2 text files"
        assert msg in str(excinfo.value)

# ==========================================================================================
#                                       corrcheck
# ==========================================================================================
# Use a list of lists here since ``corrcheck`` accepts multiple input args (use `*args` to invoke ``corrcheck`` below)
@pytest.fixture(scope="class", params=[[np.empty((3,1))],
                                       [np.empty((1,3))],
                                       [np.empty((3,2))],
                                       [np.empty((2,3))],
                                       [np.empty((3,3)), np.empty((3,2))],
                                       [np.empty((3,3)), np.empty((2,2))],
                                       [np.empty((3,3,2)), np.empty((3,3))],
                                       [np.empty((3,1,3))],
                                       [np.empty((1,3,3))],
                                       [np.empty((2,3,3))],
                                       [np.empty((3,2,3))]])
def corrcheck_valerr(request):
    return request.param

# Perform the actual error checking
@skiplocal
class Test_corrcheck(object):

    # Recycle ``check_notarray`` to throw some invalid arguments at ``corrcheck``
    def test_nonarrays(self, capsys, check_notarray):
        with capsys.disabled():
            sys.stdout.write("-> Test if non-NumPy arrays can get through... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.corrcheck(check_notarray)
        assert "Expected NumPy array(s) as input, found" in str(excinfo.value)
    
    # Screw around with labels
    def test_labels(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Screwing up labels... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.corrcheck(np.empty((2,2)),["A"],["B"])
        assert "All but last input must be NumPy arrays!" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            nwt.corrcheck(np.empty((2,2)),np.empty((2,2)),["A"])
        assert "Numbers of labels and matrices do not match up!" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            nwt.corrcheck(np.empty((2,2,2)),["A"])
        assert "Numbers of labels and matrices do not match up!" in str(excinfo.value)
        
    # Feed in invalid arrays (`TypeErrors` raise different messages, so we don't check for that here)
    def test_types(self, capsys, check_not2darray):
        with capsys.disabled():
            sys.stdout.write("-> Invalid arrays... ")
            sys.stdout.flush()
        with pytest.raises(TypeError):
            nwt.corrcheck(check_not2darray)
    
    # Screw around with array input dimensions
    def test_values(self, capsys, corrcheck_valerr):
        with capsys.disabled():
            sys.stdout.write("-> Screwing up array dimenions... ")
            sys.stdout.flush()
        with pytest.raises(ValueError):
            nwt.corrcheck(*corrcheck_valerr)

    # Arrays with nonsense (note: `corrcheck` discards the imag. part of complex numbers, so we don't check for that)
    def test_values(self, capsys, invalid_mats):
        with capsys.disabled():
            sys.stdout.write("-> Invalid array values... ")
            sys.stdout.flush()
        with pytest.raises(ValueError):
            nwt.corrcheck(invalid_mats)

# ==========================================================================================
#                                       rm_selfies
# ==========================================================================================
# This should be quick (inputs are vetted by ``arrcheck`` and ``scalarcheck``)
@skiplocal
class Test_rm_selfies(object):
    def test_rm_selfies(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Test if self-connections are removed correctly... ")
            sys.stdout.flush()
        msg = "Self connections not removed correctly. This should *really* work as expected... "
        N = 5
        conns = np.ones((N,N))
        res = (conns - np.eye(N)).reshape(N,N,1)
        assert np.all(nwt.rm_selfies(conns.reshape(N,N,1)) == res) == True, msg

# ==========================================================================================
#                                       get_meannw
# ==========================================================================================
# Only test numerics and handling of missing connections (inputs are vetted by ``arrcheck`` and ``scalarcheck``)
@skiplocal
class Test_get_meannw(object):
    def test_numerics(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Start torture-testing ``get_meannw``: ")
            sys.stdout.flush()

        # Set network dimension `N`, group-size `M`, and common ratio `r` for geometric progression
        N = 10
        M = 10
        r = 0.5
        nws = np.zeros((N,N,M))
        for m in xrange(M):
            nws[:,:,m] = r**m

        # The group-average is given by the `M`-th partial sum of the geometric series scaled by `1/M`
        values = 1/M*(1 - r**M)/(1 - r)
        res = np.ones((N,N)) - np.eye(N)
        res *= values

        # Start by making sure the code averages correctly
        with capsys.disabled():
            sys.stdout.write("\n\t-> See if group-average is computed correctly... ")
            sys.stdout.flush()
        msg = "Group-average network not computed correctly! "
        res_nwt, pval_nwt = nwt.get_meannw(nws)
        assert np.all(np.isclose(res_nwt,res)) == True, msg
        msg = "Returned percentage value does not match up!"
        assert pval_nwt == 0.0, msg

        # Remove randomly chosen connections from randomly chosen networks (but don't disconnect nodes though)
        # Note: the removal percentage `rm_perc` *has* to be > 0.5 (strict!) for the test below to work
        rm_perc = 0.7
        all_gr = np.arange(M)
        all_nd = np.arange(N)
        graphs = np.random.choice(all_gr,size=(int(np.round(rm_perc*M)),),replace=False)
        nodes = np.random.choice(all_nd,size=(int(np.round(rm_perc*N)),),replace=False)
        graphs.sort(); nodes.sort()
        edges = zip(nodes,nodes[::-1])
        for gr in graphs:
            for edg in edges:
                nws[edg[0],edg[1],gr] = 0.0
                nws[edg[1],edg[0],gr] = 0.0

        # Use `rm_perc` as cutoff percentage in ``nwt.get_meannw``, thus effectively removing these
        # edges from the group-averaged network
        with capsys.disabled():
            sys.stdout.write("\n\t-> Testing percentage-based group-averaging... ")
            sys.stdout.flush()
        res_nwt, pval_nwt = nwt.get_meannw(nws,percval=rm_perc)
        for edg in edges:
            res[edg[0],edg[1]] = 0.0
            res[edg[1],edg[0]] = 0.0
        msg = "Percentage-based average not correct!"
        assert np.all(np.isclose(res_nwt,res)) == True, msg
        msg = "Returned percentage value does not match up!"
        assert pval_nwt == rm_perc, msg

        # Now don't just remove some edges but disconnect the selected nodes in the chosen graphs
        for gr in graphs:
            for nd in nodes:
                nws[:,nd,gr] = 0.0
                nws[nd,:,gr] = 0.0

        # Use the fact that diagonals are not zero in `nws` to compute the `1 - rm_perc` weighted
        # average across all non-pruned graphs
        rem_nd = np.setdiff1d(all_nd,nodes)
        rem_gr = np.setdiff1d(all_gr,graphs)
        val = nws[rem_nd[0],rem_nd[0],rem_gr].sum()/M

        # Use again `rm_perc` as cutoff percentage, which should trigger the safe-guard percentage
        # decrease in ``get_meannw``
        with capsys.disabled():
            sys.stdout.write("\n\t-> Testing percentage-correction in group-averaging... ")
            sys.stdout.flush()
        res_nwt, pval_nwt = nwt.get_meannw(nws,percval=rm_perc)
        res[:] = val
        for nd in rem_nd:
            res[nd,rem_nd] = values
        np.fill_diagonal(res,0)
        msg = "Percentage-corrected average not correct!"
        assert np.all(np.isclose(res_nwt,res)) == True, msg
        msg = "Corrected percentage not computed correctly!"
        assert np.isclose(pval_nwt, 1-rm_perc), msg

        # Finally, cut off a node in all networks to trigger a `ValueError`
        with capsys.disabled():
            sys.stdout.write("\n\t-> Disconnecting a node from all networks... ")
            sys.stdout.flush()
        nws[:,nodes[0],:] = 0.0
        nws[nodes[0],:,:] = 0.0
        with pytest.raises(ValueError) as excinfo:
            nwt.get_meannw(nws)
        assert "Mean network disconnected for `percval` = 0%" in str(excinfo.value)
        
# ==========================================================================================
#                                       rm_negatives
# ==========================================================================================
# This should be quick (inputs are vetted by ``arrcheck``)
@skiplocal
class Test_rm_negatives(object):
    def test_rm_negatives(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Test if negative edge-weights are removed correctly... ")
            sys.stdout.flush()
        msg = "Negative edges not removed correctly. This should *really* work as expected... "
        N = 10
        no_negatives = True
        while no_negatives:
            conns = np.random.normal(loc=0.0, scale=1.0, size=((N,N)))
            conns = np.triu(conns,1)
            conns += conns.T
            msk = (conns < 0)
            no_negatives = (msk.max() == False)
        res = conns.copy()
        res[msk] = 0.0
        assert np.all(nwt.rm_negatives(conns.reshape(N,N,1)) == res.reshape(N,N,1)) == True, msg

# ==========================================================================================
#                                       thresh_nws
# ==========================================================================================
# Only test methodology and handling of disconnected nodes (inputs are vetted by ``arrcheck`` and ``scalarcheck``)
@skiplocal
class Test_thresh_nws(object):
    def test_numerics(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Start torture-testing ``thresh_nws``: ")
            sys.stdout.flush()

        # Set up our testing network cohort filled with random weights and pick an arbitrary node/network
        N = 10
        M = 10
        msk = np.triu(np.ones((N,N),dtype=bool),1)
        all_nw = np.arange(M)
        all_nd = np.arange(N)
        all_eg = np.arange(msk.sum())
        nws = np.zeros((N,N,M))
        for m in all_nw:
            nw = np.random.normal(loc=0.5,scale=0.2,size=((N,N)))
            nw[nw < 0] = 0.0
            nw = np.triu(nw,1)
            nw += nw.T
            nws[:,:,m] = nw.copy()
        rand_nw = np.random.choice(M,1)[0]
        nw_rand = nws[:,:,rand_nw].copy()

        # Start with a cheap symmetry violation
        with capsys.disabled():
            sys.stdout.write("\n\t-> Try to sneak in a directed network... ")
            sys.stdout.flush()
        nws[0,1,rand_nw] = nws.max() + 0.1
        with pytest.raises(ValueError) as excinfo:
            nwt.thresh_nws(nws)
        assert "Matrix "+str(rand_nw)+" is not symmetric!" in str(excinfo.value)
        
        # Now a cheap sign error
        with capsys.disabled():
            sys.stdout.write("\n\t-> Try to sneak in negative edge weights... ")
            sys.stdout.flush()
        nws[0,1,rand_nw] = -1.0
        nws[1,0,rand_nw] = -1.0
        with pytest.raises(ValueError) as excinfo:
            nwt.thresh_nws(nws)
        assert "Only non-negative weights supported!" in str(excinfo.value)
        
        # Next a cheap density error
        with capsys.disabled():
            sys.stdout.write("\n\t-> Try to sneak in a zero-density graph... ")
            sys.stdout.flush()
        nws[:,:,rand_nw] = 0.0
        with pytest.raises(ValueError) as excinfo:
            nwt.thresh_nws(nws)
        assert "Network "+str(rand_nw)+" has density 0%" in str(excinfo.value)

        # Now start screwing around with the density of our input networks: lower it to `tgt_dens`
        tgt_dens = 0.7
        nws[:,:,rand_nw] = nw_rand.copy()
        rm_no = int(np.round((1 - tgt_dens)*all_eg.size))
        for m in all_nw:
            nw = nws[:,:,m].copy()
            vals = nw[msk]
            nw[:] = 0.0
            rm_edg = np.random.choice(all_eg,size=(rm_no,),replace=False)
            vals[rm_edg] = 0.0
            nw[msk] = vals
            nw += nw.T
            nws[:,:,m] = nw.copy()

        # Ironically, make sure ``nwt.thresh_nws`` does exactly nothing here
        with capsys.disabled():
            sys.stdout.write("\n\t-> Networks do not need to be thresholded... ")
            sys.stdout.flush()
        res_nwt = nwt.thresh_nws(nws, userdens=tgt_dens*100+10)
        msg = "Networks were thresholded despite having lower density than required!"
        assert res_nwt["tau_levels"] is None, msg
        
        # Same thing, but use spanning tree to do (almost) nothing
        with capsys.disabled():
            sys.stdout.write("\n\t-> Networks do not need to be thresholded with spanning trees either... ")
            sys.stdout.flush()
        res_nwt = nwt.thresh_nws(nws, userdens=tgt_dens*100+10, span_tree=True)
        assert np.all(res_nwt["th_nws"] == nws), msg

        # Manipulate the `rand_nw` network so that its minimal admissible density `min_den`
        # is 10% below `tgt_dens`. Note: `min_den` can be expressed in terms of `K`, the no. of entries
        # in the upper triangular portion of the network's connection matrix, as follows:
        #       `min_den = K/((N**2 - N)/2)`
        # Use this to compute `K` for a given `min_den`
        nw_rand = nws[:,:,rand_nw].copy()
        min_den = tgt_dens - 0.1
        nw[:] = 0.0
        K = int(np.round((min_den*(N**2 - N)/2)))
        vals = nw[msk]
        ad_edg = np.random.choice(all_eg,size=(K,), replace=False)
        vals[ad_edg] = nw_rand.max()
        nw[msk] = vals
        nw += nw.T
        nws[:,:,rand_nw] = nw.copy()

        # Make sure that ``thresh_nws`` computes the minimal admissible density across `nws` correctly
        with capsys.disabled():
            sys.stdout.write("\n\t-> Check correct calculation of minimal admissible density... ")
            sys.stdout.flush()
        res_nwt = nwt.thresh_nws(nws)
        msg = "Minimal admissible density not calculated correctly!"
        assert np.abs(res_nwt["den_values"] - min_den).mean() < 1e-2, msg

        # Use `foce_den` with a `userdens` that guarantess that the `rand_nw` network disconnects
        with capsys.disabled():
            sys.stdout.write("\n\t-> Force-threshold networks to disconnect nodes... ")
            sys.stdout.flush()
        res_nwt = nwt.thresh_nws(nws, userdens=100*min_den-10, force_den=True)
        msg = "Thresholding should have yielded a disconnected network!"
        assert (res_nwt["th_nws"][:,:,rand_nw] > 0).sum(axis=0).min() == 0

        # Ensure that maximum spanning trees are populated correctly (the actual spanning trees
        # are computed by `backbone_wu` in BCT and thus not error-checked
        nws[:,:,rand_nw] = nw_rand.copy()
        with capsys.disabled():
            sys.stdout.write("\n\t-> Use spanning trees to construct density-reduced networks... ")
            sys.stdout.flush()
        res_nwt = nwt.thresh_nws(nws, userdens=100*min_den, span_tree=True)
        msg = "Spanning trees not populated correctly!"
        assert np.abs(res_nwt["den_values"] - min_den).max() < 1e-1, msg

# ==========================================================================================
#                                       normalize
# ==========================================================================================
# A fixture to that creates a bunch of 2darrays filled with invalid entries
@pytest.fixture(scope="class", params=[np.inf, np.nan, "string", plt.cm.jet, complex(2,3)])
def invalid_2darrs(request):
    val = np.empty((2,2)).astype(type(request.param))
    val[:] = request.param
    return val

# Only test array handling and methodology (scalar inputs are vetted by ``scalarcheck``)
@skiplocal
class Test_normalize(object):
    
    # Recycle ``check_notarray`` to throw some invalid arguments at ``normalize``
    def test_nonarrays(self, capsys, check_notarray):
        with capsys.disabled():
            sys.stdout.write("-> Test if non-NumPy arrays can get through... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.normalize(check_notarray)
        assert "Input `arr` has to be a NumPy ndarray" in str(excinfo.value)
        
    # Arrays with nonsense
    def test_values(self, capsys, invalid_2darrs):
        with capsys.disabled():
            sys.stdout.write("-> Invalid array values... ")
            sys.stdout.flush()
        with pytest.raises(ValueError):
            nwt.normalize(invalid_2darrs)

    # Test the actual computational performance of ``normalize``
    def test_numerics(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Start torture-testing ``normalize``: ")
            sys.stdout.flush()

        # Use a diagonla matrix for testing below
        arr = sp.diag([1.,2.,3.], k=0)

        # Start with a cheap normalization bound violation
        with capsys.disabled():
            sys.stdout.write("\n\t-> Lower bound greater than upper bound... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.normalize(arr, vmin=1., vmax=0.)
        assert "Lower bound `vmin` has to be strictly smaller than upper bound `vmax`!" in str(excinfo.value)

        # Bounds are too close
        with capsys.disabled():
            sys.stdout.write("\n\t-> Difference in lower and upper bounds close to machine precision... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.normalize(arr, vmin=0., vmax=np.finfo(float).eps)
        assert "Bounds too close: `|vmin - vmax| < eps`, no normalization possible" in str(excinfo.value)

        # Array not normalizable
        with capsys.disabled():
            sys.stdout.write("\n\t-> Array min/max close to machine precision... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.normalize(np.array([0,np.finfo(float).eps]))
        assert "Minimal and maximal values of array too close, no normalization possible" in str(excinfo.value)

        # All elements in array are equal
        brr = np.ones((3,3))
        with capsys.disabled():
            sys.stdout.write("\n\t-> Array elements identical... ")
            sys.stdout.flush()
        res_arr = nwt.normalize(brr)
        msg = "Output array different from input array!"
        assert np.all(brr == res_arr), msg
        
        # Use Matplotlib's ``Normalize`` as reference implementation for the case `vmin = 0.0`, `vmax = 1.0`
        with capsys.disabled():
            sys.stdout.write("\n\t-> Compare unit normalization to Matplotlib's ``Normalize``... ")
            sys.stdout.flush()
        res_arr = nwt.normalize(arr)
        mpl_nrm = plt.Normalize()
        msg = "Unit normalization differs from Matplotlib's reference implementation!"
        assert np.all(mpl_nrm(arr) == res_arr), msg

        # Construct an integral example for which we can easily establish the normalization
        with capsys.disabled():
            sys.stdout.write("\n\t-> Check normalization based on an integral example... ")
            sys.stdout.flush()
        res_arr = nwt.normalize(arr, vmin=4., vmax=10.)
        ref_arr = 2*arr + 4.
        msg = "Exemplary toy array was not normalized correctly!"
        assert np.all(ref_arr == res_arr), msg

# ==========================================================================================
#                                       csv2dict
# ==========================================================================================
# A fixture that uses `pytest`'s builtin `tmpdir` fixture to assemble two dummy csv files
@pytest.fixture(scope="class")
def csv_file(tmpdir_factory):

    # Save an array of dummy nodal coordinates as invalid/regular csv-files
    nroi = 20
    coords = np.random.normal(size=(nroi,3))
    rand_line = np.random.choice(nroi,1)[0]
    csvpath = tmpdir_factory.mktemp("csv_files")
    csvnormal = csvpath.join("normal.csv")
    csvbroken = csvpath.join("broken.csv")
    with open(str(csvnormal), "w") as cvnrm: 
        for n in xrange(nroi):
            cvnrm.write("".join(str(val)+"," for val in coords[n,:])[:-1]+"\n")
    with open(str(csvbroken), "w") as cvbrk:
        for n in xrange(nroi):
            if n != rand_line:
                cvbrk.write("".join(str(val)+"," for val in coords[n,:])[:-1]+"\n")
            else:
                cvbrk.write("a, b, c\n")
    return {"csvnormal":csvnormal, "csvbroken":csvbroken, "coords":coords, "rand_line":rand_line}

# Test if csv-files are read correctly (don't go full OCD over the single string input of ``csv2dict`` though...)
@skiplocal
class Test_csv2dict(object):
    
    # Test error-checking of `csvfile`
    def test_csvfile(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Testing error handling of `csvfile`... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.csv2dict(2)
        assert "Name of csv-file has to be a string!" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            nwt.csv2dict("invalid")
        assert "File: `invalid` does not exist!" in str(excinfo.value)
        
    # Test actual csv-reading
    def test_torture(self, capsys, csv_file):
        with capsys.disabled():
            sys.stdout.write("-> Start torture-testing ``csv2dict``: ")
            sys.stdout.flush()

        # Read a csv file that contains an invalid line
        with capsys.disabled():
            sys.stdout.write("\n\t-> Invalid csv-file... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.csv2dict(str(csv_file["csvbroken"]))
        msg = "Error reading file `"+str(csv_file["csvbroken"])+"` on line "+str(csv_file["rand_line"]+1)
        assert msg in str(excinfo.value)

        # Make sure a valid csv-file is read correctly 
        with capsys.disabled():
            sys.stdout.write("\n\t-> Check correct read-out of csv-files... ")
            sys.stdout.flush()
        res_nwt = nwt.csv2dict(str(csv_file["csvnormal"]))
        res_arr = np.zeros((len(res_nwt.keys()),3))
        for k in res_nwt.keys():
            res_arr[k,:] = res_nwt[k]
        msg = "Exemplary csv file was not read correctly from disk!"
        assert np.all(np.isclose(res_arr,csv_file["coords"])) == True, msg

# ==========================================================================================
#                                       shownet
# ==========================================================================================
# Let's just not worry about Mayavi right now...
def test_shownet():
    pytest.skip("Let's not deal with Mayavi here. Skipping this...")

# ==========================================================================================
#                                       show_nw
# ==========================================================================================
# Modify the global `notlistlike` fixture by removing the `None` type
@pytest.fixture(params=[ks for ks in kitchensink if type(ks) != list and type(ks) != np.ndarray and ks is not None])
def check_notnonelistlike(request):
    return request.param

# Test if invalid inputs can slip by the input-checking in ``show_nw`` (however, most of the heavy lifting is done by
# ``arrcheck`` and ``scalarcheck`` anyway...)
@skiplocal
class Test_show_nw(object):

    # Construct a dummy connection matrix
    N = 10
    A = np.random.normal(loc=0.0, scale=1.0, size=((N,N)))
    coords = {}
    xyz = [1., 2., 3.]
    for k in xrange(N):
        coords[k] = xyz
    global A, coords
    
    # See if we can sneak in non-dictionaries as coordinates
    def test_nondicts(self, capsys, check_notdict):
        with capsys.disabled():
            sys.stdout.write("-> Test if non-dictionaries can get through... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.show_nw(A, check_notdict)
        assert "Nodal coordinates have to be provided as dictionary!" in str(excinfo.value)

    # Screw with the contents of `coords`
    def test_nonlists(self, capsys, check_notlistlike):
        with capsys.disabled():
            sys.stdout.write("-> Invalid entries in `coords`... ")
            sys.stdout.flush()
        coords[0] = check_notlistlike
        with pytest.raises(TypeError) as excinfo:
            nwt.show_nw(A, coords)
        assert "All elements of the coords dictionary have to be lists/arrays!" in str(excinfo.value)
        
    # Fine-grained error-checking of `coords`
    def test_coords(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Invalid format of `coords`... ")
            sys.stdout.flush()
        coords.pop(0)
        with pytest.raises(ValueError) as excinfo:
            nwt.show_nw(A, coords)
        assert "The coordinate dictionary has to have `N` keys!" in str(excinfo.value)
        coords[0] = np.ones((4,))
        with pytest.raises(ValueError) as excinfo:
            nwt.show_nw(A, coords)
        assert "All elements of the coords dictionary have to be 3-dimensional!" in str(excinfo.value)
        coords[0] = coords[1]

    # Check color- and nodal size-vectors
    def test_colsizvecs(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Invalid color- and size-vectors... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.show_nw(A, coords, colorvec=np.zeros((A.shape[0]-1)))
        assert "`colorvec` has to have length `N`!" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            nwt.show_nw(A, coords, sizevec=np.zeros((A.shape[0]+1)))
        assert "`sizevec` has to have length `N`!" in str(excinfo.value)
        
    # Screw with `labels`
    def test_invalidlabels(self, capsys, check_notnonelistlike):
        with capsys.disabled():
            sys.stdout.write("-> Invalid `labels`... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.show_nw(A, coords, labels=check_notnonelistlike)
        assert "Nodal labels have to be provided as list/NumPy 1darray!" in str(excinfo.value)

    # Fine-grained error-checking of `labels`
    def test_labels(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Invalid format of `labels`... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.show_nw(A, coords, labels=coords.keys()[:-1])
        assert "Number of nodes and labels does not match up!" in str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            nwt.show_nw(A, coords, labels=coords.keys())
        assert "Each individual label has to be a string type!" in str(excinfo.value)

    # See if anything but matplotlib colormaps can get in
    def test_nodecmap(self, capsys, check_notcmap):
        with capsys.disabled():
            sys.stdout.write("-> Invalid colormaps... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.show_nw(A, coords, nodecmap=check_notcmap)
        assert "Nodal colormap has to be a Matplotlib colormap!" in str(excinfo.value)
    def test_edgecmap(self, capsys, check_notcmap):
        with capsys.disabled():
            sys.stdout.write("-> Invalid colormaps... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.show_nw(A, coords, edgecmap=check_notcmap)
        assert "Edge colormap has to be a Matplotlib colormap!" in str(excinfo.value)
        
    # Screw with `nodes3d`
    def test_invalidnodes3d(self, capsys, check_notbool):
        with capsys.disabled():
            sys.stdout.write("-> Invalid `nodes3d`... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.show_nw(A, coords, nodes3d=check_notbool)
        assert "The `nodes3d` flag has to be a Boolean variable!" in str(excinfo.value)

    # Screw with `viewtype`
    def test_invalidviewtype(self, capsys, check_notstring):
        with capsys.disabled():
            sys.stdout.write("-> Invalid `viewtype`... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.show_nw(A, coords, viewtype=check_notstring)
        msg = "The optional input `viewtype` must be 'axial(_{t/b})', 'sagittal(_{l/r})' or 'coronal(_{f/b})'"
        assert msg in str(excinfo.value)
                
    # Check `linewidths`
    def test_lwdths(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Invalid format of `linewidths`... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.show_nw(A, coords, linewidths=np.ones((A.shape[0]+1,A.shape[0]+1)))
        msg = "Linewidths must be provided as square array of the same dimension as the connection matrix!"
        assert msg in str(excinfo.value)

# ==========================================================================================
#                                       generate_randnws
# ==========================================================================================
# Mainly test numerics of ``generate_randnws`` (main inputs are vetted by ``arrcheck`` and ``scalarcheck``)
@skiplocal
class Test_generate_randnws(object):

    # Screw around with `method`
    def test_invalidmethod(self, capsys, check_notstring):
        with capsys.disabled():
            sys.stdout.write("-> Invalid `method`...")
            sys.stdout.flush()
        nw = np.ones((3,3))
        with pytest.raises(TypeError) as excinfo:
            nwt.generate_randnws(nw, 2, method=check_notstring)
        assert "Randomization algorithm must be specified as string!" in str(excinfo.value)

    # Here comes the actual torture-testing of the routine (we need to assess the command line output
    # of ``generate_randnws`` here, so don't screw around with ``capsys`` in here
    def test_torture(self, capsys):

        # Construct directed and undirected dummy networks of size `N` and set no. of random networks to generate (`M`)
        N = 10
        M = 3
        nw_dir = np.random.normal(loc=0.5,scale=0.2,size=((N,N)))
        nw_dir[nw_dir < 0] = 0.0
        nw_und = np.triu(nw_dir,1)
        nw_und += nw_und.T

        # Lower the density of both networks to `tgt_dens` so that they're shuffable with the ``randmio_*`` routines
        tgt_dens = 0.5

        # Start with the undirected graph
        msk = np.triu(np.ones((N,N),dtype=bool),1)
        all_eg = np.arange(msk.sum())
        rm_no = int(np.round((1 - tgt_dens)*all_eg.size))
        vals = nw_und[msk]
        nw_und[:] = 0.0
        rm_edg = np.random.choice(all_eg,size=(rm_no,),replace=False)
        vals[rm_edg] = 0.0
        nw_und[msk] = vals
        nw_und += nw_und.T

        # Now the directed network
        msk = np.ones((N,N),dtype=bool) - np.eye(N,dtype=bool)
        all_eg = np.arange(N**2-N)
        rm_no = int(np.round((1 - tgt_dens)*all_eg.size))
        vals = nw_dir[msk]
        nw_dir[:] = 0.0
        rm_edg = np.random.choice(all_eg,size=(rm_no,),replace=False)
        vals[rm_edg] = 0.0
        nw_dir[msk] = vals

        # Make sure the correct randomization strategies are chosen
        nwt.generate_randnws(nw_dir, M)
        out, err = capsys.readouterr()
        msg = "Directed network was not randomized properly!"
        assert "randmio_dir" in out, msg
        nwt.generate_randnws(nw_und, M)
        out, err = capsys.readouterr()
        msg = "Undirected network was not randomized properly!"
        assert "randmio_und" in out, msg

        # Randomize network using user-defined methods
        res_nwt = nwt.generate_randnws(nw_dir, M, method="randmio_dir_connected")
        msg = "Directed network was not properly randomized using ``randmio_dir_connected``!"
        assert np.all([np.all(np.isclose(rnw, nw_dir)) for rnw in res_nwt.T]) == False, msg
        
        res_nwt = nwt.generate_randnws(nw_und, M, method="randmio_und_connected")
        msg = "Directed network was not properly randomized using ``randmio_und_connected``!"
        assert np.all([np.all(np.isclose(rnw, nw_und)) for rnw in res_nwt.T]) == False, msg

# ==========================================================================================
#                                       hdfburp
# ==========================================================================================
# Use `pytest`'s builtin `tmpdir` fixture to assemble a dummy HDF5 container
@pytest.fixture(scope="class")
def hdf_file(tmpdir_factory):

    # Allocate some dummy quantities that will be written to/read from the container
    flt = 2.0
    it = int(3)
    arr = np.array([[1,2,3],[4,5,6]])
    st = "string"

    # Prepare the container and dictionary holding its contents as it should appear
    # in local workspace after calling ``hdfburp``
    hdf_path = tmpdir_factory.mktemp("hdf_dymmy")
    hdf_fle = str(hdf_path) + "dummy.h5"
    h5f = h5py.File(hdf_fle)
    val_dict = {}

    # Save vars in the base-group
    for var in ["flt", "arr", "st"]:
        h5f.create_dataset(var,data=eval(var))
        val_dict[var] = eval(var)
    h5f.create_dataset("it",data=it,dtype=int)
    val_dict["it"] = it

    # Save vars in subgroups
    g1name = "group1"
    g2name = "group2"
    group1 = h5f.create_group(g1name)
    group1.create_dataset("flt",data=flt)
    val_dict[g1name+"_flt"] = flt
    group1.create_dataset("it",data=it,dtype=int)
    val_dict[g1name+"_it"] = it
    group2 = h5f.create_group(g2name)
    for var in ["arr", "st"]:
        group2.create_dataset(var,data=eval(var))
        val_dict[g2name+"_"+var] = eval(var)
    h5f.close()

    return {"hdf_fle":hdf_fle, "val_dict":val_dict}

# Test if variables are read correctly from previously generated HDF5 container
@skiplocal
class Test_hdfburp(object):

    # Screw with the one mandatory input
    def test_invalidhdf(self, capsys, check_kitchensink):
        with capsys.disabled():
            sys.stdout.write("-> Invalid HDF5 container...")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.hdfburp(check_kitchensink)
        assert "Input must be a valid HDF5 file identifier!" in str(excinfo.value)

    # Here comes the actual torture-testing
    def test_torture(self, capsys, hdf_file):
        with capsys.disabled():
            sys.stdout.write("-> Start torture-testing ``hdfburp``: ")
            sys.stdout.flush()

        # Let ``hdfburp`` work its magic and assess local workspace afterwards
        h5fle = h5py.File(hdf_file["hdf_fle"],"r")
        nwt.hdfburp(h5fle)
        h5fle.close()
        loc_dict = locals()

        # Make sure the entire contents of our dummy HDF5 container exists in the local workspace
        with capsys.disabled():
            sys.stdout.write("\n\t-> Entire contents of HDF5 container extracted... ")
            sys.stdout.flush()
        msg = "Not all data read from container!"
        assert np.all([loc_dict.has_key(key) for key in hdf_file["val_dict"].keys()]) == True, msg

        # Ascertain that local variables are identical to container contents
        with capsys.disabled():
            sys.stdout.write("\n\t-> Extracted quantities identical to HDF5 container contents... ")
            sys.stdout.flush()
        msg = "Data not correctly read from container!"
        assert np.all([(np.all(loc_dict[key] == value) and isinstance(loc_dict[key],type(value)))\
                       for key, value in hdf_file["val_dict"].items()]) == True, msg
        
# ==========================================================================================
#                                       mutual_info
# ==========================================================================================
# Test numerics of (N)MI computation and consistency of C++/Python versions
class Test_mutual_info(object):

    # Construct two time-series that have zero linear correlation but are quadratically dependent:
    # `x` and `x^2` on the interval [-1, +1]
    N = 1000
    nbins = int(np.ceil(np.sqrt(N)))
    x = np.linspace(-1,1,N)
    data = np.vstack([x,x**2]).T
    global data, nbins

    # See if non-NumPy arrays can get in
    def test_nonarrays(self, capsys, check_notarray):
        with capsys.disabled():
            sys.stdout.write("-> Test if non-NumPy arrays can get through... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.mutual_info(check_notarray)
        assert "Input must be a timepoint-by-index NumPy 2d array" in str(excinfo.value)

    # See if non-2darrays can get in
    def test_non2darrs(self, capsys, check_not2darray):
        with capsys.disabled():
            sys.stdout.write("-> Invalid array dimension... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.mutual_info(check_not2darray)
        
    # Fill input arrays with nonsense
    def test_invalidarrs(self, capsys, invalid_mats):
        with capsys.disabled():
            sys.stdout.write("-> Invalid input arrays... ")
            sys.stdout.flush()
        with pytest.raises(ValueError) as excinfo:
            nwt.mutual_info(invalid_mats)
        assert "real-valued" in str(excinfo.value)
        
    # Screw with `normalized`, `fast` and `norm_ts` (`n_bins` is vetted by ``scalarcheck``)
    def test_normalized(self, capsys, check_notbool):
        with capsys.disabled():
            sys.stdout.write("-> Make `normalized` invalid...")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.mutual_info(data, normalized=check_notbool)
        assert "The flags `normalized`, `fast` and `norm_ts` must be Boolean!" in str(excinfo.value)
    def test_fast(self, capsys, check_notbool):
        with capsys.disabled():
            sys.stdout.write("-> Make `fast` invalid...")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.mutual_info(data, fast=check_notbool)
        assert "The flags `normalized`, `fast` and `norm_ts` must be Boolean!" in str(excinfo.value)
    def test_norm_ts(self, capsys, check_notbool):
        with capsys.disabled():
            sys.stdout.write("-> Make `norm_ts` invalid...")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.mutual_info(data, norm_ts=check_notbool)
        assert "The flags `normalized`, `fast` and `norm_ts` must be Boolean!" in str(excinfo.value)
        
    # Here comes the actual torture-testing
    def test_torture(self, capsys):
        with capsys.disabled():
            sys.stdout.write("-> Start torture-testing ``mutual_info``: ")
            sys.stdout.flush()

        # Start by checking if the time-series pre-processing is working as expected
        with capsys.disabled():
            sys.stdout.write("\n\t-> Checking time-series pre-processing... ")
            sys.stdout.flush()
        data_test = data.copy()
        nwt.normalize_time_series(data_test)
        msg = "De-meaning and unit-variance normalization incorrect!"
        assert np.all(np.isclose(data_test.mean(axis=0),0.0)) and np.all(np.isclose(data_test.std(axis=0),1.0)), msg
        
        # Make sure MI and NMI computed in Python are correct
        with capsys.disabled():
            sys.stdout.write("\n\t-> See if Python-based MI is computed correctly... ")
            sys.stdout.flush()
        py_mi = nwt.mutual_info(data, n_bins=nbins, fast=False, normalized=False)[0,1]
        msg = "Python-based MI-calculation incorrect!"
        assert py_mi > 1.79, msg
        with capsys.disabled():
            sys.stdout.write("\n\t-> See if Python-based NMI is computed correctly... ")
            sys.stdout.flush()
        py_nmi = nwt.mutual_info(data, n_bins=nbins, fast=False, normalized=True)[0,1]
        msg = "Python-based NMI-calculation incorrect!"
        assert py_nmi > 0.65, msg

        # Compute (N)MI in C++ and make sure we get identical results
        with capsys.disabled():
            sys.stdout.write("\n\t-> See if C++-based MI is computed correctly... ")
            sys.stdout.flush()
        cp_mi = nwt.mutual_info(data, n_bins=nbins, fast=True, normalized=False)[0,1]
        msg = "C++-based MI-calculation incorrect!"
        assert np.isclose(py_mi, cp_mi), msg
        with capsys.disabled():
            sys.stdout.write("\n\t-> See if C++-based NMI is computed correctly... ")
            sys.stdout.flush()
        cp_nmi = nwt.mutual_info(data, n_bins=nbins, fast=True, normalized=True)[0,1]
        msg = "C++-based NMI-calculation incorrect!"
        assert np.isclose(py_nmi, cp_nmi), msg
