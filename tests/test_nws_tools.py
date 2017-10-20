# test_nws_tools.py - Testing module for `nws_tools.py`, run with `pytest --tb=short -v` or `pytest --pdb`
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: October  5 2017
# Last modified: <2017-10-20 16:59:01>

from __future__ import division
import pytest
import os
import sys
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
module_pth = os.path.dirname(os.path.abspath(__file__))
module_pth = module_pth[:module_pth.rfind(os.sep)]
sys.path.insert(0, module_pth)
import nws_tools as nwt

# Check if we're running locallly or on the Travis servers to avoid always running the entire thing while
# appending additional tests
runninglocal = (os.environ["PWD"] == "/home/travis/analytic_tools")
skiplocal = pytest.mark.skipif(runninglocal, reason="debugging new tests")

# ==========================================================================================
#                                       arrcheck
# ==========================================================================================
# Assemble a bunch of fixtures to iterate over a plethora of obscure invalid inputs
# Start by checking if non-array objects and array-likes can get past the gatekeeper
@pytest.fixture(scope="session", params=[2, [2,3], (2,3), pd.DataFrame([2,3]), "string", plt.cm.jet])
def check_nonarray(request):
    return request.param

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
    def test_nonarrays(self, capsys, check_nonarray):
        with capsys.disabled():
            sys.stdout.write("-> Test if non-NumPy arrays can get through... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.arrcheck(check_nonarray, "tensor", "testname")
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
# A fixture that uses `pytest`'s builtin `tmpdir` to assemble some dummy time-series txt-files
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
# The usual fixtures to automate things
@pytest.fixture(scope="class", params=[np.empty(()), np.empty((0)), np.empty((1)), np.empty((3,)), np.empty((3,3,2,1))])
def corrcheck_typeerr(request):
    return request.param

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

@pytest.fixture(scope="class", params=[np.inf, np.nan, "string", plt.cm.jet])
def invalid_mats(request):
    val = np.empty((2,2)).astype(type(request.param))
    val[:] = request.param
    return val

# Perform the actual error checking
@skiplocal
class Test_corrcheck(object):

    # Recycle ``check_nonarray`` to throw some invalid arguments at ``corrcheck``
    def test_nonarrays(self, capsys, check_nonarray):
        with capsys.disabled():
            sys.stdout.write("-> Test if non-NumPy arrays can get through... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.corrcheck(check_nonarray)
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
    def test_types(self, capsys, corrcheck_typeerr):
        with capsys.disabled():
            sys.stdout.write("-> Invalid arrays... ")
            sys.stdout.flush()
        with pytest.raises(TypeError):
            nwt.corrcheck(corrcheck_typeerr)
    
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
            nws[:,:,m] = nw
        rand_nw = np.random.choice(M,1)[0]
        rand_nd = np.random.choice(N,1)[0]
        nw_rand = nws[:,:,rand_nw].copy()

        # Start with a cheap symmetry violation
        with capsys.disabled():
            sys.stdout.write("\n\t-> Try to sneak in a directed network... ")
            sys.stdout.flush()
        nws[rand_nd,1,rand_nw] = nws.max() + 0.1
        with pytest.raises(ValueError) as excinfo:
            nwt.thresh_nws(nws)
        assert "Matrix "+str(rand_nw)+" is not symmetric!" in str(excinfo.value)
        
        # Now a cheap sign error
        with capsys.disabled():
            sys.stdout.write("\n\t-> Try to sneak in negative edge weights... ")
            sys.stdout.flush()
        nws[rand_nd,1,rand_nw] = -1.0
        nws[1,rand_nd,rand_nw] = -1.0
        with pytest.raises(ValueError) as excinfo:
            nwt.thresh_nws(nws)
        assert "Only non-negative weights supported!" in str(excinfo.value)
        
        # Next a cheap density error
        with capsys.disabled():
            sys.stdout.write("\n\t-> Try to sneak in zero-density graph... ")
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
            nw = nws[:,:,m]
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
        res_nwt = nwt.thresh_nws(nws,userdens=tgt_dens*100+10)
        msg = "Networks were thresholded despite being of lower density than required!"
        assert res_nwt["tau_levels"] is None, msg
        
# For MI try np.vstack([np.cos(xvec),np.sin(xvec)]).T

