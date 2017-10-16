# test_nws_tools.py - Testing module for `nws_tools.py`, run with `pytest --tb=short -v` or `pytest --pdb`
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: October  5 2017
# Last modified: <2017-10-16 17:15:09>

import pytest
import os
import sys
import numpy as np
sys.path.insert(0,'../analytic_tools/')
import nws_tools as nwt

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

    # corr = np.zeros(res.shape)
    # err = np.zeros((nsubs,))

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

        # corr[:,:,ns] = np.corrcoef(arr[:,:,ns],rowvar=0)
        # err[ns] = np.linalg.norm(corr[:,:,ns] - res[:,:,ns])

    return {"txtpath":txtpath, "sub_fls":sub_fls, "sublist":sublist, "arr":arr, "res":res}

# This class performs the actual testing    
class Test_get_corr(object):

    # Test error-checking of `txtpath`
    def test_txtpath(self,capsys):
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
    def test_corrtype(self,capsys,txt_files):
        with capsys.disabled():
            sys.stdout.write("-> Testing error handling of `corrtype`... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']),corrtype=3)
        assert "Statistical dependence type" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']),corrtype='neither_pearson_nor_mi')
        assert "Currently, only Pearson" in str(excinfo.value)

    # Test error-checking of `sublist`
    def test_sublist(self,capsys,txt_files):
        with capsys.disabled():
            sys.stdout.write("-> Testing error handling of `sublist`... ")
            sys.stdout.flush()
        with pytest.raises(TypeError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']),sublist=3)
        assert "Subject codes have to be provided as Python list/NumPy 1darray, not int!" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']),sublist=np.ones((2,2)))
        assert "Subject codes have to be provided as 1-d list" in str(excinfo.value) 

    # Here comes the actual torture-testing of the routine.... 
    def test_torture(self,capsys,txt_files):
        with capsys.disabled():
            sys.stdout.write("-> Torture-testing `get_corr`: ")
            sys.stdout.flush()

        # Let ``get_corr`` loose on the artifical data.
        # Note: we're only testing Pearson correlations here, (N)MI computation is
        # checked when testing ``nwt.mutual_info``
        res_dict = nwt.get_corr(str(txt_files['txtpath']),corrtype='pearson')

        # Make sure `bigmat` was assembled correctly
        with capsys.disabled():
            sys.stdout.write("\n\t-> data extraction... ")
            sys.stdout.flush()
        msg = "Data not correctly read from disk!"
        assert np.allclose(res_dict['bigmat'],txt_files['arr']), msg

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
    
        # Eliminate one time-point
        with capsys.disabled():
            sys.stdout.write("\n\t-> variable time-series-length... ")
            sys.stdout.flush()
        target_txt.write("\n".join(str(val) for val in backup_val[:-1]))
        with pytest.raises(ValueError) as excinfo:
            nwt.get_corr(str(txt_files['txtpath']))
        msg = "Expected a time-series of length "+str(tlen)+\
              ", but actual length is "+str(tlen-1)
        assert msg in str(excinfo.value)
        
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
        msg = "Found "+str(int(nroi-1))+" time-series for subject s"+\
              str(rand_sub+1)+", expected "+str(int(nroi))        
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


# For MI try np.vstack([np.cos(xvec),np.sin(xvec)]).T
