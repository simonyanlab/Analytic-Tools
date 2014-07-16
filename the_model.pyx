# the_model.pyx - Cythonized version of model equations including RK15 solver
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# June 25 2014

# Python imports
from __future__ import division
from scipy.stats import norm
import numpy as np
import h5py
import progressbar as pb

# Cythonized compile time imports
cimport numpy as np
from libc.math cimport tanh, exp, sqrt
cimport cython

# Assign default datatype for arrays and compile time types 
DTYPE = np.double 
ctypedef np.double_t DTYPE_t

# Because think different...
IF UNAME_SYSNAME == "Darwin":

    # External declaration of BLAS dot/symmetric matrix-vector product (for doubles) routines
    cdef extern from "Accelerate/Accelerate.h":
        double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY)

        cdef enum CBLAS_ORDER:
                CblasRowMajor=101 
                CblasColMajor=102
        cdef enum CBLAS_UPLO: 
                CblasUpper=121 
                CblasLower=122
        cdef enum CBLAS_TRANSPOSE: 
                CblasNoTrans=111
                CblasTrans=112 
                CblasConjTrans=113

        void dsymv "cblas_dsymv"(CBLAS_ORDER Order, CBLAS_UPLO uplo, int n, double alpha,
               double *A, int lda, double *x, int incx, double beta,
               double *y, int incy)

        void dgemv "cblas_dgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N, double alpha, 
                         double *A, int lda, double *X, int incX, double beta,
                         double *Y, int incY)

ELSE:

    # External declaration of BLAS dot/symmetric matrix-vector product (for doubles) routines
    cdef extern from "cblas.h":
        double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY)

        cdef enum CBLAS_ORDER:
                CblasRowMajor=101 
                CblasColMajor=102
        cdef enum CBLAS_UPLO: 
                CblasUpper=121 
                CblasLower=122
        cdef enum CBLAS_TRANSPOSE: 
                CblasNoTrans=111
                CblasTrans=112 
                CblasConjTrans=113

        void dsymv "cblas_dsymv"(CBLAS_ORDER Order, CBLAS_UPLO uplo, int n, double alpha,
               double *A, int lda, double *x, int incx, double beta,
               double *y, int incy)

        void dgemv "cblas_dgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N, double alpha, 
                         double *A, int lda, double *X, int incX, double beta,
                         double *Y, int incY)

# Turn of bounds-checking for entire function
@cython.boundscheck(False) 

# Assign parameters and tmp-variables
cdef class par:

    # Declare class fields: scalars
    cdef public int N
    cdef public DTYPE_t TCa
    cdef public DTYPE_t deCa
    cdef public DTYPE_t gCa
    cdef public DTYPE_t VCa
    cdef public DTYPE_t TK
    cdef public DTYPE_t deK
    cdef public DTYPE_t gK
    cdef public DTYPE_t VK
    cdef public DTYPE_t TNa
    cdef public DTYPE_t deNa
    cdef public DTYPE_t gNa
    cdef public DTYPE_t VNa
    cdef public DTYPE_t VL
    cdef public DTYPE_t gL
    cdef public DTYPE_t VT
    cdef public DTYPE_t ZT
    cdef public DTYPE_t deZ
    cdef public DTYPE_t aee
    cdef public DTYPE_t phi
    cdef public DTYPE_t tau
    cdef public DTYPE_t rNMDA
    cdef public DTYPE_t deV
    cdef public DTYPE_t QVmax
    cdef public DTYPE_t QZmax
    cdef public DTYPE_t W0
    cdef public DTYPE_t b
    cdef public DTYPE_t delta
    cdef public DTYPE_t v_m
    cdef public DTYPE_t k_m
    cdef public DTYPE_t tonic
    cdef public DTYPE_t rmin
    cdef public DTYPE_t rmax
    cdef public DTYPE_t a
    cdef public DTYPE_t b_hi
    cdef public DTYPE_t b_lo
    cdef public DTYPE_t dt
    cdef public DTYPE_t s_step
    cdef public DTYPE_t speechon
    cdef public DTYPE_t speechoff
    cdef public DTYPE_t len_cycle

    # Declare class fields: arrays
    cdef public np.ndarray C
    cdef public np.ndarray D
    cdef public np.ndarray V
    cdef public np.ndarray DA
    cdef public np.ndarray Z
    cdef public np.ndarray QV
    cdef public np.ndarray QZ
    cdef public np.ndarray W
    cdef public np.ndarray mK
    cdef public np.ndarray beta
    cdef public np.ndarray cplng
    cdef public np.ndarray aei
    cdef public np.ndarray aie
    cdef public np.ndarray ani
    cdef public np.ndarray ane

    # Class constructor
    def __cinit__(self, dict p_dict):

        # Check if C is symmetric and in row-major order (NumPy default)
        if np.isfortran(p_dict['C']): 
            raise TypeError("Coupling matrix has to be in row-major order (NumPy default)!")
        if np.isfortran(p_dict['D']): 
            raise TypeError("Dopamine matrix has to be in row-major order (NumPy default)!")
        if (np.linalg.norm(p_dict['C'] - p_dict['C'].T,ord='fro') > 1e-9*np.linalg.norm(p_dict['C'],ord='fro')):
            raise ValueError("Coupling matrix has to be symmetric!")

        # Initialize scalars
        self.N     = p_dict['C'].shape[0]
        self.TCa   = p_dict['TCa']
        self.deCa  = p_dict['deCa']
        self.gCa   = p_dict['gCa']
        self.VCa   = p_dict['VCa']
        self.TK    = p_dict['TK']
        self.deK   = p_dict['deK']
        self.gK    = p_dict['gK']
        self.VK    = p_dict['VK']
        self.TNa   = p_dict['TNa']
        self.deNa  = p_dict['deNa']
        self.gNa   = p_dict['gNa']
        self.VNa   = p_dict['VNa']
        self.VL    = p_dict['VL']
        self.gL    = p_dict['gL']
        self.VT    = p_dict['VT']
        self.ZT    = p_dict['ZT']
        self.deZ   = p_dict['deZ']
        self.aee   = p_dict['aee']
        self.phi   = p_dict['phi']
        self.tau   = p_dict['tau']
        self.rNMDA = p_dict['rNMDA']
        self.deV   = p_dict['deV']
        self.QVmax = p_dict['QVmax']
        self.QZmax = p_dict['QZmax']
        self.W0    = p_dict['W0']
        self.b     = p_dict['b']
        self.delta = p_dict['delta']
        self.v_m   = p_dict['v_m']
        self.k_m   = p_dict['k_m']
        self.rmin  = p_dict['rmin']
        self.rmax  = p_dict['rmax']
        self.a     = p_dict['a']
        self.b_hi  = p_dict['b_hi']
        self.b_lo  = p_dict['b_lo']

        # Scalar parameters not really for the model but for the code
        self.dt        = p_dict['dt']
        self.s_step    = p_dict['s_step']
        self.speechon  = p_dict['speechon']
        self.speechoff = p_dict['speechoff']
        self.len_cycle = p_dict['len_cycle']

        # Allocate arrays
        self.V     = np.zeros([self.N,], dtype=DTYPE)
        self.DA    = np.zeros([self.N,], dtype=DTYPE)
        self.Z     = np.zeros([self.N,], dtype=DTYPE)
        self.QV    = np.zeros([self.N,], dtype=DTYPE)
        self.QZ    = np.zeros([self.N,], dtype=DTYPE)
        self.W     = np.zeros([self.N,], dtype=DTYPE)
        self.mK    = np.zeros([self.N,], dtype=DTYPE)
        self.beta  = np.ones([self.N,], dtype=DTYPE)
        self.cplng = np.zeros([self.N,], dtype=DTYPE)
        self.C     = p_dict['C']
        self.D     = p_dict['D']
        self.aei   = p_dict['aei']
        self.aie   = p_dict['aie']
        self.ani   = p_dict['ani']
        self.ane   = p_dict['ane']

# Vectorized tanh computation
cdef DTYPE_t vectanh(np.ndarray[DTYPE_t, ndim = 1] invec, np.ndarray[DTYPE_t, ndim = 1] outvec, int n):
    
    cdef int i
    for i in xrange(n):
        outvec[i] = tanh(invec[i])

# C delcaration of the neural mass model
cdef void model_eqns(np.ndarray[DTYPE_t, ndim = 1] X, \
                     DTYPE_t t, \
                     par p, \
                     np.ndarray[DTYPE_t, ndim = 1] A, \
                     np.ndarray[DTYPE_t, ndim = 1] B, \
                     np.ndarray[DTYPE_t, ndim = 1] tmp):
    """
    The model equations written in the form
        dXt = A(Xt,t)dt + B dWt
    """
    # Declare local variables
    cdef int i
    cdef int n = p.N
    cdef DTYPE_t tsec, r, denom, isspeech

    # The input has the format X=[V0,...,V997,Z0,...,Z997]
    p.V  = X[0:p.N]
    p.Z  = X[p.N:2*p.N]
    p.DA = X[2*p.N:3*p.N]

    # Cell firing rates
    vectanh((p.V - p.VT)/p.deV,tmp,n)
    p.QV = 0.5*p.QVmax*(1 + tmp)
    vectanh((p.Z - p.ZT)/p.deZ,tmp,n)
    p.QZ = 0.5*p.QZmax*(1 + tmp)

    # Get position within the current cycle (in seconds)
    tsec     = t/1000.
    tsec     = tsec - int(tsec/p.len_cycle)*p.len_cycle
    isspeech = DTYPE(tsec >= p.speechon and tsec <= p.speechoff)

    # Prepare to compute y = alpha*A*x + beta*y, result stored in y. 
    # Here: alpha = (1-p.a)*(p.b_hi-p.b_lo), beta = p.b_lo and y = p.beta (that's why we set p.beta=1 below)
    p.beta[:] = 1.0

    # Compute dopamine gain: p.beta = p.D.dot(p.DA)*(1 - p.a)*(p.b_hi - p.b_lo) + p.b_lo using CBLAS
    dgemv(CblasRowMajor,CblasNoTrans,p.D.shape[0],p.D.shape[1],(1-p.a)*(p.b_hi-p.b_lo),\
          <DTYPE_t*>(p.D.data),p.D.shape[0],<DTYPE_t*>(p.DA.data),p.DA.strides[0]//sizeof(DTYPE_t), p.b_lo,\
          <DTYPE_t*>(p.beta.data), p.beta.strides[0]//sizeof(DTYPE_t))

    # Neural activation functions
    vectanh((p.V - p.TCa)/p.deCa,tmp,n)
    mCa = 0.5*(1 + tmp)
    vectanh((p.V - p.TNa)/p.deNa,tmp,n)
    mNa = 0.5*(1 + tmp)

    # Fraction of open potassium channels
    vectanh(p.beta*(p.V - p.TK)/p.deK,tmp,n)
    p.mK = 0.5*(1 + tmp)
    p.W  = (p.W0 - p.mK*p.phi)*exp(-t/p.tau) + p.mK*p.phi

    # Compute excitatory coupling based on connection matrix (cplng = C.dot(QV))
    dsymv(CblasRowMajor,CblasUpper,p.C.shape[1],1.0,<DTYPE_t*>(p.C.data),p.C.shape[0],<DTYPE_t*>(p.QV.data),\
          p.QV.strides[0]//sizeof(DTYPE_t),0.0,<DTYPE_t*>(p.cplng.data),p.cplng.strides[0]//sizeof(DTYPE_t))

    # Deterministic part for V (make sure denom = 1 for rest, other we have a divide by zero on our hands...)
    denom    = p.b_hi - p.b_lo + (1. - DTYPE(p.b_lo != p.b_hi))
    tmp      = (p.beta - p.b_lo)/denom + 1
    A[0:p.N] = - (p.gCa + p.cplng*p.rNMDA*p.aee*tmp)*mCa*(p.V - p.VCa) \
               - p.gK*p.W*(p.V - p.VK) - p.gL*(p.V - p.VL) \
               - (p.gNa*mNa + p.cplng*p.aee*tmp)*(p.V - p.VNa) \
               + p.aie*p.Z*p.QZ

    # Deterministic part for Z
    A[p.N:2*p.N] = p.b*p.aei*p.V*p.QV

    # Dopamine model
    r              = (p.rmax - p.rmin)*isspeech + p.rmin
    # A[2*p.N:3*p.N] = r*p.QV - p.v_m*(1 + p.k_m/p.DA)**(-1)
    A[2*p.N:3*p.N] = r*p.QV - (p.v_m*p.DA)*((p.DA + p.k_m)**(-1))
    # A[2*p.N:3*p.N] = r*p.QV - (p.v_m*p.DA*(1 + p.k_m/p.DA)**(-1))

    # Stochastic (diffusion) parts for V and Z (note that diffusion for DA is 0)
    B[0:p.N]     = p.ane*p.delta
    B[p.N:2*p.N] = p.b*p.ani*p.delta
        
# RK 15 solver SPECIFIALLY for the model
cpdef np.ndarray[DTYPE_t, ndim = 2] solve_model(np.ndarray[DTYPE_t, ndim = 1] x0, \
                                                np.ndarray[DTYPE_t, ndim = 1] tsteps, \
                                                object myp, \
                                                np.ndarray[np.long_t, ndim = 1] blocksize, \
                                                np.ndarray[np.long_t, ndim = 1] chunksize, \
                                                int seed, \
                                                str outfile):
    """
    Solve the model using a custom-tailored Runge--Kutta method of strong order 1.5
    """

    # Allocate local variables: scalars
    cdef int j, chunk, k, bsize, n, pb_i
    cdef int tsize      = tsteps.size - 1
    cdef int x0size     = x0.size
    cdef int numblocks  = blocksize.size
    cdef DTYPE_t dt     = tsteps[1] - tsteps[0]
    cdef DTYPE_t sqrtdt = sqrt(dt)
    cdef DTYPE_t dt1    = dt**(-1)
    cdef DTYPE_t dt2    = 0.5*sqrtdt**(-1)

    # Allocate local variables: arrays
    cdef np.ndarray zeta1   = np.zeros([tsize,], dtype=DTYPE)
    cdef np.ndarray zeta2   = np.zeros([tsize,], dtype=DTYPE)
    cdef np.ndarray DW      = np.zeros([tsize,], dtype=DTYPE)
    cdef np.ndarray DZ      = np.zeros([tsize,], dtype=DTYPE)
    cdef np.ndarray dtplus  = np.zeros([tsize,], dtype=DTYPE)
    cdef np.ndarray dtminus = np.zeros([tsize,], dtype=DTYPE)
    cdef np.ndarray A       = np.zeros([x0size,], dtype=DTYPE)
    cdef np.ndarray B       = np.zeros([x0size,], dtype=DTYPE)
    cdef np.ndarray tmp     = np.zeros([myp.N,], dtype=DTYPE)
    cdef np.ndarray tmpAB   = np.zeros([x0size,], dtype=DTYPE)
    cdef np.ndarray Aplus   = np.zeros([x0size,], dtype=DTYPE)
    cdef np.ndarray Aminus  = np.zeros([x0size,], dtype=DTYPE)
    cdef np.ndarray Atmp    = np.zeros([x0size,], dtype=DTYPE)
    cdef np.ndarray Btmp    = np.zeros([x0size,], dtype=DTYPE)
    cdef np.ndarray Y       = np.zeros([x0size,blocksize[0]], dtype=DTYPE)
    cdef np.ndarray QV      = np.zeros([myp.N,blocksize[0]], dtype=DTYPE)
    cdef np.ndarray Beta    = np.zeros([myp.N,blocksize[0]], dtype=DTYPE)

    # Open HDF5 container that will hold the output
    fout = h5py.File(outfile)

    # Generate pair of correlated normally distributed random variables and a linear combination of them
    np.random.seed(seed)
    zeta1   = norm.rvs(size=(tsteps.size,),loc=0,scale=1)
    zeta2   = norm.rvs(size=(tsteps.size,),loc=0,scale=1)
    DW      = zeta1*sqrtdt
    DZ      = 0.5*(zeta1 + sqrt(3)**(-1)*zeta2)*sqrtdt**3
    dtplus  = 0.25*dt + dt2*DZ
    dtminus = 0.25*dt - dt2*DZ

    # Due to chunking the last column of Y is the IC
    Y[:,-1] = x0

    # Initialize progressbar
    widgets = ['Running simulation... ',pb.Percentage(),' ',pb.Bar(marker='#'),' ',pb.ETA()]
    pbar    = pb.ProgressBar(widgets=widgets,maxval=tsize)
    pbar.start()

    # Compute solution recursively block by block
    j     = 0
    chunk = 0
    pb_i  = 0
    for i in xrange(numblocks):

        # Get current block-/chunksize and set new IC for Y to where the last chunk left off
        Y[:,0] = Y[:,-1] 
        bsize  = blocksize[i]
        csize  = chunksize[i]

        # Compute solution for current block
        k = 0
        for n in xrange(j,j+bsize-1):

            # Get drift/diffusion from func and save stuff in temporaray arrays
            model_eqns(Y[:,k], tsteps[n], myp, A, B, tmp)
            QV[:,k]   = myp.QV 
            Beta[:,k] = myp.beta
            Atmp      = Y[:,k] + A*dt
            Btmp      = B*sqrtdt

            # Second evaluations to update coefficients for solver
            model_eqns(Atmp + Btmp, tsteps[n+1], myp, Aplus,tmpAB,tmp)
            model_eqns(Atmp - Btmp, tsteps[n+1], myp, Aminus,tmpAB,tmp)
            
            # Compute solution at next time point
            Y[:,k+1] = Y[:,k] + 0.5*A*dt +B*DW[n] + Aplus*dtplus[n] + Aminus*dtminus[n]

            # Update internal block counter
            k += 1

            # Update progressbar
            pbar.update(pb_i)
            pb_i += 1

        # Write computed blocks to file
        fout['V'][:,chunk:chunk+csize]    = Y[0:myp.N,0:bsize:myp.s_step]
        fout['Z'][:,chunk:chunk+csize]    = Y[myp.N:2*myp.N,0:bsize:myp.s_step]
        fout['DA'][:,chunk:chunk+csize]   = Y[2*myp.N:3*myp.N,0:bsize:myp.s_step]
        fout['QV'][:,chunk:chunk+csize]   = QV[:,0:bsize:myp.s_step]
        fout['Beta'][:,chunk:chunk+csize] = Beta[:,0:bsize:myp.s_step]

        # Update index counters
        j     += bsize
        chunk += csize

    # Terminate progressbar
    pbar.finish()

    # Close HDF5 container and write everything to disk
    fout.close()
