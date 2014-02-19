# sde_solvers.py - Collection of numerical methods to solve (vector-valued) SDEs
# 
# Author: Stefan Fuertinger [stefan.fuertinger@mssm.edu]
# September 27 2013

from __future__ import division
import numpy as np
from scipy.stats import norm

def rk_1(func,x0,tsteps,**kwargs):
    r"""
    Explicit first order (strong and weak) Runge--Kutta method for SDEs with 
    additive/multiplicative (non-)autonomous scalar noise

    Parameters
    ----------
    func : callable (X,t,**kwargs)
        Returns drift `A` and diffusion `B` of the SDE. See Examples for details.  
    x0 : NumPy 1darray
        Initial condition 
    tsteps : NumPy 1darray
        Sequence of time points for which to solve (including initial time `t0`)
    **kwargs : keyword arguments
        Additional keyword arguments to be passed on to `func`. See `Examples` for details.  

    Returns
    -------
    Y : NumPy 2darray
        Approximate solution at timepoints given by `tsteps`. Format is 
                `Y[:,tk]` approximate solution at time `tk`
        Thus `Y` is a `numstate`-by-`timesteps` array

    Notes
    -----
    The general form of an SDE with additive/multiplicative (non-)autonomous scalar noise is

    .. math:: (1) \qquad dX_t = A(X_t,t)dt + B(X_t,t)dW_t, \quad X(t_0) = x_0

    The method for solving the SDE (1) is described in Sec. 11.1 of 
    Kloeden, P.E., & Platen, E. (1999). `Numerical Solution of Stochastic Differential Equations.` 
    Berlin: Springer.
    
    Examples
    --------
    Consider the SDE system 

    .. math:: 

                    dV_t & = - \alpha t V_t + t Z_t \beta dW_t,\\
                    dZ_t & =  \alpha t Z_t + t V_t \gamma dW_t,\\
                    V_{t_0} & = 0.5, \quad Z_{t_0} = -0.5, \quad t_0 = 1,

    thus with :math:`X_t = (V_t,Z_t)` we have

    .. math:: 

                    A(X_t,t) & = (-\alpha t V_t,\alpha t Z_t),\\
                    B(t)    & = (t Z_t \beta,t V_t \gamma).

    Hence `func` would look like this:
    
    ::

            import numpy as np
            def myrhs(Xt,t,alpha=0.2,beta=0.01,gamma=0.02):
                    A = np.array([-alpha*t*Xt[0],alpha*t*Xt[1]])
                    B = np.array([t*Xt[1]*beta,t*Xt[0]*gamma])
                    return A,B

    Thus, the full call to `rk_1` to approximate the SDE system on :math:`[t_0,2]` could be 
    something like (assuming the function `myrhs` is defined in `myrhs.py`)

    >>> import numpy as np
    >>> from sde_solvers import rk_1
    >>> from myrhs import myrhs
    >>> Xt = rk_1(myrhs,np.array([0.5,-0.5]),np.arange(1,2,1e-3),beta=.02)

    Hence we used :math:`\beta = 0.02` in `myrhs` instead of the default value 0.01. 

    See also
    --------
    pc_1 : an implicit first order strong Runge--Kutta method 
           (it uses a strong order 0.5 Euler--Maruyama method as predictor and an implicit Runge--Kutta
           update formula as corrector) for stiff SDEs
    """

    # Check for correctness of input and allocate common tmp variables
    Y,dt,sqrtdt,zeta1,zeta2 = checkinput(func,x0,tsteps)

    # Generate i.i.d. normal random variables with mean=0 (loc) and std=sqrt(delta) (scale) (Var=std^2)
    DW = zeta1*sqrtdt

    # Compute solution recursively
    for n in xrange(tsteps.size - 1):

        # Get drift/diffusion from func
        t      = tsteps[n]
        A, B   = func(Y[:,n], t,**kwargs)
        BGamma = func(Y[:,n] + A*dt + B*sqrtdt, t,**kwargs)[1]

        # Compute solution at next time point
        Y[:,n+1] = Y[:,n] + A*dt + B*DW[n] + 0.5*(BGamma - B)*(DW[n]**2 - dt)*sqrtdt**(-1)

    return Y

def pc_1(func,x0,tsteps,**kwargs):
    r"""
    Predictor-Corrector solver based on an 
    implicit first order (strong and weak) Runge--Kutta method for SDEs with 
    additive/multiplicative (non-)autonomous scalar noise

    Parameters
    ----------
    func : callable (X,t,**kwargs)
        Returns drift `A` and diffusion `B` of the SDE. See Examples for details.  
    x0 : NumPy 1darray
        Initial condition 
    tsteps : NumPy 1darray
        Sequence of time points for which to solve (including initial time `t0`)
    **kwargs : keyword arguments
        Additional keyword arguments to be passed on to `func`. See `Examples` for details.  

    Returns
    -------
    Y : NumPy 2darray
        Approximate solution at timepoints given by `tsteps`. Format is 
                `Y[:,tk]` approximate solution at time `tk`
        Thus `Y` is a `numstate`-by-`timesteps` array

    Notes
    -----
    The general form of an SDE with additive/multiplicative (non-)autonomous scalar noise is 

    .. math:: (1) \qquad dX_t = A(X_t,t)dt + B(X_t,t)dW_t, \quad X(t_0) = x_0

    The code  implements a two-fold approach to approximate solutions of (1). At each time-step 
    :math:`t_n` an order 0.5 strong Euler--Maruyama method is employed to estimate the solution 
    at time :math:`t_{n+1}` (predictor). This approximation is then used in the implicit 
    Runge--Kutta update formula (corrector). 
    The implicit Runge--Kutta method for solving the SDE (1) is described in Sec. 12.3 of 
    Kloeden, P.E., & Platen, E. (1999). `Numerical Solution of Stochastic Differential Equations. 
    Berlin: Springer.` The explicit Euler--Maruyama scheme is detailed in Sec. 9.1 ibid.

    Examples
    --------
    Consider the SDE system 

    .. math:: 

                    dV_t & = - \alpha t V_t + t Z_t \beta dW_t,\\
                    dZ_t & =  \alpha t Z_t + t V_t \gamma dW_t,\\
                    V_{t_0} & = 0.5, \quad Z_{t_0} = -0.5, \quad t_0 = 1,

    thus with :math:`X_t = (V_t,Z_t)` we have

    .. math:: 

                    A(X_t,t) & = (-\alpha t V_t,\alpha t Z_t),\\
                    B(t)    & = (t Z_t \beta,t V_t \gamma).

    Hence `func` would look like this:
    
    ::

            import numpy as np
            def myrhs(Xt,t,alpha=0.2,beta=0.01,gamma=0.02):
                    A = np.array([-alpha*t*Xt[0],alpha*t*Xt[1]])
                    B = np.array([t*Xt[1]*beta,t*Xt[0]*gamma])
                    return A,B

    Thus, the full call to `pc_1` to approximate the SDE system on :math:`[t_0,2]` could be 
    something like (assuming the function `myrhs` is defined in `myrhs.py`)

    >>> import numpy as np
    >>> from sde_solvers import pc_1
    >>> from myrhs import myrhs
    >>> Xt = pc_1(myrhs,np.array([0.5,-0.5]),np.arange(1,2,1e-3),beta=.02)

    Hence we used :math:`\beta = 0.02` in `myrhs` instead of the default value 0.01. 
    
    See also
    --------
    pc_15 : an implicit order 1.5 order strong Runge--Kutta method (it uses the mehod of ``rk_15``
            as predictor and the corresponding implicit update formula as corrector). 
    """

    # Check for correctness of input and allocate common tmp variables
    Y,dt,sqrtdt,zeta1,zeta2 = checkinput(func,x0,tsteps)

    # Generate i.i.d. normal random variables with mean=0 (loc) and std=sqrt(delta) (scale) (Var=std^2)
    DW = zeta1*sqrtdt

    # Compute solution recursively
    for n in xrange(tsteps.size - 1):

        # Get drift/diffusion from func
        t      = tsteps[n]
        A, B   = func(Y[:,n], t,**kwargs)
        BGamma = func(Y[:,n] + A*dt + B*sqrtdt, t,**kwargs)[1]

        # Explicit Euler-Maruyama step
        yt = Y[:,n] + A*dt + B*DW[n]

        # Evaluate function at estimate yt and t_n+1
        A1 = func(yt, tsteps[n+1],**kwargs)[0]

        # Compute solution at next time point
        Y[:,n+1] = Y[:,n] + A1*dt + B*DW[n] + 0.5*(BGamma - B)*(DW[n]**2 - dt)*sqrtdt**(-1)

    return Y

def rk_15(func,x0,tsteps,**kwargs):
    r"""
    Explicit order 1.5 strong Runge--Kutta method for SDEs with additive 
    (non-)autonomous scalar noise

    Parameters
    ----------
    func : callable (X,t,**kwargs)
        Returns drift `A` and diffusion `B` of the SDE. See Examples for details.  
    x0 : NumPy 1darray
        Initial condition 
    tsteps : NumPy 1darray
        Sequence of time points for which to solve (including initial time `t0`)
    **kwargs : keyword arguments
        Additional keyword arguments to be passed on to `func`. See `Examples` for details.  

    Returns
    -------
    Y : NumPy 2darray
        Approximate solution at timepoints given by `tsteps`. Format is 
                `Y[:,tk]` approximate solution at time `tk`
        Thus `Y` is a `numstate`-by-`timesteps` array

    Notes
    -----
    The general form of an SDE with additive (non-)autonomous scalar noise is 

    .. math:: (1) \qquad dX_t = A(X_t,t)dt + B(t)dW_t, \quad X(t_0) = x_0

    The method for solving the SDE (1) is described in Sec. 11.2 of 
    Kloeden, P.E., & Platen, E. (1999). `Numerical Solution of Stochastic Differential Equations.` 
    Berlin: Springer.

    Examples
    --------
    Consider the SDE system 

    .. math:: 

                     dV_t & = -\alpha \sin(V_t) \cos(t Z_t) + t \beta dW_t,\\
                     dZ_t & = -\alpha \cos(t V_t) \sin(Z_t) + t \gamma dW_t,\\
                     V_{t_0} & = 0.5, \quad Z_{t_0} = -0.5, \quad t_0 = 1,

    thus with :math:`X_t = (V_t,Z_t)` we have

    .. math:: 

                     A(X_t,t) & = (-\alpha \sin(V_t) \cos(t Z_t),-\alpha \cos(t V_t) \sin(Z_t)),\\
                     B(t)    & = (t \beta,t \gamma).

    Hence `func` would look like this:
    
    ::

                import numpy as np
                def myrhs(Xt,t,alpha=0.2,beta=0.01,gamma=0.02):
                        A = np.array([-alpha*np.sin(Xt[0])*np.cos(t*Xt[1]),
                                      -alpha*np.cos(t*Xt[0])*np.sin(Xt[1])])
                        B = np.array([t*beta,t*gamma])
                        return A,B

    Thus, the full call to `rk_15` to approximate the SDE system on :math:`[t_0,2]` could be 
    something like (assuming the function `myrhs` is defined in `myrhs.py`)

    >>> import numpy as np
    >>> from sde_solvers import rk_15
    >>> from myrhs import myrhs
    >>> Xt = rk_15(myrhs,np.array([0.5,-0.5]),np.arange(1,2,1e-3),beta=.02)

    Hence we used :math:`\beta = 0.02` in `myrhs` instead of the default value 0.01. 
    
    See also
    --------
    pc_15 : an implicit order 1.5 order strong Runge--Kutta method (it uses the method of ``rk_15`` 
            as predictor and the corresponding implicit update formula as corrector) for stiff SDEs 
    """

    # Check for correctness of input and allocate common tmp variables
    Y,dt,sqrtdt,zeta1,zeta2 = checkinput(func,x0,tsteps)

    # Generate pair of correlated normally distributed random variables and a linear combination of them
    DW  = zeta1*sqrtdt
    DZ  = 0.5*(zeta1 + np.sqrt(3)**(-1)*zeta2)*sqrtdt**3
    DWZ = dt*DW - DZ

    # More temp variables
    dt1     = dt**(-1)
    dt2     = 0.5*sqrtdt**(-1)
    dtplus  = 0.25*dt + dt2*DZ
    dtminus = 0.25*dt - dt2*DZ

    # Compute solution recursively
    for n in xrange(tsteps.size - 1):

        # Get drift/diffusion from func
        A, B   = func(Y[:,n], tsteps[n],**kwargs)
        Atmp   = Y[:,n] + A*dt
        Btmp   = B*sqrtdt
        Aplus  = func(Atmp + Btmp, tsteps[n+1],**kwargs)[0]
        Aminus = func(Atmp - Btmp, tsteps[n+1],**kwargs)[0]
        B1     = func(Y[:,n], tsteps[n+1],**kwargs)[1]

        # Compute solution at next time point
        Y[:,n+1] = Y[:,n] + 0.5*A*dt +B*DW[n] \
                   + Aplus*dtplus[n] + Aminus*dtminus[n] \
                   + dt1*(B1 - B)*DWZ[n]

    return Y

def pc_15(func,x0,tsteps,**kwargs):
    r"""
    Predictor-Corrector solver based on an 
    implicit order 1.5 strong Runge--Kutta method for SDEs with additive 
    (non-)autonomous scalar noise

    Parameters
    ----------
    func : callable (X,t,**kwargs)
        Returns drift `A` and diffusion `B` of the SDE. See Examples for details.  
    x0 : NumPy 1darray
        Initial condition 
    tsteps : NumPy 1darray
        Sequence of time points for which to solve (including initial time `t0`)
    **kwargs : keyword arguments
        Additional keyword arguments to be passed on to `func`. See `Examples` for details.  

    Returns
    -------
    Y : NumPy 2darray
        Approximate solution at timepoints given by `tsteps`. Format is 
                `Y[:,tk]` approximate solution at time `tk`
        Thus `Y` is a `numstate`-by-`timesteps` array

    Notes
    -----
    The general form of an SDE with additive (non-)autonomous scalar noise is 

    .. math:: (1) \qquad dX_t = A(X_t,t)dt + B(t)dW_t, \quad X(t_0) = x_0

    The code  implements a two-fold approach to approximate solutions of (1). At each time-step 
    :math:`t_n` an explicit order 1.5 strong Runge--Kutta method (compare ``rk_15``) is employed 
    to estimate the solution at time :math:`t_{n+1}` (predictor). This approximation is then used 
    in the implicit Runge--Kutta update formula of the same order (corrector). 
    The implicit Runge--Kutta method for solving the SDE (1) is described in Sec. 12.3 of 
    Kloeden, P.E., & Platen, E. (1999). `Numerical Solution of Stochastic Differential Equations`. 
    Berlin: Springer. For details on the explicit Runge--Kutta formula see the documentation of ``rk_15``. 

    Examples
    --------
    Consider the SDE system 

    .. math:: 

                     dV_t & = -\alpha \sin(V_t) \cos(t Z_t) + t \beta dW_t,\\
                     dZ_t & = -\alpha \cos(t V_t) \sin(Z_t) + t \gamma dW_t,\\
                     V_{t_0} & = 0.5, \quad Z_{t_0} = -0.5, \quad t_0 = 1,

    thus with :math:`X_t = (V_t,Z_t)` we have

    .. math:: 

                     A(X_t,t) & = (-\alpha \sin(V_t) \cos(t Z_t),-\alpha \cos(t V_t) \sin(Z_t)),\\
                     B(t)    & = (t \beta,t \gamma).

    Hence `func` would look like this:
    
    ::

                import numpy as np
                def myrhs(Xt,t,alpha=0.2,beta=0.01,gamma=0.02):
                        A = np.array([-alpha*np.sin(Xt[0])*np.cos(t*Xt[1]),
                                      -alpha*np.cos(t*Xt[0])*np.sin(Xt[1])])
                        B = np.array([t*beta,t*gamma])
                        return A,B

    Thus, the full call to `pc_15` to approximate the SDE system on :math:`[t_0,2]` could be 
    something like (assuming the function `myrhs` is defined in `myrhs.py`)

    >>> import numpy as np
    >>> from sde_solvers import pc_15
    >>> from myrhs import myrhs
    >>> Xt = pc_15(myrhs,np.array([0.5,-0.5]),np.arange(1,2,1e-3),beta=.02)

    Hence we used :math:`\beta = 0.02` in `myrhs` instead of the default value 0.01. 

    See also
    --------
    pc_1 : a lower order (but faster) implicit solver
    """

    # Check for correctness of input and allocate common tmp variables
    Y,dt,sqrtdt,zeta1,zeta2 = checkinput(func,x0,tsteps)

    # Generate pair of correlated normally distributed random variables and a linear combination of them
    DW    = zeta1*sqrtdt
    DZ    = 0.5*(zeta1 + np.sqrt(3)**(-1)*zeta2)*sqrtdt**3
    DWZ   = dt*DW - DZ
    DWZ2  = DZ - 0.5*dt*DW

    # More temp variables
    dt1 = dt**(-1)
    dt2 = 0.5*sqrtdt**(-1)
    dtplus  = 0.25*dt + dt2*DZ
    dtminus = 0.25*dt - dt2*DZ

    # Compute solution recursively
    for n in xrange(tsteps.size - 1):

        # Get drift/diffusion from func
        A, B    = func(Y[:,n], tsteps[n],**kwargs)
        Atmp    = Y[:,n] + A*dt
        Btmp    = B*sqrtdt
        Aplus   = func(Atmp + Btmp, tsteps[n+1],**kwargs)[0]
        Aminus  = func(Atmp - Btmp, tsteps[n+1],**kwargs)[0]
        Aplus1  = func(Atmp + Btmp, tsteps[n],**kwargs)[0]
        Aminus1 = func(Atmp - Btmp, tsteps[n],**kwargs)[0]
        B1      = func(Y[:,n], tsteps[n+1],**kwargs)[1]

        # Predictor step: explicit order 1.5 method
        Adiff = Aplus*dtplus[n] + Aminus*dtminus[n]
        # Adiff = 0.25*(Aplus + Aminus)*dt + dt2*(Aplus - Aminus)*DZ[n]
        yt    = Y[:,n] + 0.5*A*dt + B*DW[n] + Adiff + dt1*(B1 - B)*DWZ[n]

        # Use predicted value to evaluate function at Y_n+1,t_n+1
        A1 = func(yt, tsteps[n+1],**kwargs)[0]

        # Corrector step: implicit order 1.5 method
        Y[:,n+1] = yt + 0.5*A1*dt + dt2*(Aplus1 - Aminus1)*DWZ2[n] - Adiff

    return Y

def rk_2(func,x0,tsteps,strato_p=15,strato_q=30,**kwargs):
    r"""
    Explicit second order strong Runge--Kutta method for SDEs with additive 
    (non-)autonomous scalar noise

    Parameters
    ----------
    func : callable (X,t,**kwargs)
        Returns drift `A` and diffusion `B` of the SDE. See Examples for details.  
    x0 : NumPy 1darray
        Initial condition. 
    tsteps : NumPy 1darray
        Sequence of time points for which to solve (including initial time `t0`).
    strato_p : int
        Approximation order used to estimate the occuring multiple Stratonovich integrals. 
        Can be lowered to `strato_p = 1` in many cases, however, only change if you know 
        what you are doing. 
    strato_q : int
        Number of summands for partial sum approximation of Stratonovich integral
        coefficients. Can be lowered to `strato_q = 2` in many cases, however, only change 
        if you know what you are doing. 
    **kwargs : keyword arguments
        Additional keyword arguments to be passed on to `func`. See `Examples` for details.  

    Returns
    -------
    Y : NumPy 2darray
        Approximate solution at timepoints given by `tsteps`. Format is 
                `Y[:,tk]` approximate solution at time `tk`
        Thus `Y` is a `numstate`-by-`timesteps` array

    Examples
    --------
    Consider the SDE system 

    .. math:: 

                     dV_t & = -\alpha \sin(V_t) \cos(t Z_t) + t \beta dW_t,\\
                     dZ_t & = -\alpha \cos(t V_t) \sin(Z_t) + t \gamma dW_t,\\
                     V_{t_0} & = 0.5, \quad Z_{t_0} = -0.5, \quad t_0 = 1,

    thus with :math:`X_t = (V_t,Z_t)` we have

    .. math:: 

                     A(X_t,t) & = (-\alpha \sin(V_t) \cos(t Z_t),-\alpha \cos(t V_t) \sin(Z_t)),\\
                     B(t)    & = (t \beta,t \gamma).

    Hence `func` would look like this:
    
    ::

                import numpy as np
                def myrhs(Xt,t,alpha=0.2,beta=0.01,gamma=0.02):
                        A = np.array([-alpha*np.sin(Xt[0])*np.cos(t*Xt[1]),
                                      -alpha*np.cos(t*Xt[0])*np.sin(Xt[1])])
                        B = np.array([t*beta,t*gamma])
                        return A,B

    Thus, the full call to `rk_2` to approximate the SDE system on :math:`[t_0,2]` could be 
    something like (assuming the function `myrhs` is defined in `myrhs.py`)

    >>> import numpy as np
    >>> from sde_solvers import rk_2
    >>> from myrhs import myrhs
    >>> Xt = rk_2(myrhs,np.array([0.5,-0.5]),np.arange(1,2,1e-3),beta=.02)

    Hence we used :math:`\beta = 0.02` in `myrhs` instead of the default value 0.01. 

    Notes
    -----
    The general form of an SDE with additive (non-)autonomous scalar noise is 

    .. math:: (1) \qquad dX_t = A(X_t,t)dt + B(t)dW_t, \quad X(t_0) = x_0

    The method for solving the SDE (1) is described in Sec. 11.3 of 
    Kloeden, P.E., & Platen, E. (1999). `Numerical Solution of Stochastic Differential Equations.` 
    Berlin: Springer.
    
    See also
    --------
    pc_2 : an implicit second order method (it uses the method of ``rk_2`` as predictor 
           and the corresponding implicit update formula as corrector). 
    """

    # Check for correctness of input and allocate common tmp variables
    Y,dt,sqrtdt,zeta1,zeta2 = checkinput(func,x0,tsteps)

    # Compute Stratonovich integral approximation
    J,DW,DZ = get_stratonovich(strato_p,strato_q,dt,zeta1,tsteps.size-1)

    # More temp variables
    dt1 = dt**(-1)
    dt2 = 0.5*dt
    DWZ = dt*DW - DZ

    # This is going to be multiplied by the diffusion term
    Jtmp   = np.sqrt(np.abs(2*dt*J - DZ**2))
    Jplus  = dt1*(DZ + Jtmp)
    Jminus = dt1*(DZ - Jtmp)

    # Compute solution recursively
    for n in xrange(tsteps.size - 1):

        # Get drift/diffusion from func
        A, B   = func(Y[:,n], tsteps[n],**kwargs)
        Atmp   = Y[:,n] + A*dt2
        Aplus  = func(Atmp + B*Jplus[n], tsteps[n] + dt2,**kwargs)[0]
        Aminus = func(Atmp + B*Jminus[n], tsteps[n] + dt2,**kwargs)[0]
        B1     = func(Y[:,n], tsteps[n+1],**kwargs)[1]

        # Solution at t_n+1
        Y[:,n+1] = Y[:,n] + (Aplus + Aminus)*dt2 + B*DW[n] + dt1*(B1 - B)*DWZ[n]

    return Y

def pc_2(func,x0,tsteps,strato_p=15,strato_q=30,**kwargs):
    r"""
    Predictor-Corrector solver based on an 
    implicit second order strong Runge--Kutta method for SDEs with additive 
    (non-)autonomous scalar noise

    Parameters
    ----------
    func : callable (X,t,**kwargs)
        Returns drift `A` and diffusion `B` of the SDE. See Examples for details.  
    x0 : NumPy 1darray
        Initial condition 
    tsteps : NumPy 1darray
        Sequence of time points for which to solve (including initial time `t0`)
    **kwargs : keyword arguments
        Additional keyword arguments to be passed on to `func`. See `Examples` for details.  

    Returns
    -------
    Y : NumPy 2darray
        Approximate solution at timepoints given by `tsteps`. Format is 
                `Y[:,tk]` approximate solution at time `tk`
        Thus `Y` is a `numstate`-by-`timesteps` array

    Notes
    -----
    The general form of an SDE with additive (non-)autonomous scalar noise is 

    .. math:: (1) \qquad dX_t = A(X_t,t)dt + B(t)dW_t, \quad X(t_0) = x_0

    The code  implements a two-fold approach to approximate solutions of (1). At each time-step 
    :math:`t_n` an explicit second order strong Runge--Kutta method (compare `rk_2`) is employed to 
    estimate the solution at time :math:`t_{n+1}` (predictor). This approximation is then used 
    in the implicit Runge--Kutta update formula of the same order (corrector). 
    The implicit Runge--Kutta method for solving the SDE (1) is described in Sec. 12.3 of 
    Kloeden, P.E., & Platen, E. (1999). `Numerical Solution of Stochastic Differential Equations.` 
    Berlin: Springer. For details on the explicit Runge--Kutta formula see the documentation of `rk_2`. 

    Examples
    --------
    Consider the SDE system 

    .. math:: 

                     dV_t & = -\alpha \sin(V_t) \cos(t Z_t) + t \beta dW_t,\\
                     dZ_t & = -\alpha \cos(t V_t) \sin(Z_t) + t \gamma dW_t,\\
                     V_{t_0} & = 0.5, \quad Z_{t_0} = -0.5, \quad t_0 = 1,

    thus with :math:`X_t = (V_t,Z_t)` we have

    .. math:: 

                     A(X_t,t) & = (-\alpha \sin(V_t) \cos(t Z_t),-\alpha \cos(t V_t) \sin(Z_t)),\\
                     B(t)    & = (t \beta,t \gamma).

    Hence `func` would look like this:
    
    ::

                import numpy as np
                def myrhs(Xt,t,alpha=0.2,beta=0.01,gamma=0.02):
                        A = np.array([-alpha*np.sin(Xt[0])*np.cos(t*Xt[1]),
                                      -alpha*np.cos(t*Xt[0])*np.sin(Xt[1])])
                        B = np.array([t*beta,t*gamma])
                        return A,B

    Thus, the full call to `pc_2` to approximate the SDE system on :math:`[t_0,2]` could be 
    something like (assuming the function `myrhs` is defined in `myrhs.py`)

    >>> import numpy as np
    >>> from sde_solvers import pc_2
    >>> from myrhs import myrhs
    >>> Xt = pc_2(myrhs,np.array([0.5,-0.5]),np.arange(1,2,1e-3),beta=.02)

    Hence we used :math:`\beta = 0.02` in `myrhs` instead of the default value 0.01. 

    See also
    --------
    pc_15 : a lower order (but faster) implicit solver
    """

    # Check for correctness of input and allocate common tmp variables
    Y,dt,sqrtdt,zeta1,zeta2 = checkinput(func,x0,tsteps)

    # Compute Stratonovich integral approximation
    J,DW,DZ = get_stratonovich(strato_p,strato_q,dt,zeta1,tsteps.size-1)

    # More temp variables
    dt1 = dt**(-1)
    dt2 = 0.5*dt
    DWZ = dt*DW - DZ
    DZZ = 0.5*DZ + 0.25*dt*DW

    # This is going to be multiplied by the diffusion term
    Jtmp    = np.sqrt(np.abs(2*dt*J - DZ**2))
    Jplus   = dt1*(DZ + Jtmp)
    Jminus  = dt1*(DZ - Jtmp)
    Jtmp    = np.sqrt(np.abs(dt*J - 0.25*DZ**2 + 0.125*dt**2*(DW**2 + 0.5*(2*DZ*dt**(-1) - DW)**2)))
    Jplus1  = dt*(DZZ + Jtmp)
    Jminus1 = dt*(DZZ - Jtmp)

    # Compute solution recursively
    for n in xrange(tsteps.size - 1):

        # Get drift/diffusion from func
        A, B    = func(Y[:,n], tsteps[n],**kwargs)
        Atmp    = Y[:,n] + A*dt2
        Aplus   = func(Atmp + B*Jplus[n], tsteps[n] + dt2,**kwargs)[0]
        Aminus  = func(Atmp + B*Jminus[n], tsteps[n] + dt2,**kwargs)[0]
        Aplus1  = func(Atmp + B*Jplus1[n], tsteps[n],**kwargs)[0]
        Aminus1 = func(Atmp + B*Jminus1[n], tsteps[n],**kwargs)[0]
        B1      = func(Y[:,n], tsteps[n+1],**kwargs)[1]
        Atmp    = (Aplus + Aminus)*dt2

        # Predictor
        yt = Y[:,n] + Atmp + B*DW[n] + dt1*(B1 - B)*DWZ[n]

        # Evaluate drift at predicted yt
        A1 = func(yt, tsteps[n], **kwargs)[0]

        # Corrector
        Y[:,n+1] = yt - Atmp + dt*(Aplus1 + Aminus1 - 0.5*(A1 + A))

    return Y

def get_stratonovich(p,q,dt,zeta1,tsize):
    """
    Function used internally by the second order solvers
    to approximate multiple Stratonovich integrals
    """

    # Sanity checks
    try:
        ptest = (p == int(p))
    except: raise TypeError("Input strato_p must be an integer!")
    if ptest == False: raise ValueError("Input strato_p must be an integer")
    if p <= 0: raise ValueError("Input strato_p must be >0!")

    try:
        qtest = (q == int(q))
    except: raise TypeError("Input strato_q must be an integer!")
    if qtest == False: raise ValueError("Input strato_q must be an integer")
    if q <= 1: raise ValueError("Input strato_q must be >1!")

    # Coefficients for approximations below
    rho = 0
    for r in xrange(1,p+1):
        rho += r**(-2)
    rho = 1/12 - (2*np.pi**2)**(-1)*rho

    al = 0
    for r in xrange(1,p+1):
        al += r**(-4)
    al = np.pi**2/180 - (2*np.pi**2)**(-1)*al

    # Allocate memory for random variables
    Xi  = np.zeros((p,tsize))
    Eta = np.zeros((p,tsize))

    # Generate standard Gaussian variables
    sc    = np.sqrt(dt/(2*np.pi**2))
    dttmp = np.sqrt(2/dt)*np.pi
    for r in xrange(1,p+1):
        Xi[r-1,:]  = dttmp*norm.rvs(size=(tsize,),loc=0,scale=sc*r**(-1))
        Eta[r-1,:] = dttmp*norm.rvs(size=(tsize,),loc=0,scale=sc*r**(-1))

    mu  = np.zeros((tsize,))
    phi = np.zeros((tsize,))
    for r in xrange(p+1,p+2+q):
        mu  += norm.rvs(size=(tsize,),loc=0,scale=sc*r**(-1))
        phi += norm.rvs(size=(tsize,),loc=0,scale=sc*r**(-2))
    mu  = np.sqrt(dt*rho)**(-1)*mu
    phi = np.sqrt(dt*al)**(-1)*phi

    # Approximation of Stratonovich stochastic integrals
    a10 = np.zeros((tsize,))
    for r in xrange(1,p+1):
        a10 += r**(-1)*Xi[r-1,:]
    a10 = -np.pi**(-1)*np.sqrt(2*dt)*a10 - 2*np.sqrt(dt*rho)*mu

    b1 = np.zeros((tsize,))
    for r in xrange(1,p+1):
        b1 += r**(-2)*Eta[r-1,:]
    b1 = np.sqrt(0.5*dt)*b1 + np.sqrt(dt*al)*phi

    C = np.zeros((tsize,))
    for r in xrange(1,p+1):
        for l in xrange(1,p+1):
            if r != l:
                C += r/(r**2 - l**2)*(l**(-1)*Xi[r-1,:]*Xi[l-1,:] - l/r*Eta[r-1,:]*Eta[l-1,:])
    C = -(2*np.pi**2)**(-1)*C

    # Everything so far was done to compute this monster: a double Stratonovich integral...
    J = 1/6*dt**2*zeta1**2 + 0.25*dt*a10**2 - (2*np.pi)**(-1)*dt**(1.5)*zeta1*b1 \
        + 0.25*dt**(1.5)*a10*zeta1 - dt**2*C

    # zeta1 = zeta1*sqrtdt**(-1)
    DW  = zeta1*np.sqrt(dt)
    DZ  = 0.5*dt*(zeta1*np.sqrt(dt) + a10)

    return J, DW, DZ

def checkinput(func,x0,tsteps):
    """
    Function used internally by all solvers of this module to perform sanity checks and allocate stuff
    """
    
    # Sanity checks
    if type(func).__name__ != 'function' and type(func).__name__ != 'builtin_function_or_method':
        raise TypeError("First argument has to be a valid Python function!")

    try:
        x0s = x0.shape
    except:
        raise TypeError("Input x0 must be a NumPy 1darray, not "+type(x0).__name__+"!")
    if len(x0s) > 2 or (len(x0s)==2 and min(x0s)>1):
        raise ValueError("Input x0 must be a NumPy 1darray!")
    if np.isnan(x0).max()==True or np.isinf(x0).max()==True or np.isreal(x0).min()==False:
        raise ValueError('Input x0 must be a real valued NumPy array without Infs or NaNs!')
        
    try:
        tstepss = tsteps.shape
    except:
        raise TypeError("Input tsteps must be a NumPy 1darray, not "+type(tsteps).__name__+"!")
    if len(tstepss) > 1:
        raise ValueError("Input tsteps must be a NumPy 1darray!")
    if np.isnan(tsteps).max()==True or np.isinf(tsteps).max()==True or np.isreal(tsteps).min()==False:
        raise ValueError('Input tsteps must be a real valued NumPy array without Infs or NaNs!')

    # Allocate temp variables
    Y = np.zeros((x0.size,tsteps.size))

    # First component of solution is IC
    Y[:,0] = x0

    # Time step size and its square root (=std of stochastic terms)
    dt     = tsteps[1] - tsteps[0]
    sqrtdt = np.sqrt(dt)

    # Generate i.i.d. normal random variables with mean=0 (loc) and std=1 (scale) (Var=std^2)
    zeta1 = norm.rvs(size=(tsteps.size-1,),loc=0,scale=1)
    zeta2 = norm.rvs(size=(tsteps.size-1,),loc=0,scale=1)

    return Y,dt,sqrtdt,zeta1,zeta2

