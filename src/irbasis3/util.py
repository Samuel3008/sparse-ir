import numpy as np


def high_freq_moment(gl, basis, num_moments, axis=0):
    """Compute high-frequency moments of Green's function

    G(iv) \simeq \sum_{n=1}^N G_n/(iv)^n, where
       G_n = (-1)^n (G^{n-1}(0^+) - G^{n-1}(0^-)).

    Attributes:
    -----------
     - `gl` : Expansion coefficients of Green's function in IR. Three-/one dimensional array.
     - `basis` : IR Basis instance
     - `num_moments` : Number of moments to be computed (>=1)
     - `axis` : Axis of gl corresponding to IR 
    
    Return list `[G_1, G_2, ...]`,
    where G_n are the computed high-frequency moments (each of them is a numpy.ndarray instance).
    """
    assert gl.ndim in [1, 3]

    beta = basis.beta
    stat_sign = -1 if basis.statistics == "F" else 1

    _eval = lambda u, gl, tau: np.tensordot(u(tau), gl, axes=(0,axis))

    res = []
    u_ = basis.u
    for n_ in range(num_moments):
        res.append(
            ((-1)**(n_+1)) *
            (_eval(u_, gl, 0) - stat_sign * _eval(u_, gl, beta))
        )
        u_ = u_.deriv()
    return res


