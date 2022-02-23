# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from .kernel import LogisticKernel, RegularizedBoseKernel
from .sampling import DecomposedMatrix
from .basis import FiniteTempBasis
from typing import Optional

import xprec
from xprec import ddouble
class MatsubaraPoleBasis:
    def __init__(self, beta: float, poles: np.ndarray):
        self._beta = beta
        self._poles = np.array(poles)

    def __call__(self, n: np.ndarray) -> np.ndarray:
        """Evaluate basis functions at given frequency n"""
        iv = 1j*n * np.pi/self._beta
        return 1/(iv[None, :] - self._poles[:, None])


class TauPoleBasis:
    def __init__(self, beta: float, statistics: str, poles: np.ndarray):
        self._beta = beta
        self._statistics = statistics
        self._poles = np.array(poles)
        self._wmax = np.abs(poles).max()

    def __call__(self, tau) -> np.ndarray:
        """ Evaluate basis functions at tau """
        tau = np.asarray(tau)
        if (tau < 0).any() or (tau > self._beta).any():
            raise RuntimeError("tau must be in [0, beta]!")

        x = 2 * tau/self._beta - 1
        y = self._poles/self._wmax
        lambda_ = self._beta * self._wmax

        if self._statistics == "F":
            res = -LogisticKernel(lambda_)(x[:, None], y[None, :])
        else:
            K = RegularizedBoseKernel(lambda_)
            res = -K(x[:, None], y[None, :])/y[None, :]
        return res.T


class SparsePoleRepresentation:
    """
    Sparse pole representation (SPR)
    The poles are the extrema of V'_{L-1}(ω) and +/- wmax.
    """
    def __init__(
            self, basis: FiniteTempBasis,
            sampling_points: Optional[np.ndarray] = None):
        self._basis = basis

        self._poles = basis.default_omega_sampling_points() \
            if sampling_points is None else np.asarray(sampling_points)
        self._y_sampling_points = basis.beta*self._poles/basis.wmax

        self.u = TauPoleBasis(basis.beta, basis.statistics, self._poles)
        self.uhat = MatsubaraPoleBasis(basis.beta, self._poles)

        # Fitting matrix from IR
        weight = \
            basis.kernel.weight_func(self.statistics)(self._y_sampling_points)
        work_dtype = ddouble
        #work_dtype = np.float64
        s = np.asarray(basis.s, dtype=work_dtype)
        v = np.asarray(basis.v(self._poles), dtype=work_dtype)
        weight = np.asarray(weight, dtype=work_dtype)

        fit_mat = -s[:, None] * v * weight[None, :]
        fit_mat2 = v * weight[None, :]
        self.svd_result_dd = xprec.linalg.svd(fit_mat)
        self.svd_result_dd2 = xprec.linalg.svd(fit_mat2)
        self.matrix = DecomposedMatrix(np.asarray(fit_mat, dtype=np.float64))
        self.matrix2 = DecomposedMatrix(np.asarray(fit_mat2, dtype=np.float64))

    @property
    def statistics(self):
        return self.basis.statistics

    @property
    def sampling_points(self):
        return self._poles

    @property
    def size(self):
        """Number of basis functions / singular values."""
        return self._poles.size

    @property
    def basis(self) -> FiniteTempBasis:
        """ Underlying basis """
        return self._basis

    @property
    def beta(self) -> float:
        """Inverse temperature (this is `None` because unscaled basis)"""
        return self.basis.beta

    @property
    def wmax(self) -> float:
        """Frequency cutoff (this is `None` because unscaled basis)"""
        return self.basis.wmax

    def from_IR(self, gl: np.ndarray, axis=0) -> np.ndarray:
        """
        From IR to SPR

        gl:
            Expansion coefficients in IR
        """
        u, s, vt = self.svd_result_dd
        r = u.T @ gl
        r = r / (s[:, None] if r.ndim > 1 else s)
        return np.asarray(vt.T @ r, dtype=np.float64)

    def from_rhol(self, rhol: np.ndarray, axis=0) -> np.ndarray:
        """
        From IR to SPR

        gl:
            Expansion coefficients in IR
        """
        u, s, vt = self.svd_result_dd2
        r = u.T @ rhol
        r = r / (s[:, None] if r.ndim > 1 else s)
        return np.asarray(vt.T @ r, dtype=np.float64)

    def to_IR(self, g_spr: np.ndarray, axis=0) -> np.ndarray:
        """
        From SPR to IR

        g_spr:
            Expansion coefficients in SPR
        """
        return self.matrix.matmul(g_spr, axis)

    def default_tau_sampling_points(self):
        """Default sampling points on the imaginary time/x axis"""
        return self.basis.default_tau_sampling_points()

    def default_matsubara_sampling_points(self, *, mitigate=True):
        """Default sampling points on the imaginary frequency axis"""
        return self.basis.default_matsubara_sampling_points(mitigate= mitigate)


#def expand_IR_in_SPR(spr):
#    return spr.from_rhol(np.identity(spr.basis.size, dtype=ddouble))

def expand_IR_in_SPR(spr):
    basis = spr._basis
    tau = basis.default_tau_sampling_points()

    y = basis.u(tau).T
    A = spr.u(np.array(tau, dtype=ddouble)).T
    u, s, vt = xprec.linalg.svd(A)
    u = u[:, 0:s.size]
    vt = vt[0:s.size, :]
    r = u.T @ y
    r = r / (s[:, None] if r.ndim > 1 else s)
    return np.asarray(vt.T @ r, dtype=np.float64)