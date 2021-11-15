# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from warnings import warn
from .high_freq import evalulator_high_freq_moment

class SamplingBase:
    """Base class for sparse sampling.

    Encodes the "basis transformation" of a propagator from the truncated IR
    basis coefficients `G_ir[l]` to time/frequency sampled on sparse points
    `G(x[i])` together with its inverse, a least squares fit::

             ________________                   ___________________
            |                |    evaluate     |                   |
            |     Basis      |---------------->|     Value on      |
            |  coefficients  |<----------------|  sampling points  |
            |________________|      fit        |___________________|

    Optionally, in fitting, we can use known high-frequency moments as linear equality conditions:
     G(iv) \simeq \sum_{n=1}^N G_n/(iv)^n, where
           G_n = (-1)^n (G^{n-1}(tau=0^+) - G^{n-1}(tau=0^-)).

    Attributes:
    -----------
     - `basis` : IR Basis instance
     - `matrix` : Evaluation matrix is decomposed form
     - `sampling_points` : Set of sampling points
     - `warn_cond` : Warn if the condition number is too large (>1e8)
     - `known_moments` : Known high-frequency moments given as a dictionary.
        For example, {1: G1, 2: G2}.
    """
    def __init__(self, basis, sampling_points=None, warn_cond=True, known_moments=None):
        if sampling_points is None:
            sampling_points = self.__class__.default_sampling_points(basis)
        else:
            sampling_points = np.array(sampling_points)
        

        self.basis = basis
        self.sampling_points = sampling_points

        if known_moments is None:
            self.matrix = DecomposedMatrix(self.__class__.eval_matrix(basis, sampling_points))
        else:
            assert isinstance(known_moments, dict)
            c_ = []
            d_ = []
            for n, Gn in known_moments.items():
                c_.append(evalulator_high_freq_moment(basis, n))
                d_.append(Gn)
            self.matrix = DecomposedMatrixContrainedFitting(
                self.__class__.eval_matrix(basis, sampling_points),
                np.vstack(c_), np.array(d_))

        # Check conditioning
        self.cond = self.matrix.s[0] / self.matrix.s[-1]
        if warn_cond and self.cond > 1e8:
            warn("Sampling matrix is poorly conditioned (cond = %.2g)"
                 % self.cond, ConditioningWarning)

    def evaluate(self, al, axis=None):
        """Evaluate the basis coefficients at the sparse sampling points"""
        return self.matrix.matmul(al, axis)

    def fit(self, ax, axis=None):
        """Fit basis coefficients from the sparse sampling points"""
        return self.matrix.lstsq(ax, axis)

    @classmethod
    def default_sampling_points(cls, basis):
        """Return default sampling points"""
        raise NotImplementedError()

    @classmethod
    def eval_matrix(cls, basis, x):
        """Return evaluation matrix from coefficients to sampling points"""
        raise NotImplementedError()


class TauSampling(SamplingBase):
    """Sparse sampling in imaginary time.

    Allows the transformation between the IR basis and a set of sampling points
    in (scaled/unscaled) imaginary time.
    """
    @classmethod
    def default_sampling_points(cls, basis):
        poly = basis.u[-1]
        maxima = poly.deriv().roots()
        left = .5 * (maxima[:1] + poly.xmin)
        right = .5 * (maxima[-1:] + poly.xmax)
        return np.concatenate([left, maxima, right])

    @classmethod
    def eval_matrix(cls, basis, x):
        return basis.u(x).T

    @property
    def tau(self):
        """Sampling points in (reduced) imaginary time"""
        return self.sampling_points


class MatsubaraSampling(SamplingBase):
    """Sparse sampling in Matsubara frequencies.

    Allows the transformation between the IR basis and a set of sampling points
    in (scaled/unscaled) imaginary frequencies.

    Attributes:
    -----------
     - `basis` : IR Basis instance
     - `matrix` : Evaluation matrix is decomposed form
     - `sampling_points` : Set of sampling points
    """
    @classmethod
    def default_sampling_points(cls, basis, mitigate=True):
        # Use the (discrete) extrema of the corresponding highest-order basis
        # function in Matsubara.  This turns out to be close to optimal with
        # respect to conditioning for this size (within a few percent).
        polyhat = basis.uhat[-1]
        wn = polyhat.extrema()

        # While the condition number for sparse sampling in tau saturates at a
        # modest level, the conditioning in Matsubara steadily deteriorates due
        # to the fact that we are not free to set sampling points continuously.
        # At double precision, tau sampling is better conditioned than iwn
        # by a factor of ~4 (still OK). To battle this, we fence the largest
        # frequency with two carefully chosen oversampling points, which brings
        # the two sampling problems within a factor of 2.
        if mitigate:
            wn_outer = wn[[0, -1]]
            wn_diff = 2 * np.round(0.03 * wn_outer).astype(int)
            if wn.size >= 20:
                wn = np.hstack([wn, wn_outer - wn_diff])
            if wn.size >= 42:
                wn = np.hstack([wn, wn_outer + wn_diff])
            wn = np.unique(wn)

        return wn

    @classmethod
    def eval_matrix(cls, basis, x):
        return basis.uhat(x).T

    @property
    def wn(self):
        """Sampling points as (reduced) Matsubara frequencies"""
        return self.sampling_points


class FittingMatrix:
    """Fitting matrix in SVD decomposed form for fast and accurate fitting.
    Stores a matrix `A` 
    """
    def __init__(self, a):
        a = np.asarray(a)
        if a.ndim != 2:
            raise ValueError("a must be of matrix form")
        self.a = a

    def __matmul__(self, x):
        """Matrix-matrix multiplication along the first axis"""
        return np.einsum('ij,j...->i...', self.a, x, optimize=True)

    def matmul(self, x, axis=None):
        """Compute `A @ x` (optionally along specified axis of x)"""
        if axis is None:
            return self @ x

        x = np.asarray(x)
        target_axis = 0
        x = np.moveaxis(x, axis, target_axis)
        r = self @ x
        return np.moveaxis(r, target_axis, axis)

    def lstsq(self, x, axis=None):
        """Return `y` such that `np.linalg.norm(A @ y - x)` is minimal"""
        if axis is None:
            return self._lstsq(x)

        x = np.asarray(x)
        target_axis = 0
        x = np.moveaxis(x, axis, target_axis)
        r = self._lstsq(x)
        return np.moveaxis(r, target_axis, axis)

class DecomposedMatrix(FittingMatrix):
    """Matrix in SVD decomposed form for fast and accurate fitting.

    Stores a matrix `A` together with its thin SVD form: `A == (u * s) @ vt`.
    This allows for fast and accurate least squares fits using `A.lstsq(x)`.
    """
    @classmethod
    def get_svd_result(cls, a, eps=None):
        """Construct decomposition from matrix"""
        u, s, vH = np.linalg.svd(a, full_matrices=False)
        where = s.astype(bool) if eps is None else s/s[0] <= eps
        if not where.all():
            return u[:, where], s[where], vH[where]
        else:
            return u, s, vH

    def __init__(self, a, svd_result=None):
        super().__init__(a)
        if svd_result is None:
            u, s, vt = self.__class__.get_svd_result(a)
        else:
            u, s, vt = map(np.asarray, svd_result)

        self.a = a
        self.u = u
        self.s = s
        self.vt = vt


    def matmul(self, x, axis=None):
        """Compute `A @ x` (optionally along specified axis of x)"""
        if axis is None:
            return self @ x

        x = np.asarray(x)
        target_axis = 0
        x = np.moveaxis(x, axis, target_axis)
        r = self @ x
        return np.moveaxis(r, target_axis, axis)

    def _lstsq(self, x):
        r = np.einsum('ij,j...->i...', self.u.conj().T, x, optimize=True)
        r = np.einsum('i...,i->i...', r, 1/self.s)
        return np.einsum('ij,j...->i...', self.vt.conj().T, r, optimize=True)

    def __array__(self, dtype=None):
        """Convert to numpy array."""
        return self.a.astype(dtype)


class DecomposedMatrixContrainedFitting(FittingMatrix):
    """ DecomposedMatrix with linear equality constraints

    For computing argmin_{x} (1/2) |a @ x - b|^2, 
    subject to the linear equality contrain c @ x = d.

    Mathematically, the solution is given by
      [ a^\dagger a   c^\dagger ] [x] = [a^dagger b]
      [     c             0     ] [v]   [ d ].
    
    Because `a` is squared, the condition number of the extended coefficient matrix
    may be worse than the unconstrained counterpart.
    """
    def __init__(self, a, c, d):
        super().__init__(a)

        assert c.ndim == 2
        assert c.shape[1] == a.shape[1]
        num_const = c.shape[0]
        assert d.shape[0] == num_const
        self._decomposed_matrix = DecomposedMatrix(
            np.block([
                [a.T.conjugate()@a,  c.T.conjugate()],
                [c                ,  np.zeros((num_const, num_const))]
            ])
        )
        self._d = d
    
    @property
    def s(self):
        return self._decomposed_matrix.s

    def _lstsq(self, x):
        """ Fit along the first axis """
        at_x = np.einsum('ij,j...->i...',  self.a.T.conjugate(), x, optimize=True)
        x_ = np.concatenate([at_x, self._d], axis=0)
        return self._decomposed_matrix.lstsq(x_)[0:self.a.shape[1]]

class ConditioningWarning(RuntimeWarning):
    pass