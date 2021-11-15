# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np

import irbasis3
from irbasis3 import sampling
import pytest


def test_decomp():
    rng = np.random.RandomState(4711)
    A = rng.randn(49, 39)

    Ad = sampling.DecomposedMatrix(A)
    norm_A = Ad.s[0] / Ad.s[-1]
    np.testing.assert_allclose(A, np.asarray(Ad), atol=1e-15 * norm_A, rtol=0)

    x = rng.randn(39)
    np.testing.assert_allclose(A @ x, Ad @ x, atol=1e-14 * norm_A, rtol=0)

    x = rng.randn(39, 3)
    np.testing.assert_allclose(A @ x, Ad @ x, atol=1e-14 * norm_A, rtol=0)

    y = rng.randn(49)
    np.testing.assert_allclose(np.linalg.lstsq(A, y, rcond=None)[0],
                               Ad.lstsq(y), atol=1e-14 * norm_A, rtol=0)

    y = rng.randn(49, 2)
    np.testing.assert_allclose(np.linalg.lstsq(A, y, rcond=None)[0],
                               Ad.lstsq(y), atol=1e-14 * norm_A, rtol=0)


def test_axis():
    rng = np.random.RandomState(4712)
    A = rng.randn(17, 21)

    Ad = sampling.DecomposedMatrix(A)
    norm_A = Ad.s[0] / Ad.s[-1]

    x = rng.randn(2, 21, 4, 7)
    ref = np.tensordot(A, x, (-1,1)).transpose((1,0,2,3))
    np.testing.assert_allclose(
            Ad.matmul(x, axis=1), ref,
            atol=1e-13 * norm_A, rtol=0)

def test_axis0():
    rng = np.random.RandomState(4712)
    A = rng.randn(17, 21)

    Ad = sampling.DecomposedMatrix(A)
    norm_A = Ad.s[0] / Ad.s[-1]

    x = rng.randn(21, 2)

    np.testing.assert_allclose(
            Ad.matmul(x, axis=0), A@x,
            atol=1e-13 * norm_A, rtol=0)

    np.testing.assert_allclose(
            Ad.matmul(x), A@x,
            atol=1e-13 * norm_A, rtol=0)


def test_tau_noise():
    K = irbasis3.KernelFFlat(100)
    basis = irbasis3.IRBasis(K, 'F')
    smpl = irbasis3.TauSampling(basis)
    rng = np.random.RandomState(4711)

    rhol = basis.v([-.999, -.01, .5]) @ [0.8, -.2, 0.5]
    Gl = basis.s * rhol
    Gl_magn = np.linalg.norm(Gl)
    Gtau = smpl.evaluate(Gl)

    noise = 1e-5
    Gtau_n = Gtau +  noise * np.linalg.norm(Gtau) * rng.randn(*Gtau.shape)
    Gl_n = smpl.fit(Gtau_n)

    np.testing.assert_allclose(Gl, Gl_n, atol=12 * noise * Gl_magn, rtol=0)


def test_wn_noise():
    K = irbasis3.KernelBFlat(99)
    basis = irbasis3.IRBasis(K, 'B')
    smpl = irbasis3.MatsubaraSampling(basis)
    rng = np.random.RandomState(4711)

    rhol = basis.v([-.999, -.01, .5]) @ [0.8, -.2, 0.5]
    Gl = basis.s * rhol
    Gl_magn = np.linalg.norm(Gl)
    Giw = smpl.evaluate(Gl)

    noise = 1e-5
    Giw_n = Giw +  noise * np.linalg.norm(Giw) * rng.randn(*Giw.shape)
    Gl_n = smpl.fit(Giw_n)
    np.testing.assert_allclose(Gl, Gl_n, atol=12 * noise * Gl_magn, rtol=0)


def _randn(*shape, dtype=np.float64):
    if dtype == np.float64:
        return np.random.randn(*shape)
    elif dtype == np.complex128:
        return np.random.randn(*shape) + 1J*np.random.randn(*shape)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_constrained_lstsq(dtype):
    """
    Comptue argmin_{x} (1/2) |a @ x - b|^2, 
    subject to the linear equality contrain c @ x = d.
    """
    np.random.seed(100)
    M, N = 2, 4
    num_const = N - M
    a = _randn(M, N, dtype=dtype)
    b = _randn(M, dtype=dtype)
    c = _randn(num_const, N, dtype=dtype)
    d = _randn(num_const, dtype=dtype)

    dmat = sampling.DecomposedMatrixContrainedFitting(a, c, d)
    x = dmat.lstsq(b)

    assert np.abs(a@x - b).max() < 1e-13
    assert np.abs(c@x - d).max() < 1e-13

def test_wn_known_moment():
    lambda_ = 100
    beta = 100
    wmax = lambda_/beta

    # 1/(iv - H) \simeq 1/iv + H/(iv)^2 + ...
    H = np.array([[0.0, 0.1], [0.1, 0.2]])
    #H = np.array([[0.1]])
    nf = H.shape[0]
    evals, _ = np.linalg.eigh(H)
    assert all(np.abs(evals) < wmax)
    known_moments = {
        1: np.identity(nf),
        #2: H
    }

    K = irbasis3.KernelFFlat(lambda_)
    basis = irbasis3.FiniteTempBasis(K, 'F', beta)
    smpl = irbasis3.MatsubaraSampling(basis, known_moments=known_moments)
    #smpl = irbasis3.MatsubaraSampling(basis)

    iv = np.einsum('w,ij->wij', 1J * smpl.sampling_points * np.pi/beta, np.identity(nf))
    giv_smpl = np.linalg.inv(iv - H[None,:,:])
    #giv_smpl = np.zeros((smpl.sampling_points.size, nf, nf), dtype=np.complex128)

    #print("basis", basis.size)
    #print("giv_smpl", giv_smpl.shape)
    gl = smpl.fit(giv_smpl)

    giv_reconst = smpl.evaluate(gl)

    print(np.abs(giv_smpl - giv_reconst).max())