import sparse_ir
from sparse_ir.spr import SparsePoleRepresentation, expand_IR_in_SPR
from sparse_ir.sampling import MatsubaraSampling, TauSampling
import numpy as np
import pytest
from xprec import ddouble


@pytest.mark.parametrize("stat", ["F", "B"])
def test_compression(stat):
    beta = 1e+4
    wmax = 1
    eps = 1e-12
    basis = sparse_ir.FiniteTempBasis(stat, beta, wmax, eps=eps)
    spr = SparsePoleRepresentation(basis)

    np.random.seed(4711)

    num_poles = 10
    poles = wmax * (2*np.random.rand(num_poles) - 1)
    coeffs = 2*np.random.rand(num_poles) - 1
    assert np.abs(poles).max() <= wmax

    Gl = SparsePoleRepresentation(basis, poles).to_IR(coeffs)

    g_spr = spr.from_IR(Gl)

    # Comparison on Matsubara frequencies
    smpl = MatsubaraSampling(basis)
    smpl_for_spr = MatsubaraSampling(spr, smpl.sampling_points)

    giv = smpl_for_spr.evaluate(g_spr)

    giv_ref = smpl.evaluate(Gl, axis=0)

    np.testing.assert_allclose(giv, giv_ref, atol=300*eps, rtol=0)

    # Comparison on tau
    smpl_tau = TauSampling(basis)
    gtau = smpl_tau.evaluate(Gl)

    smpl_tau_for_spr = TauSampling(spr)
    gtau2 = smpl_tau_for_spr.evaluate(g_spr)

    np.testing.assert_allclose(gtau, gtau2, atol=300*eps, rtol=0)

@pytest.mark.parametrize("stat", ["F"])
def test_save_IR_basis(stat):
    beta = 1e+5
    wmax = 1.0
    eps = 1e-25
    u, s, v = sparse_ir.sve.compute(
        sparse_ir.LogisticKernel(beta * wmax), eps,
        work_dtype=ddouble,
        dtype=ddouble
    )
    basis_dd = sparse_ir.FiniteTempBasis(stat, beta, wmax, sve_result=(u, s, v))
    basis = sparse_ir.FiniteTempBasis(stat, beta, wmax)
    spr = SparsePoleRepresentation(basis_dd)
    shift = {"F": 1, "B": 0}[stat]

    u_IR_in_spr = expand_IR_in_SPR(spr)
    for j in range(u_IR_in_spr.shape[1]):
        for i in range(u_IR_in_spr.shape[0]):
            print(i, j, np.abs(u_IR_in_spr[i, j]))
        print()

    v = 2*np.array([10**p for p in np.arange(5)]) + shift
    iv = 1j * v * np.pi/beta

    uiv_spr = 1/(iv[:, None] - spr.sampling_points[None, :])

    uiv_IR = uiv_spr @ u_IR_in_spr
    uiv_IR_ref = basis.uhat(v).T

    print()
    for l in range(basis.size):
        diff = np.abs(uiv_IR[:, l] - uiv_IR_ref[:, l]).max()
        maxmag = np.abs(uiv_IR_ref[:, l]).max()
        print(l, basis.s[l], diff/maxmag, diff, maxmag)
#        for idx_v in range(v.size):
#            print(idx_v,
#                  uiv_IR[idx_v, l].real,
#                  uiv_IR[idx_v, l].imag,
#                  uiv_IR_ref[idx_v, l].real,
#                  uiv_IR_ref[idx_v, l].imag,
#                  )
        print()

