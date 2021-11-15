# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np

import irbasis3
from irbasis3 import util
import pytest


@pytest.mark.parametrize("statistics", ["F", "B"])
def test_high_freq_moments(statistics):
    """
    G(iv) = G_1/iv + G_2/(iv)^2

    G_1 = [[1, 0], [0, 1]]
    G_2 = [[0, 1], [2, 0]]
    """
    lambda_ = 1e+3
    beta = 1e+2
    eps = 1e-7
    basis = irbasis3.FiniteTempBasis(
        irbasis3.KernelFFlat(lambda_),
        'F', beta, eps=eps
    )
    nf = 2

    smpl = irbasis3.MatsubaraSampling(basis)
    high_freq_mom = [np.identity(2), np.array([[0,1],[2,0]])]
    num_moments = len(high_freq_mom)

    giv = np.zeros((smpl.sampling_points.size, nf, nf), dtype=np.complex128)
    for m in range(num_moments):
        giv += high_freq_mom[m][None,:,:]/(1J*(np.pi/beta)*smpl.sampling_points[:,None,None])**(m+1)
    gl = smpl.fit(giv, axis=0)
    high_freq_mom_reconst = util.high_freq_moment(gl, basis, num_moments=2, axis=0)

    high_freq_mom_reconst2 = []
    for m in range(num_moments):
        ev = util.evalulator_high_freq_moment(basis, m+1)
        high_freq_mom_reconst2.append(np.einsum('l,lij->ij', ev, gl))

    for m in range(num_moments):
        assert (np.abs(high_freq_mom_reconst[m] - high_freq_mom[m]).max()) < (100**m) * 100 * eps
        assert (np.abs(high_freq_mom_reconst2[m] - high_freq_mom[m]).max()) < (100**m) * 100 * eps