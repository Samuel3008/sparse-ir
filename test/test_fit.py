# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np

from irbasis3 import _fit
import pytest

def _randn(*shape, dtype=np.float64):
    if dtype == np.float64:
        return np.random.randn(*shape)
    elif dtype == np.complex128:
        return np.random.randn(*shape) + 1J*np.random.randn(*shape)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_constrained_lstsq(dtype):
    np.random.seed(100)
    M, N = 2, 4
    num_const = N - M
    a = _randn(M, N, dtype=dtype)
    b = _randn(M, dtype=dtype)
    c = _randn(num_const, N, dtype=dtype)
    d = _randn(num_const, dtype=dtype)

    x = _fit.constrained_lstsq(a, b, c, d)

    assert np.abs(a@x - b).max() < 1e-13
    assert np.abs(c@x - d).max() < 1e-13