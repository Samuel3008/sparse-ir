import numpy as np


def _contrained_lstsq_mat(a, c):
    """ Compute coefficient matrix for least-squares fitting with linear equality contraint

    return [ a^\dagger a   c^\dagger ]
           [     c             0     ]
    """
    num_const = c.shape[0]
    return np.block([
        [a.T.conjugate()@a,  c.T.conjugate()],
        [c                ,  np.zeros((num_const, num_const))]
    ])


def constrained_lstsq(a, b, c, d):
    """Linear equality constrained lest-squares fitting

    Comptue argmin_{x} (1/2) |a @ x - b|^2, 
    subject to the linear equality contrain c @ x = d.

    Mathematically, the solution is given by
      [ a^\dagger a   c^\dagger ] [x] = [a^dagger b]
      [     c             0     ] [v]   [ d ].

    Attributes:
    -----------
     - `a` : (M, N) array_like
    """
    assert a.ndim == 2
    assert b.ndim == 1
    assert c.ndim == 2
    assert d.ndim == 1

    M, N = a.shape
    num_const = d.shape[0]
    assert b.shape == (M,)
    assert c.shape == (num_const, N)

    a_ = _contrained_lstsq_mat(a, c)
    b_ = np.hstack([a.T.conjugate()@b, d])
    return np.linalg.solve(a_, b_)[:N]
