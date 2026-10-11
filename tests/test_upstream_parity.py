"""With Optimistix' own line search, our BFGS/DFP/LBFGS are Optimistix' solvers.

Upstream BFGS, DFP and LBFGS use ``BacktrackingArmijo``; ours use ``Zoom`` by default
but take ``search=``. With the same search the iterates must agree bit for bit with
the installed (upstream) Optimistix. Skipped when the installed Optimistix is the fork.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import optimistix as optx
import ssbroyden_optimistix as sso

FORK = hasattr(optx, "SSBroyden")
pytestmark = pytest.mark.skipif(FORK, reason="installed optimistix is the fork")


def rosenbrock(y, args):
    return jnp.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2)


_Q = np.linalg.qr(np.random.default_rng(0).normal(size=(12, 12)))[0]
_A = jnp.asarray(_Q @ np.diag(np.logspace(0, 3, 12)) @ _Q.T)


def quadratic(y, args):
    return 0.5 * y @ _A @ y - jnp.sum(y)


PROBLEMS = {
    "rosenbrock": (rosenbrock, jnp.zeros(6)),
    "quadratic": (quadratic, jnp.ones(12)),
}
PAIRS = {
    "bfgs": (lambda s: sso.BFGS(1e-8, 1e-8, search=s), lambda: optx.BFGS(1e-8, 1e-8)),
    "bfgs_hessian": (
        lambda s: sso.BFGS(1e-8, 1e-8, use_inverse=False, search=s),
        lambda: optx.BFGS(1e-8, 1e-8, use_inverse=False),
    ),
    "dfp": (lambda s: sso.DFP(1e-8, 1e-8, search=s), lambda: optx.DFP(1e-8, 1e-8)),
    "lbfgs": (
        lambda s: sso.LBFGS(1e-8, 1e-8, history_length=5, search=s),
        lambda: optx.LBFGS(1e-8, 1e-8, history_length=5),
    ),
}


@pytest.mark.parametrize("problem", list(PROBLEMS))
@pytest.mark.parametrize("pair", list(PAIRS))
def test_same_as_upstream_with_armijo(problem, pair):
    fn, y0 = PROBLEMS[problem]
    ours, theirs = PAIRS[pair]
    kw = dict(max_steps=200, throw=False)
    a = optx.minimise(fn, ours(optx.BacktrackingArmijo()), y0, **kw)
    b = optx.minimise(fn, theirs(), y0, **kw)
    assert int(a.stats["num_steps"]) == int(b.stats["num_steps"])
    assert np.array_equal(np.asarray(a.value), np.asarray(b.value))
