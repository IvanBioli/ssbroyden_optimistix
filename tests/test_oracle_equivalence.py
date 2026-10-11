"""Bitwise equivalence with the optimistix fork used for the CMAME 2026 experiments.

The reference ("oracle") is the fork at tag cmame-2026 (commit 83f872e). If the installed
``optimistix`` is that fork it is used directly; otherwise set ``OPTX_ORACLE_PATH`` to the
fork's inner ``optimistix/`` package directory and it is loaded as ``optimistix_oracle``.
"""

import importlib.util
import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import optimistix
import ssbroyden_optimistix as sso


def _load_oracle():
    if hasattr(optimistix, "SSBroyden") and hasattr(optimistix, "Zoom"):
        return optimistix
    path = os.environ.get("OPTX_ORACLE_PATH")
    if not path or not os.path.isfile(os.path.join(path, "__init__.py")):
        return None
    if "optimistix_oracle" in sys.modules:
        return sys.modules["optimistix_oracle"]
    spec = importlib.util.spec_from_file_location(
        "optimistix_oracle",
        os.path.join(path, "__init__.py"),
        submodule_search_locations=[path],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["optimistix_oracle"] = mod
    spec.loader.exec_module(mod)
    return mod


ORACLE = _load_oracle()
pytestmark = pytest.mark.skipif(ORACLE is None, reason="fork oracle not available")
ORACLE_COMMIT = "83f872e"  # tag cmame-2026


def test_oracle_is_the_fork_of_the_paper():
    """The oracle checkout must be the fork at the commit of the paper's runs."""
    import subprocess

    path = os.path.dirname(os.path.abspath(ORACLE.__file__))
    res = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if res.returncode != 0:
        pytest.skip("oracle is not a git checkout; cannot check its commit")
    assert res.stdout.strip().startswith(ORACLE_COMMIT), res.stdout


# --- problems -------------------------------------------------------------------------


def rosenbrock(y, args):
    return jnp.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2), None


_Q = np.linalg.qr(np.random.default_rng(0).normal(size=(40, 40)))[0]
_A = jnp.asarray(_Q @ np.diag(np.logspace(0, 4, 40)) @ _Q.T)
_b = jnp.asarray(np.random.default_rng(1).normal(size=40))


def quadratic(y, args):
    return 0.5 * y @ _A @ y - _b @ y, None


def _mlp_params(key):
    sizes = [2, 8, 8, 1]
    params = {}
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k1, k2 = jax.random.split(key, 3)
        params[f"W{i}"] = jax.random.normal(k1, (n, m)) / jnp.sqrt(m)
        params[f"b{i}"] = 0.1 * jax.random.normal(k2, (n,))
    return params


def _mlp(params, x):
    h = x
    for i in range(2):
        h = jnp.tanh(params[f"W{i}"] @ h + params[f"b{i}"])
    return (params["W2"] @ h + params["b2"])[0]


_X = jax.random.uniform(jax.random.PRNGKey(3), (32, 2))


def pinn_lite(params, args):
    lap = lambda x: jnp.trace(jax.hessian(lambda z: _mlp(params, z))(x))
    f = lambda x: 2 * jnp.pi**2 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
    r = jax.vmap(lambda x: lap(x) + f(x))(_X)
    return 0.5 * jnp.mean(r**2), None


PROBLEMS = {
    "rosenbrock": (rosenbrock, jnp.array([-1.2, 1.0, 0.5, -0.3])),
    "quadratic": (quadratic, jnp.zeros(40)),
    "pinn_lite": (pinn_lite, _mlp_params(jax.random.PRNGKey(0))),
}


# --- solver pairs: ours (as-run arguments) vs the fork's call in the experiments ------


def _pairs():
    O = ORACLE
    return {
        "bfgs": (
            sso.BFGS(rtol=0.0, atol=0.0),
            O.BFGS(rtol=0.0, atol=0.0, search=O.Zoom()),
        ),
        "bfgs_increase": (
            sso.BFGS(
                rtol=0.0, atol=0.0, search=sso.Zoom(initial_guess_strategy="increase")
            ),
            O.BFGS(
                rtol=0.0, atol=0.0, search=O.Zoom(initial_guess_strategy="increase")
            ),
        ),
        "lbfgs": (
            sso.LBFGS(rtol=0.0, atol=0.0, history_length=500),
            O.LBFGS(rtol=0.0, atol=0.0, search=O.Zoom(), history_length=500),
        ),
        "ssbroyden": (
            sso.SSBroyden(rtol=0.0, atol=0.0, alpha="search", initial_scaling=False),
            O.SSBroyden(rtol=0.0, atol=0.0, search=O.Zoom()),
        ),
    }


def _comparable(state):
    """The state without the counter that the fork does not have."""
    if hasattr(state, "num_steps"):
        state = eqx.tree_at(lambda s: s.num_steps, state, None)
    return state


def _drive(solver, fn, y0, steps):
    f_struct = jax.ShapeDtypeStruct((), jnp.float64)
    step = eqx.filter_jit(eqx.Partial(solver.step, fn=fn, options={}, tags=frozenset()))
    state = solver.init(fn, y0, None, {}, f_struct, None, frozenset())
    y, out = y0, []
    for _ in range(steps):
        y, state, _ = step(y=y, state=state, args=None)
        leaves = [
            np.asarray(l)
            for l in jax.tree_util.tree_leaves(_comparable(state))
            if eqx.is_array(l)
        ]
        out.append(
            (
                b"".join(np.asarray(l).tobytes() for l in jax.tree_util.tree_leaves(y)),
                [(l.dtype.str, l.shape, l.tobytes()) for l in leaves],
            )
        )
    return out


@pytest.mark.parametrize("problem", list(PROBLEMS))
@pytest.mark.parametrize("pair", ["bfgs", "bfgs_increase", "lbfgs", "ssbroyden"])
def test_bitwise_equal_to_fork(problem, pair):
    fn, y0 = PROBLEMS[problem]
    ours, ref = _pairs()[pair]
    a = _drive(ours, fn, y0, 120)
    b = _drive(ref, fn, y0, 120)
    for k, (sa, sb) in enumerate(zip(a, b)):
        assert sa[0] == sb[0], f"step {k}: iterate differs"
        assert len(sa[1]) == len(sb[1]), f"step {k}: state structure differs"
        for i, (la, lb) in enumerate(zip(sa[1], sb[1])):
            assert la == lb, f"step {k}: state leaf {i} differs"
