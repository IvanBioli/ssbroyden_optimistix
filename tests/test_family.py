"""The self-scaling Broyden family: consistency of the update formulas and convergence."""

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import optimistix as optx
import ssbroyden_optimistix as sso
from optimistix import FunctionInfo
from ssbroyden_optimistix._ssbroyden import _SSBroydenUpdateState

FAMILY = {
    "SSBroyden": sso.SSBroyden,
    "Broyden": sso.Broyden,
    "SSBFGS": sso.SSBFGS,
    "SSDFP": sso.SSDFP,
}


def _spd(n, seed, cond=1e2):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    return Q @ np.diag(np.logspace(0, np.log10(cond), n)) @ Q.T


def _data(n=6, seed=0, y_scale=1.0):
    """``H0 = B0^{-1}``, a previous gradient ``g`` and a step ``s`` along ``-H0 g``.

    A small ``y_scale`` makes ``b = s^T B s / y^T s`` large, so that the later-step
    scaling ``tau`` differs from 1 for every member of the family."""
    rng = np.random.default_rng(seed)
    B0 = _spd(n, seed)
    H0 = np.linalg.inv(B0)
    g = rng.normal(size=n)
    alpha = 0.7
    s = -alpha * H0 @ g
    # Gradient difference with y^T s > 0 (a curvature pair of some SPD matrix).
    y = y_scale * _spd(n, seed + 1, cond=10.0) @ s
    return B0, H0, g, s, y, alpha


def _update(solver, M, g, s, y, alpha, first_step):
    y0 = jnp.zeros_like(jnp.asarray(s))
    op = lx.PyTreeLinearOperator(
        jnp.asarray(M),
        jax.ShapeDtypeStruct(y0.shape, y0.dtype),
        lx.positive_semidefinite_tag,
    )
    grad_prev = jnp.asarray(g)
    if solver.use_inverse:
        f_info = FunctionInfo.EvalGradHessianInv(jnp.array(1.0), grad_prev, op)
    else:
        f_info = FunctionInfo.EvalGradHessian(jnp.array(1.0), grad_prev, op)
    f_eval_info = FunctionInfo.EvalGrad(jnp.array(0.5), grad_prev + jnp.asarray(y))
    state = _SSBroydenUpdateState(
        first_step=jnp.array(first_step), step_size=jnp.array(1.0)
    )
    new_info, _ = solver.update_hessian(
        y0, y0 + jnp.asarray(s), f_info, f_eval_info, state, jnp.array(alpha)
    )
    out = new_info.hessian_inv if solver.use_inverse else new_info.hessian
    return np.asarray(out.as_matrix())


@pytest.mark.parametrize("name", list(FAMILY))
@pytest.mark.parametrize("first_step", [True, False])
@pytest.mark.parametrize("y_scale", [1.0, 0.01])
def test_hessian_form_is_inverse_of_inverse_form(name, first_step, y_scale):
    """One update of B0 (B-form) and of H0 = B0^{-1} (H-form) stay inverse to each other."""
    B0, H0, g, s, y, alpha = _data(y_scale=y_scale)
    cls = FAMILY[name]
    H1 = _update(cls(0.0, 0.0, use_inverse=True), H0, g, s, y, alpha, first_step)
    B1 = _update(cls(0.0, 0.0, use_inverse=False), B0, g, s, y, alpha, first_step)
    np.testing.assert_allclose(B1 @ H1, np.eye(len(s)), atol=1e-10)


@pytest.mark.parametrize("first_step", [True, False])
def test_secant_condition(first_step):
    """``H+ y = s`` for every member (the defining property of quasi-Newton updates)."""
    B0, H0, g, s, y, alpha = _data(seed=3)
    for cls in FAMILY.values():
        H1 = _update(cls(0.0, 0.0), H0, g, s, y, alpha, first_step)
        np.testing.assert_allclose(H1 @ y, s, rtol=1e-10, atol=1e-12)


class _GeneralTheta(sso.AbstractSSBroyden):
    """The general update with a fixed ``theta`` and ``tau`` (for cross-checks)."""

    rtol: float = 0.0
    atol: float = 0.0
    norm: object = optx.max_norm
    use_inverse: bool = True
    descent: optx.NewtonDescent = optx.NewtonDescent(linear_solver=lx.Cholesky())
    search: object = sso.Zoom()
    verbose: frozenset = frozenset()
    alpha: str = "geometric"
    initial_scaling: bool = False
    theta: float = 0.0
    tau: float | None = None

    def _compute_thetak(self, ak, bk, hk):
        return jnp.array(self.theta)

    def _compute_tauk(self, thetak, ak, bk, hk, is_first, N):
        if self.tau is None:
            return super()._compute_tauk(thetak, ak, bk, hk, is_first, N)
        return jnp.array(self.tau)


@pytest.mark.parametrize("use_inverse", [True, False])
@pytest.mark.parametrize("name,theta", [("SSBFGS", 0.0), ("SSDFP", 1.0)])
def test_specialised_formulas_match_general_formula(name, theta, use_inverse):
    B0, H0, g, s, y, alpha = _data(seed=5)
    M = H0 if use_inverse else B0
    special = _update(
        FAMILY[name](0.0, 0.0, use_inverse=use_inverse), M, g, s, y, alpha, False
    )
    general = _update(
        _GeneralTheta(theta=theta, use_inverse=use_inverse), M, g, s, y, alpha, False
    )
    np.testing.assert_allclose(special, general, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("use_inverse", [True, False])
@pytest.mark.parametrize("theta,classic", [(0.0, sso.BFGS), (1.0, sso.DFP)])
def test_unscaled_family_reduces_to_bfgs_and_dfp(theta, classic, use_inverse):
    B0, H0, g, s, y, alpha = _data(seed=7)
    M = H0 if use_inverse else B0
    family = _update(
        _GeneralTheta(theta=theta, tau=1.0, use_inverse=use_inverse),
        M,
        g,
        s,
        y,
        alpha,
        False,
    )
    ref = _update(classic(0.0, 0.0, use_inverse=use_inverse), M, g, s, y, alpha, False)
    np.testing.assert_allclose(family, ref, rtol=1e-10, atol=1e-12)


def _rosenbrock(y, args):
    return (args - y[0]) ** 2 + 100 * (y[1] - y[0] ** 2) ** 2


def _matyas(y, args):
    c1, c2 = args
    return c1 * (y[0] ** 2 + y[1] ** 2) - c2 * y[0] * y[1]


def _quadratic(y, args):
    return y @ args @ y


_Q = jnp.asarray(_spd(4, 11, cond=1e2))
PROBLEMS = {
    "rosenbrock": (_rosenbrock, jnp.array([0.0, 0.0]), jnp.array(1.0)),
    "matyas": (_matyas, jnp.array([6.0, 6.0]), (jnp.array(0.26), jnp.array(0.48))),
    "quadratic": (_quadratic, jnp.ones(4), _Q),
}


@pytest.mark.parametrize("problem", list(PROBLEMS))
@pytest.mark.parametrize("name", list(FAMILY))
@pytest.mark.parametrize("use_inverse", [True, False])
def test_converges(problem, name, use_inverse):
    fn, y0, args = PROBLEMS[problem]
    solver = FAMILY[name](1e-8, 1e-8, use_inverse=use_inverse)
    sol = optx.minimise(fn, solver, y0, args=args, max_steps=500, throw=False)
    assert float(fn(sol.value, args)) < 1e-8


@pytest.mark.parametrize("problem", list(PROBLEMS))
def test_alpha_search_matches_geometric_under_keep(problem):
    """With ``Zoom(initial_guess_strategy="keep")`` the search reports the step taken,
    so both choices of ``alpha`` give the same iterates up to rounding (compared over
    the first steps, before rounding differences are amplified)."""
    fn, y0, args = PROBLEMS[problem]
    sols = [
        optx.minimise(
            fn,
            sso.SSBroyden(0.0, 0.0, alpha=a),
            y0,
            args=args,
            max_steps=10,
            throw=False,
        )
        for a in ("geometric", "search")
    ]
    np.testing.assert_allclose(sols[0].value, sols[1].value, rtol=1e-13, atol=1e-300)


def test_step_counters():
    fn, y0, args = PROBLEMS["rosenbrock"]
    sol = optx.minimise(fn, sso.SSBroyden(1e-8, 1e-8), y0, args=args, max_steps=500)
    state = sol.state
    assert int(state.num_steps) == int(sol.stats["num_steps"])
    assert 0 < int(state.num_accepted_steps) <= int(state.num_steps)


def test_contrib_run_counts_accepted_steps():
    from ssbroyden_optimistix.contrib import run

    fn, y0, args = PROBLEMS["rosenbrock"]
    seen = []
    res = run(
        sso.SSBroyden(1e-12, 1e-12),
        lambda y: fn(y, args),
        y0,
        maxiter=15,
        callback=lambda it, p, s, loss: seen.append(it) or False,
    )
    assert res.num_iters == 15 and seen == list(range(1, 16))
    assert res.num_steps >= res.num_iters
