# SPDX-License-Identifier: Apache-2.0
#
# The self-scaling Broyden family. The SSBroyden update follows the implementation in
# the fork https://github.com/IvanBioli/optimistix (tag cmame-2026); the family
# structure and the Hessian (B-) forms of the updates are by Mikel Mendibe and Ivan
# Bioli (ssbroyden_jax).

import abc
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar
from optimistix import AbstractSearch, FunctionInfo, NewtonDescent

from ._custom_types import Aux, Y
from ._misc import filter_cond, max_norm, tree_dot
from ._quasi_newton import _Hessian, _identity_pytree, _outer, AbstractQuasiNewton
from ._zoom import Zoom


class _SSBroydenUpdateState(eqx.Module):
    """State of the self-scaling Broyden family update."""

    first_step: Bool[Array, ""]
    step_size: Scalar


class AbstractSSBroydenFamily(AbstractQuasiNewton[Y, Aux, _Hessian, _SSBroydenUpdateState]):
    """The self-scaling Broyden family of quasi-Newton updates.

    With ``s = y_{k+1} - y_k``, ``y = g_{k+1} - g_k`` (the gradient difference),
    ``rho = 1 / (y^T s)`` and the scalars ``h = y^T H y rho``, ``b = s^T B s rho`` and
    ``a = b h - 1``, the inverse-Hessian (H-) form of the update is

        H+ = (H - H y y^T H / (y^T H y) + phi (y^T H y) v v^T) / tau + rho s s^T,
        v = rho s - H y / (y^T H y),  phi = (1 - theta) / (1 + a theta),

    and the Hessian (B-) form, its inverse, is

        B+ = (B - B s s^T B / (s^T B s) + theta (s^T B s) w w^T) tau + rho y y^T,
        w = rho y - B s / (s^T B s).

    ``theta`` interpolates between BFGS (0) and DFP (1); ``tau`` scales the
    approximation. Subclasses choose them (``_compute_thetak``, ``_compute_tauk``)
    and may specialise the update formulas (``_invhessian_update_term``,
    ``_hessian_update_term``).

    In the H-form, ``b = -alpha rho s^T g_k`` with the step length ``alpha``:
    ``alpha="geometric"`` takes ``alpha = <s, p> / <p, p>`` for the search direction
    ``p = -H g_k`` (exact for any line search along ``p``); ``alpha="search"`` takes the
    step size reported by the line search, which is the length of the accepted step
    only for searches that report it (e.g. ``Zoom(initial_guess_strategy="keep")``).
    The B-form computes ``b`` from ``s^T B s`` directly and ignores ``alpha``.

    With ``initial_scaling=True`` the first update uses the initial scaling
    ``tau = h / (1 + a theta)``; with ``False`` it is skipped, as in the paper's
    experiments.
    """

    alpha: AbstractVar[str]
    initial_scaling: AbstractVar[bool]

    def init_hessian(self, y: Y, f: Scalar, grad: Y) -> tuple[_Hessian, _SSBroydenUpdateState]:
        identity_operator = _identity_pytree(y)
        if self.use_inverse:
            f_info = FunctionInfo.EvalGradHessianInv(f, grad, identity_operator)
        else:
            f_info = FunctionInfo.EvalGradHessian(f, grad, identity_operator)
        return f_info, _SSBroydenUpdateState(
            first_step=jnp.array(True),
            step_size=jnp.array(1.0),
        )  # pyright: ignore

    def _compute_thetak(self, ak: Scalar, bk: Scalar, hk: Scalar) -> Scalar:
        """The SSBroyden choice of ``theta`` (in the interval that keeps ``H+`` SPD)."""
        ck = jnp.sqrt(jnp.abs(ak / (1 + ak)))
        rhokm = jnp.minimum(1.0, hk * (1 - ck))
        thetakm = (rhokm - 1) / ak
        thetakp = 1 / rhokm
        return jnp.maximum(thetakm, jnp.minimum(thetakp, (1 - bk) / bk))

    def _compute_tauk(
        self,
        thetak: Scalar,
        ak: Scalar,
        bk: Scalar,
        hk: Scalar,
        is_first: Bool[Array, ""],
        N: int,
    ) -> Scalar:
        """The SSBroyden choice of ``tau``; ``N`` is the number of unknowns."""

        def first_iter_tauk(_):
            return hk / (1 + ak * thetak)

        def later_iter_tauk(_):
            rhokk = jnp.minimum(1.0, 1.0 / bk)
            sigmak = 1 + thetak * ak
            sigmaknm1 = jnp.abs(sigmak) ** (1.0 / (1.0 - N))
            return jax.lax.cond(
                thetak <= 0,
                lambda _: jnp.minimum(rhokk * sigmaknm1, sigmak),
                lambda _: rhokk * jnp.minimum(sigmaknm1, 1 / thetak),
                operand=None,
            )

        return jax.lax.cond(is_first, first_iter_tauk, later_iter_tauk, operand=None)

    def _invhessian_update_term(
        self, hessian_pytree, y_diff, Hy, inner, yHy, rho, thetak, tauk, ak
    ) -> PyTree:
        """General H-form update."""
        # v_k = s_k * rho - H_k y_k / (y_k^T H_k y_k)
        vk = (y_diff**ω * rho - Hy**ω / yHy).ω
        # phi_k = (1 - theta_k) / (1 + a_k * theta_k)
        phik = (1 - thetak) / (1 + ak * thetak)
        term1 = _outer(Hy, Hy)
        term2 = _outer(vk, vk)
        term3 = _outer(y_diff, y_diff)
        return (
            (hessian_pytree**ω - term1**ω / yHy + term2**ω * (phik * yHy)) / tauk
            + term3**ω * rho
        ).ω

    def _hessian_update_term(self, hessian_pytree, grad_diff, Bs, sBs, rho, thetak, tauk) -> PyTree:
        """General B-form update."""
        # w_k = y_k / (y_k^T s_k) - B_k s_k / (s_k^T B_k s_k)
        wk = (grad_diff**ω * rho - Bs**ω / sBs).ω
        term1 = _outer(Bs, Bs)
        term2 = _outer(wk, wk)
        term3 = _outer(grad_diff, grad_diff)
        return (
            (hessian_pytree**ω - term1**ω / sBs + thetak * sBs * term2**ω) * tauk
            + term3**ω * rho
        ).ω

    def update_hessian(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: _SSBroydenUpdateState,
        step_size: Scalar,
    ) -> tuple[_Hessian, _SSBroydenUpdateState]:
        f_eval = f_eval_info.f
        grad = f_eval_info.grad
        y_diff = (y_eval**ω - y**ω).ω
        grad_diff = (grad**ω - f_info.grad**ω).ω
        inner = tree_dot(grad_diff, y_diff)

        # inner = 0 on the first step (the initial gradient is 0) and inner <= eps on
        # convergence: no update then.
        inner_nonzero = inner > jnp.finfo(inner.dtype).eps
        N = sum(jnp.size(leaf) for leaf in jtu.tree_leaves(y_diff))

        def no_update(args):
            *_, f_info, hessian_update_state, _step_size = args
            if self.use_inverse:
                return f_info.hessian_inv, hessian_update_state
            else:
                return f_info.hessian, hessian_update_state

        def update(args):
            inner, grad_diff, y_diff, f_info, hessian_update_state, step_sz = args
            is_first = hessian_update_state.first_step
            if self.use_inverse:
                assert isinstance(f_info, FunctionInfo.EvalGradHessianInv)
                operator = f_info.hessian_inv
                rho = 1.0 / inner  # rho = 1 / (y_k^T s_k)
                Hy = operator.mv(grad_diff)  # H_k y_k
                yHy = tree_dot(grad_diff, Hy)  # y_k^T H_k y_k
                hk = yHy * rho
                # bk = -alpha_k * rho * (s_k^T g_k)
                grad_prev = f_info.grad
                if self.alpha == "search":
                    alpha_k = step_sz
                else:
                    p = (-(operator.mv(grad_prev) ** ω)).ω
                    alpha_k = tree_dot(y_diff, p) / tree_dot(p, p)
                bk = -alpha_k * rho * tree_dot(y_diff, grad_prev)
                ak = bk * hk - 1
                thetak = self._compute_thetak(ak, bk, hk)
                tauk = self._compute_tauk(thetak, ak, bk, hk, is_first, N)
                new_pytree = self._invhessian_update_term(
                    operator.pytree, y_diff, Hy, inner, yHy, rho, thetak, tauk, ak
                )
            else:
                assert isinstance(f_info, FunctionInfo.EvalGradHessian)
                operator = f_info.hessian
                rho = 1.0 / inner
                Bs = operator.mv(y_diff)  # B_k s_k
                sBs = tree_dot(y_diff, Bs)  # s_k^T B_k s_k
                bk = sBs * rho
                # h needs H_k y_k = B_k^{-1} y_k.
                Hy = lx.linear_solve(operator, grad_diff, solver=lx.Cholesky()).value
                yHy = tree_dot(grad_diff, Hy)
                hk = yHy * rho
                ak = bk * hk - 1
                thetak = self._compute_thetak(ak, bk, hk)
                tauk = self._compute_tauk(thetak, ak, bk, hk, is_first, N)
                new_pytree = self._hessian_update_term(
                    operator.pytree, grad_diff, Bs, sBs, rho, thetak, tauk
                )

            # Keep the old approximation if the update is not finite. (theta is 0/0 when
            # a = 0 near convergence; with tau = 1 only the theta check catches it.)
            is_finite = jnp.isfinite(rho) & jnp.isfinite(1 / tauk) & jnp.isfinite(thetak)
            new_pytree = jtu.tree_map(
                lambda new, old: jnp.where(is_finite, new, old),
                new_pytree,
                operator.pytree,
            )
            new_operator = lx.PyTreeLinearOperator(
                new_pytree,  # pyright: ignore
                output_structure=jax.eval_shape(lambda: grad_diff),
                tags=lx.positive_semidefinite_tag,
            )
            return new_operator, _SSBroydenUpdateState(
                first_step=jnp.array(False), step_size=step_sz
            )

        args = (inner, grad_diff, y_diff, f_info, hessian_update_state, step_size)
        hessian, new_update_state = filter_cond(inner_nonzero, update, no_update, args)

        if self.initial_scaling:
            # `first_step` is cleared once an update has been applied, so the first
            # update uses the initial scaling.
            first_step = new_update_state.first_step
        else:
            # The first call (always without update: the initial gradient difference
            # is zero) clears `first_step`, so the initial scaling is never used.
            first_step = jnp.array(False)
        new_update_state = _SSBroydenUpdateState(first_step=first_step, step_size=step_size)

        if self.use_inverse:
            return (
                FunctionInfo.EvalGradHessianInv(f_eval, grad, hessian),
                new_update_state,
            )  # pyright: ignore
        else:
            return (
                FunctionInfo.EvalGradHessian(f_eval, grad, hessian),
                new_update_state,
            )  # pyright: ignore


class AbstractSSBroyden(AbstractSSBroydenFamily[Y, Aux, _Hessian]):
    """Self-scaling Broyden: ``theta`` and ``tau`` chosen at every step."""


class AbstractBroyden(AbstractSSBroydenFamily[Y, Aux, _Hessian]):
    """Broyden family with the SSBroyden ``theta`` and no scaling (``tau = 1``)."""

    def _compute_tauk(self, thetak, ak, bk, hk, is_first, N) -> Scalar:
        return jnp.array(1.0)


class AbstractSSBFGS(AbstractSSBroydenFamily[Y, Aux, _Hessian]):
    """Self-scaling BFGS: ``theta = 0``, ``tau`` chosen at every step."""

    def _compute_thetak(self, ak, bk, hk) -> Scalar:
        return jnp.array(0.0)

    def _invhessian_update_term(
        self, hessian_pytree, y_diff, Hy, inner, yHy, rho, thetak, tauk, ak
    ) -> PyTree:
        # H+ = (H - (H y s^T + s y^T H) / inner) / tau + (inner + yHy / tau) s s^T / inner^2
        diff_outer = _outer(y_diff, y_diff)
        mvp_outer = _outer(y_diff, Hy)
        term1 = (((inner + yHy / tauk) * (diff_outer**ω)) / (inner**2)).ω
        term2 = ((_outer(Hy, y_diff) ** ω + mvp_outer**ω) / inner).ω
        return ((hessian_pytree**ω - term2**ω) / tauk + term1**ω).ω

    def _hessian_update_term(self, hessian_pytree, grad_diff, Bs, sBs, rho, thetak, tauk) -> PyTree:
        # B+ = (B - B s s^T B / (s^T B s)) tau + rho y y^T
        term1 = _outer(Bs, Bs)
        term2 = _outer(grad_diff, grad_diff)
        return ((hessian_pytree**ω - term1**ω / sBs) * tauk + term2**ω * rho).ω


class AbstractSSDFP(AbstractSSBroydenFamily[Y, Aux, _Hessian]):
    """Self-scaling DFP: ``theta = 1``, ``tau`` chosen at every step."""

    def _compute_thetak(self, ak, bk, hk) -> Scalar:
        return jnp.array(1.0)

    def _invhessian_update_term(
        self, hessian_pytree, y_diff, Hy, inner, yHy, rho, thetak, tauk, ak
    ) -> PyTree:
        # H+ = (H - H y y^T H / yHy) / tau + s s^T / inner
        term1 = (_outer(Hy, Hy) ** ω / yHy).ω
        term2 = (_outer(y_diff, y_diff) ** ω / inner).ω
        return ((hessian_pytree**ω - term1**ω) / tauk + term2**ω).ω

    def _hessian_update_term(self, hessian_pytree, grad_diff, Bs, sBs, rho, thetak, tauk) -> PyTree:
        # B+ = (B - (B s y^T + y s^T B) / inner) tau + (inner + tau sBs) y y^T / inner^2
        inner = 1.0 / rho
        diff_outer = _outer(grad_diff, grad_diff)
        mvp_outer = _outer(grad_diff, Bs)
        term1 = (((inner + sBs * tauk) * (diff_outer**ω)) / (inner**2)).ω
        term2 = ((_outer(Bs, grad_diff) ** ω + mvp_outer**ω) / inner).ω
        return ((hessian_pytree**ω - term2**ω) * tauk + term1**ω).ω


def _init(self, rtol, atol, norm, use_inverse, verbose, search, alpha, initial_scaling):
    if alpha not in ("geometric", "search"):
        raise ValueError(f"alpha must be 'geometric' or 'search', got {alpha!r}")
    self.rtol = rtol
    self.atol = atol
    self.norm = norm
    self.use_inverse = use_inverse
    self.descent = NewtonDescent(linear_solver=lx.Cholesky())
    self.search = Zoom() if search is None else search
    self.verbose = verbose
    self.alpha = alpha
    self.initial_scaling = initial_scaling


_INIT_DOC = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used in the convergence criteria (default `optimistix.max_norm`).
- `use_inverse`: approximate the inverse Hessian (`True`, default) or the Hessian.
- `verbose`: frozenset of `"step_size"`, `"loss"`, `"y"`: what to print at each step.
- `search`: the line search; `None` (default) means `Zoom()`.
- `alpha`: step length in the H-form update, `"geometric"` (default) or `"search"`;
    see `AbstractSSBroydenFamily`.
- `initial_scaling`: whether the first update uses the initial scaling (default
    `True`).
"""


class SSBroyden(AbstractSSBroyden[Y, Aux, _Hessian]):
    """Self-scaling Broyden (SSBroyden) minimisation algorithm.

    A quasi-Newton method of the Broyden family that chooses the family parameter
    ``theta`` and the scaling ``tau`` at every step; see `AbstractSSBroydenFamily`.

    Supports the `options` `autodiff_mode` (`"bwd"`, default, or `"fwd"`): how the
    gradient is computed.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: AbstractSearch
    verbose: frozenset[str]
    alpha: str
    initial_scaling: bool

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: frozenset[str] = frozenset(),
        search: AbstractSearch | None = None,
        alpha: str = "geometric",
        initial_scaling: bool = True,
    ):
        _init(self, rtol, atol, norm, use_inverse, verbose, search, alpha, initial_scaling)


class Broyden(AbstractBroyden[Y, Aux, _Hessian]):
    """Broyden-family minimisation algorithm: the SSBroyden ``theta``, no scaling.

    See `SSBroyden` for the arguments and options.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: AbstractSearch
    verbose: frozenset[str]
    alpha: str
    initial_scaling: bool

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: frozenset[str] = frozenset(),
        search: AbstractSearch | None = None,
        alpha: str = "geometric",
        initial_scaling: bool = True,
    ):
        _init(self, rtol, atol, norm, use_inverse, verbose, search, alpha, initial_scaling)


class SSBFGS(AbstractSSBFGS[Y, Aux, _Hessian]):
    """Self-scaling BFGS minimisation algorithm (``theta = 0``, ``tau`` per step).

    See `SSBroyden` for the arguments and options.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: AbstractSearch
    verbose: frozenset[str]
    alpha: str
    initial_scaling: bool

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: frozenset[str] = frozenset(),
        search: AbstractSearch | None = None,
        alpha: str = "geometric",
        initial_scaling: bool = True,
    ):
        _init(self, rtol, atol, norm, use_inverse, verbose, search, alpha, initial_scaling)


class SSDFP(AbstractSSDFP[Y, Aux, _Hessian]):
    """Self-scaling DFP minimisation algorithm (``theta = 1``, ``tau`` per step).

    See `SSBroyden` for the arguments and options.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: AbstractSearch
    verbose: frozenset[str]
    alpha: str
    initial_scaling: bool

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: frozenset[str] = frozenset(),
        search: AbstractSearch | None = None,
        alpha: str = "geometric",
        initial_scaling: bool = True,
    ):
        _init(self, rtol, atol, norm, use_inverse, verbose, search, alpha, initial_scaling)


for _cls in (SSBroyden, Broyden, SSBFGS, SSDFP):
    _cls.__init__.__doc__ = _INIT_DOC
