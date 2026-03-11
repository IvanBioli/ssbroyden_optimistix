import abc
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from optimistix._custom_types import (
    Aux,
    DescentState,
    Fn,
    HessianUpdateState,
    SearchState,
    Y,
)
from optimistix._minimise import AbstractMinimiser
from optimistix._misc import (
    cauchy_termination,
    default_verbose,
    filter_cond,
    lin_to_grad,
    max_norm,
    tree_dot,
    tree_full_like,
    tree_where,
)
from optimistix._search import (
    AbstractDescent,
    AbstractSearch,
    FunctionInfo,
)
from optimistix._solution import RESULTS
from optimistix._solver.backtracking import BacktrackingArmijo
from optimistix._solver.gauss_newton import NewtonDescent
from optimistix._solver.zoom import Zoom

_Hessian = TypeVar(
    "_Hessian", FunctionInfo.EvalGradHessian, FunctionInfo.EvalGradHessianInv
)


def _identity_pytree(pytree: PyTree[Array]) -> lx.PyTreeLinearOperator:
    """Create an identity pytree `I` such that
    `pytree = lx.PyTreeLinearOperator(I).mv(pytree)`

    **Arguments**:

    - `pytree`: A pytree such that the output of `_identity_pytree` is the identity
        with respect to pytrees of the same shape as `pytree`.

    **Returns**:

    A `lx.PyTreeLinearOperator` with input and output shape the shape of `pytree`.
    """
    leaves, structure = jtu.tree_flatten(pytree)
    eye_structure = structure.compose(structure)
    eye_leaves = []
    for i1, l1 in enumerate(leaves):
        for i2, l2 in enumerate(leaves):
            dtype = jnp.result_type(l1, l2)
            if i1 == i2:
                eye_leaves.append(
                    jnp.eye(jnp.size(l1), dtype=dtype).reshape(
                        jnp.shape(l1) + jnp.shape(l2)
                    )
                )
            else:
                eye_leaves.append(jnp.zeros(jnp.shape(l1) + jnp.shape(l2), dtype=dtype))

    # This has a Lineax positive_semidefinite tag. This is okay because the BFGS update
    # preserves positive-definiteness.
    return lx.PyTreeLinearOperator(
        jtu.tree_unflatten(eye_structure, eye_leaves),
        jax.eval_shape(lambda: pytree),
        lx.positive_semidefinite_tag,
    )


def _outer(tree1, tree2):
    def leaf_fn(x):
        return jtu.tree_map(lambda leaf: jnp.tensordot(x, leaf, axes=0), tree2)

    return jtu.tree_map(leaf_fn, tree1)


class _QuasiNewtonState(
    eqx.Module,
    Generic[Y, Aux, SearchState, DescentState, _Hessian, HessianUpdateState],
):
    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Y
    search_state: SearchState
    # Updated after each descent step
    f_info: _Hessian
    aux: Aux
    descent_state: DescentState
    # Used for termination
    terminate: Bool[Array, ""]
    result: RESULTS
    # Used in compat.py
    num_accepted_steps: Int[Array, ""]
    # update state
    hessian_update_state: HessianUpdateState


class AbstractQuasiNewton(
    AbstractMinimiser[Y, Aux, _QuasiNewtonState],
    Generic[Y, Aux, _Hessian, HessianUpdateState],
):
    """Abstract quasi-Newton minimisation algorithm.

    Base class for quasi-Newton solvers, which create approximations to the Hessian or
    the inverse Hessian by accumulating gradient information over multiple iterations.
    Optimistix currently includes the following three variants:
    [`optimistix.BFGS`][], [`optimistix.DFP`][] and [`optimistix.LBFGS`][], each of
    which may be used to either approximate the Hessian or its inverse.
    The concrete classes may be subclassed to choose alternative descents and searches.

    Alternative flavors of quasi-Newton approximations may be implemented by subclassing
    `AbstractQuasiNewton` and providing implementations for the abstract methods
    `init_hessian` and `update_hessian`. The former is called to initialize the Hessian
    structure and the Hessian update state, while the latter is called to compute an
    update to the approximation of the Hessian or the inverse Hessian.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    use_inverse: AbstractVar[bool]
    descent: AbstractVar[AbstractDescent[Y, _Hessian, Any]]
    search: AbstractVar[AbstractSearch[Y, _Hessian, FunctionInfo.Eval, Any]]
    verbose: AbstractVar[Callable[..., None]]

    @abc.abstractmethod
    def init_hessian(
        self, y: Y, f: Scalar, grad: Y
    ) -> tuple[_Hessian, HessianUpdateState]:
        """Initialize the Hessian structure and Hessian update state.

        Set up a template structure of the Hessian to be used (with dummy values), as
        well as the state of the update method, which can be used to store past
        gradients for limited-memory Hessian approximations.
        """

    @abc.abstractmethod
    def update_hessian(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: HessianUpdateState,
        step_size: Scalar,
    ) -> tuple[_Hessian, HessianUpdateState]:
        """Update the Hessian approximation.

        This is called in the `step` method to update the Hessian approximation based on
        the current and previous iterates, their gradients, and the previous Hessian,
        whenever a step has been accepted and we query the descent for a new direction.

        Implementations should provide an update for the Hessian approximation or its
        inverse, and toggle updates as appropriate to maintain positive-definiteness
        of the operator.
        """

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _QuasiNewtonState:
        f = tree_full_like(f_struct, 0)
        grad = tree_full_like(y, 0)
        f_info, hessian_update_state = self.init_hessian(y, f, grad)
        f_info_struct = eqx.filter_eval_shape(lambda: f_info)

        return _QuasiNewtonState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(y, f_info_struct),
            terminate=jnp.array(False),
            result=RESULTS.successful,
            num_accepted_steps=jnp.array(0),
            hessian_update_state=hessian_update_state,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _QuasiNewtonState,
        tags: frozenset[object],
    ) -> tuple[Y, _QuasiNewtonState, Aux]:
        autodiff_mode = options.get("autodiff_mode", "bwd")
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )

        if self.search._needs_grad_at_y_eval:
            grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode, f_eval.dtype)
            f_eval_info = FunctionInfo.EvalGrad(f_eval, grad)
        else:
            f_eval_info = FunctionInfo.Eval(f_eval)

        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            f_eval_info,  # pyright: ignore  # TODO Fix (jhaffner)
            state.search_state,
        )

        def accepted(descent_state):
            nonlocal f_eval_info

            if not self.search._needs_grad_at_y_eval:
                grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode, f_eval.dtype)
                f_eval_info = FunctionInfo.EvalGrad(f_eval, grad)

            f_eval_info, hessian_update_state = self.update_hessian(
                y,
                state.y_eval,
                state.f_info,
                cast(FunctionInfo.EvalGrad, f_eval_info),
                state.hessian_update_state,
                step_size,
            )

            descent_state = self.descent.query(
                state.y_eval,
                f_eval_info,
                descent_state,
            )
            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step
            return (
                state.y_eval,
                f_eval_info,
                aux_eval,
                descent_state,
                terminate,
                hessian_update_state,
            )

        def rejected(descent_state):
            return (
                y,
                state.f_info,
                state.aux,
                descent_state,
                jnp.array(False),
                state.hessian_update_state,
            )

        y, f_info, aux, descent_state, terminate, hessian_update_state = filter_cond(
            accept, accepted, rejected, state.descent_state
        )

        self.verbose(
            loss_this_step=("Loss on this step", f_eval),
            loss_last_accepted_step=("Loss on the last accepted step", state.f_info.f),
            step_size=("Step size", step_size),
            y=("y", state.y_eval),
            y_last_accepted_step=("y on the last accepted step", y),
        )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        y_eval = (y**ω + y_descent**ω).ω
        result = RESULTS.where(
            search_result == RESULTS.successful, descent_result, search_result
        )

        prev_aux = tree_where(state.first_step, aux, state.aux)
        state = _QuasiNewtonState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
            num_accepted_steps=state.num_accepted_steps + jnp.where(accept, 1, 0),
            hessian_update_state=hessian_update_state,
        )
        return y, state, prev_aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _QuasiNewtonState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _QuasiNewtonState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


class _SSBroydenUpdateState(eqx.Module):
    """State for the self-scaling Broyden family update."""

    first_step: Bool[Array, ""]
    step_size: Scalar


class AbstractSSBroydenFamily(
    AbstractQuasiNewton[Y, Aux, _Hessian, _SSBroydenUpdateState]
):
    """Abstract base class for the self-scaling Broyden family of quasi-Newton methods.

    This class implements the general self-scaling Broyden update formula, which
    encompasses BFGS, DFP, and their self-scaling variants as special cases.

    The update formula depends on two key parameters:
    - `thetak`: Controls the convex combination between BFGS (thetak=0) and DFP (thetak=1)
    - `tauk`: Controls the self-scaling (tauk=1 means no self-scaling)

    Subclasses must implement `_invhessian_update_term` to compute the new inverse Hessian pytree.
    They may also override `_compute_thetak` and `_compute_tauk` to implement
    specific variants.

    Note: This method only supports `use_inverse=True` as the self-scaling update
    operates on the inverse Hessian.
    """

    use_inverse: bool = True  # Self-scaling only works with inverse Hessian
    _self_scaled: bool = True  # When True, tauk is computed dynamically

    def init_hessian(
        self, y: Y, f: Scalar, grad: Y
    ) -> tuple[_Hessian, _SSBroydenUpdateState]:
        identity_operator = _identity_pytree(y)
        if self.use_inverse:
            f_info = FunctionInfo.EvalGradHessianInv(f, grad, identity_operator)
        else:
            f_info = FunctionInfo.EvalGradHessian(f, grad, identity_operator)
        return f_info, _SSBroydenUpdateState(
            first_step=jnp.array(True),
            step_size=jnp.array(1.0),
        )  # pyright: ignore

    def _compute_thetak(
        self,
        ak: Scalar,
        bk: Scalar,
        hk: Scalar,
        is_first: Bool[Array, ""],
    ) -> Scalar:
        """Compute thetak parameter.

        Override this method to return a fixed value or compute it dynamically.
        Default implementation computes the full self-scaling thetak.

        **Arguments:**
        - `ak`: Intermediate parameter = bk * hk - 1
        - `bk`: Intermediate parameter = -alpha_k * rho * (s_k^T grad_k)
        - `hk`: Intermediate parameter = (y_k^T H_k y_k) / (y_k^T s_k)
        - `is_first`: Whether this is the first iteration

        **Returns:**
        The thetak parameter for the Broyden family update.
        """
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
        """Compute tauk (self-scaling) parameter.

        Override this method to return 1.0 for no self-scaling, or compute dynamically.
        Default implementation computes the full self-scaling tauk.

        **Arguments:**
        - `thetak`: The thetak parameter
        - `ak`: Intermediate parameter = bk * hk - 1
        - `bk`: Intermediate parameter = -alpha_k * rho * (s_k^T grad_k)
        - `hk`: Intermediate parameter = (y_k^T H_k y_k) / (y_k^T s_k)
        - `is_first`: Whether this is the first iteration
        - `N`: Problem dimension

        **Returns:**
        The tauk (scaling) parameter.
        """

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

    @abc.abstractmethod
    def _hessian_update_term(self, **kwargs) -> PyTree:
        """Compute the new Hessian pytree from auxiliary quantities.

        Subclasses must implement this to compute the updated Hessian approximation.
        All quantities are passed as kwargs; each subclass uses what it needs.

        **Keyword Arguments:**
        - `hessian_pytree`: Current inverse Hessian pytree
        - `y_diff`: Step difference s_k = y_{k+1} - y_k
        - `grad_diff`: Gradient difference y_k = grad_{k+1} - grad_k
        - `Hy`: H_k @ y_k (inverse Hessian times gradient difference)
        - `inner`: y_k^T s_k (gradient difference dot step)
        - `yHy`: y_k^T H_k y_k
        - `rho`: 1 / inner
        - `thetak`: Thetak parameter
        - `tauk`: Tauk (scaling) parameter
        - `ak`: Intermediate parameter = bk * hk - 1
        - `bk`: Intermediate parameter
        - `hk`: Intermediate parameter = yHy / inner

        **Returns:**
        The new Hessian pytree.
        """
        ...

    @abc.abstractmethod
    def _invhessian_update_term(self, **kwargs) -> PyTree:
        """Compute the new inverse Hessian pytree from auxiliary quantities.

        Subclasses must implement this to compute the updated inverse Hessian approximation.
        All quantities are passed as kwargs; each subclass uses what it needs.

        **Keyword Arguments:**
        - `hessian_pytree`: Current inverse Hessian pytree
        - `y_diff`: Step difference s_k = y_{k+1} - y_k
        - `grad_diff`: Gradient difference y_k = grad_{k+1} - grad_k
        - `Hy`: H_k @ y_k (inverse Hessian times gradient difference)
        - `inner`: y_k^T s_k (gradient difference dot step)
        - `yHy`: y_k^T H_k y_k
        - `rho`: 1 / inner
        - `thetak`: Thetak parameter
        - `tauk`: Tauk (scaling) parameter
        - `ak`: Intermediate parameter = bk * hk - 1
        - `bk`: Intermediate parameter
        - `hk`: Intermediate parameter = yHy / inner

        **Returns:**
        The new inverse Hessian pytree.
        """
        ...

    def update_hessian(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: _SSBroydenUpdateState,
        step_size: Scalar,
    ) -> tuple[_Hessian, _SSBroydenUpdateState]:
        """Update the Hessian using the self-scaling Broyden family formula."""
        f_eval = f_eval_info.f
        grad = f_eval_info.grad
        y_diff = (y_eval**ω - y**ω).ω
        grad_diff = (grad**ω - f_info.grad**ω).ω
        inner = tree_dot(grad_diff, y_diff)

        # In particular inner = 0 on the first step (as then state.grad=0), and so for
        # this we jump straight to the line search.
        # Likewise we get inner <= eps on convergence, and so again we make no update
        # to avoid a division by zero.
        inner_nonzero = inner > jnp.finfo(inner.dtype).eps

        def no_update(args):
            *_, f_info, hessian_update_state, _step_size = args
            if self.use_inverse:
                return f_info.hessian_inv, hessian_update_state
            else:
                return f_info.hessian, hessian_update_state

        def update(args):
            inner, grad_diff, y_diff, f_info, hessian_update_state, step_sz = args
            # region
            if self.use_inverse:
                assert isinstance(f_info, FunctionInfo.EvalGradHessianInv)

                hessian_inv = f_info.hessian_inv
                rho = 1.0 / inner  # rho = 1 / (y_k^T s_k)

                # H_k * y_k
                Hy = hessian_inv.mv(grad_diff)
                # y_k^T H_k y_k
                yHy = tree_dot(grad_diff, Hy)

                # Self-scaling parameters
                hk = yHy * rho  # hk = (y_k^T H_k y_k) / (y_k^T s_k)

                # bk = -alpha_k * rho * (s_k^T grad_k)
                grad_prev = f_info.grad
                # step_sz is the step size used to go from y to y_eval
                bk = -step_sz * rho * tree_dot(y_diff, grad_prev)

                ak = bk * hk - 1

                # Get dimension from y_diff
                N = sum(jnp.size(leaf) for leaf in jtu.tree_leaves(y_diff))

                # Compute thetak and tauk using the (possibly overridden) methods
                is_first = hessian_update_state.first_step
                thetak = self._compute_thetak(ak, bk, hk, is_first)
                tauk = self._compute_tauk(thetak, ak, bk, hk, is_first, N)

                # Call the subclass-specific update term computation
                new_hessian_pytree = self._invhessian_update_term(
                    hessian_pytree=hessian_inv.pytree,
                    y_diff=y_diff,
                    grad_diff=grad_diff,
                    Hy=Hy,
                    inner=inner,
                    yHy=yHy,
                    rho=rho,
                    thetak=thetak,
                    tauk=tauk,
                    ak=ak,
                    bk=bk,
                    hk=hk,
                )

                # Check for numerical stability
                is_finite = jnp.isfinite(rho) & jnp.isfinite(1 / tauk)

                new_hessian_pytree = jtu.tree_map(
                    lambda new, old: jnp.where(is_finite, new, old),
                    new_hessian_pytree,
                    hessian_inv.pytree,
                )

                new_hessian_inv = lx.PyTreeLinearOperator(
                    new_hessian_pytree,  # pyright: ignore
                    output_structure=jax.eval_shape(lambda: grad_diff),
                    tags=lx.positive_semidefinite_tag,
                )

                new_hessian_update_state = _SSBroydenUpdateState(
                    first_step=jnp.array(False),
                    step_size=step_sz,
                )

                return new_hessian_inv, new_hessian_update_state
            # endregion
            else:
                assert isinstance(f_info, FunctionInfo.EvalGradHessian)

                hessian = f_info.hessian
                # B_k * s_k
                Bs = hessian.mv(y_diff)
                # s_k^T B_k s_k
                sBs = tree_dot(y_diff, Bs)
                rho = 1.0 / inner
                # Self-scaling parameters
                bk = sBs * rho  # bk = (s_k^T B_k s_k) / (y_k^T s_k)
                # I think i could also do: bk = -step_sz * rho * tree_dot(y_diff, grad_prev)

                # The thing is, now I need hk somehow, and for that I need Hk. To get it, I neet to solve the linear system:
                Hy = lx.linear_solve(hessian, grad_diff, solver=lx.Cholesky()).value
                yHy = tree_dot(grad_diff, Hy)
                hk = yHy * rho
                ak = bk * hk - 1
                N = sum(jnp.size(leaf) for leaf in jtu.tree_leaves(y_diff))
                # Compute thetak and tauk using the (possibly overridden) methods
                is_first = hessian_update_state.first_step
                thetak = self._compute_thetak(ak, bk, hk, is_first)
                tauk = self._compute_tauk(thetak, ak, bk, hk, is_first, N)
                # Call the subclass-specific update term computation

                new_hessian_pytree = self._hessian_update_term(
                    hessian_pytree=hessian.pytree,
                    grad_diff=grad_diff,
                    Bs=Bs,
                    sBs=sBs,
                    rho=rho,
                    thetak=thetak,
                    tauk=tauk,
                )

                # Check for numerical stability
                is_finite = jnp.isfinite(rho) & jnp.isfinite(1 / tauk)

                new_hessian_pytree = jtu.tree_map(
                    lambda new, old: jnp.where(is_finite, new, old),
                    new_hessian_pytree,
                    hessian.pytree,
                )

                new_hessian = lx.PyTreeLinearOperator(
                    new_hessian_pytree,  # pyright: ignore
                    output_structure=jax.eval_shape(lambda: grad_diff),
                    tags=lx.positive_semidefinite_tag,
                )

                new_hessian_update_state = _SSBroydenUpdateState(
                    first_step=jnp.array(False),
                    step_size=step_sz,
                )
                return new_hessian, new_hessian_update_state

        args = (inner, grad_diff, y_diff, f_info, hessian_update_state, step_size)
        hessian, new_update_state = filter_cond(
            inner_nonzero,
            update,
            no_update,
            args,
        )

        # Update state for next iteration
        new_update_state = _SSBroydenUpdateState(
            first_step=jnp.array(False),
            step_size=step_size,
        )

        # We're using type: ignore here because the type of `FunctionInfo` depends on
        # the `use_inverse` attribute.
        if self.use_inverse:
            return (
                FunctionInfo.EvalGradHessianInv(f_eval, grad, hessian),  # type: ignore
                new_update_state,
            )
        else:
            return (
                FunctionInfo.EvalGradHessian(f_eval, grad, hessian),  # type: ignore
                new_update_state,
            )


# =============================================================================
# Self-Scaling Broyden (full self-scaling with computed thetak and tauk)
# =============================================================================


class AbstractSSBroyden(AbstractSSBroydenFamily[Y, Aux, _Hessian]):
    """Abstract version of the Self-Scaling Broyden minimisation algorithm.

    This is the full self-scaling Broyden method, which computes both thetak and tauk
    dynamically at each iteration. This provides automatic scaling adjustment of the
    Hessian approximation.

    The update formula is:
    H_{k+1} = (H_k - Hy Hy^T / yHy + phik * yHy * vk vk^T) / tauk + rho * s s^T

    where phik = (1 - thetak) / (1 + ak * thetak) and vk = sk * rho - Hy / yHy.

    This class may be subclassed to implement custom solvers with alternative searches
    and descent methods.

    Note: This method only supports `use_inverse=True` as the self-scaling update
    operates on the inverse Hessian.
    """

    def _hessian_update_term(self, **kwargs) -> PyTree:
        """Compute the Hessian update term (not implemented)."""

        # B_k is the Hessian, not the inverse of the Hessian, so the update is different from the inverse Hessian case.
        hessian_pytree = kwargs["hessian_pytree"]
        grad_diff = kwargs["grad_diff"]
        Bs = kwargs["Bs"]
        sBs = kwargs["sBs"]
        rho = kwargs["rho"]
        thetak = kwargs["thetak"]
        tauk = kwargs["tauk"]

        # wk = yk/(yk^T sk) - Bk sk/(sk^T Bk sk)
        wk = (grad_diff**ω * rho - Bs**ω / sBs).ω

        # Update formula
        # B_{k+1} = (B_k - (B_k s_k s_k^T B_k) / (s_k^T B_k s_k) + thetak * (s_k^T B_k s_k) * wk wk^T) / tauk
        # + yk yk^T / (yk^T sk)
        term1 = _outer(Bs, Bs)
        term2 = _outer(wk, wk)
        term3 = _outer(grad_diff, grad_diff)

        return (
            (hessian_pytree**ω - term1**ω / sBs + thetak * sBs * term2**ω) * tauk
            + term3**ω * rho
        ).ω

    def _invhessian_update_term(self, **kwargs) -> PyTree:
        """Compute the full SS-Broyden inverse Hessian update term.

        Uses the general formula with thetak and tauk computed dynamically.
        """
        hessian_pytree = kwargs["hessian_pytree"]
        y_diff = kwargs["y_diff"]
        Hy = kwargs["Hy"]
        yHy = kwargs["yHy"]
        rho = kwargs["rho"]
        thetak = kwargs["thetak"]
        tauk = kwargs["tauk"]
        ak = kwargs["ak"]

        # v_k = s_k * rho - H_k y_k / (y_k^T H_k y_k)
        vk = (y_diff**ω * rho - Hy**ω / yHy).ω

        # phi_k = (1 - theta_k) / (1 + a_k * theta_k)
        phik = (1 - thetak) / (1 + ak * thetak)

        # Update formula:
        # H_{k+1} = (H_k - Hy Hy^T / yHy + phik * yHy * vk vk^T) / tauk + rho * s s^T
        term1 = _outer(Hy, Hy)
        term2 = _outer(vk, vk)
        term3 = _outer(y_diff, y_diff)

        return (
            (hessian_pytree**ω - term1**ω / yHy + term2**ω * (phik * yHy)) / tauk
            + term3**ω * rho
        ).ω


# =============================================================================
# Broyden (no self-scaling: tauk = 1, computed thetak)
# =============================================================================


class AbstractBroyden(AbstractSSBroyden[Y, Aux, _Hessian]):
    """Abstract version of the Broyden minimisation algorithm (without self-scaling).

    This method computes thetak dynamically but sets tauk = 1 (no self-scaling).
    The thetak parameter controls the convex combination between BFGS-like and
    DFP-like updates.

    This class may be subclassed to implement custom solvers with alternative searches
    and descent methods.
    """

    _self_scaled: bool = False  # tauk = 1, no self-scaling

    def _compute_tauk(
        self,
        thetak: Scalar,
        ak: Scalar,
        bk: Scalar,
        hk: Scalar,
        is_first: Bool[Array, ""],
        N: int,
    ) -> Scalar:
        """Return tauk = 1.0 (no self-scaling)."""
        return jnp.array(1.0)


# =============================================================================
# SS-BFGS (self-scaling with thetak = 0)
# =============================================================================


class AbstractSSBFGS(AbstractSSBroydenFamily[Y, Aux, _Hessian]):
    """Abstract version of the Self-Scaling BFGS minimisation algorithm.

    This method sets thetak = 0 (BFGS-like update) but computes tauk dynamically
    for self-scaling. This provides automatic scaling of the BFGS update.

    When thetak = 0:
    - phik = (1 - 0) / (1 + ak * 0) = 1

    The BFGS update formula (with Woodbury identity) is:
    H_{k+1} = (H_k + (inner + yHy) * s s^T / inner^2 - (Hy s^T + s Hy^T) / inner) / tauk

    This class may be subclassed to implement custom solvers with alternative searches
    and descent methods.
    """

    def _compute_thetak(
        self,
        ak: Scalar,
        bk: Scalar,
        hk: Scalar,
        is_first: Bool[Array, ""],
    ) -> Scalar:
        """Return thetak = 0 (BFGS update)."""
        return jnp.array(0.0)

    def _hessian_update_term(self, **kwargs) -> PyTree:
        """Compute the SS-BFGS Hessian update term using .

        For BFGS (thetak=0), phik=1, and the formula simplifies.
        """
        hessian_pytree = kwargs["hessian_pytree"]
        grad_diff = kwargs["grad_diff"]
        Bs = kwargs["Bs"]
        sBs = kwargs["sBs"]
        rho = kwargs["rho"]
        tauk = kwargs["tauk"]

        # term1 = (Bk sk)(Bk sk)^T / (sk^T Bk sk)
        term1 = _outer(Bs, Bs)
        # term2 = yk yk^T / (yk^T sk)
        term2 = _outer(grad_diff, grad_diff)

        return ((hessian_pytree**ω - term1**ω / sBs) / tauk + term2**ω * rho).ω

    def _invhessian_update_term(self, **kwargs) -> PyTree:
        """Compute the SS-BFGS inverse Hessian update term using Woodbury identity.

        For BFGS (thetak=0), phik=1, and the formula simplifies.
        """
        hessian_pytree = kwargs["hessian_pytree"]
        y_diff = kwargs["y_diff"]
        Hy = kwargs["Hy"]
        inner = kwargs["inner"]
        yHy = kwargs["yHy"]
        tauk = kwargs["tauk"]

        # BFGS update using Woodbury identity
        diff_outer = _outer(y_diff, y_diff)
        mvp_outer = _outer(y_diff, Hy)

        # term1 = ((inner + yHy) * s s^T) / inner^2
        term1 = (((inner + yHy / tauk) * (diff_outer**ω)) / (inner**2)).ω
        # term2 = (Hy s^T + s Hy^T) / inner
        term2 = ((_outer(Hy, y_diff) ** ω + mvp_outer**ω) / inner).ω

        # Apply self-scaling
        return ((hessian_pytree**ω - term2**ω) / tauk + term1**ω).ω


# =============================================================================
# BFGS (thetak = 0, tauk = 1 - inherits from SS-BFGS)
# =============================================================================


class AbstractBFGS(AbstractSSBFGS[Y, Aux, _Hessian]):
    """Abstract version of the BFGS (Broyden–Fletcher–Goldfarb–Shanno) minimisation
    algorithm.

    This method uses thetak = 0 and tauk = 1, which corresponds to the classic BFGS
    update. Inherits the simplified BFGS formula from AbstractSSBFGS.

    When thetak = 0:
    - phik = (1 - 0) / (1 + ak * 0) = 1

    When tauk = 1 (no self-scaling) and phik = 1:
    - The update becomes the standard BFGS formula using the Woodbury identity.

    This class may be subclassed to implement custom solvers with alternative searches
    and descent methods that use the BFGS update to approximate the Hessian or
    the inverse Hessian.
    """

    _self_scaled: bool = False  # tauk = 1, no self-scaling

    def _compute_tauk(
        self,
        thetak: Scalar,
        ak: Scalar,
        bk: Scalar,
        hk: Scalar,
        is_first: Bool[Array, ""],
        N: int,
    ) -> Scalar:
        """Return tauk = 1.0 (no self-scaling)."""
        return jnp.array(1.0)


class BFGS(AbstractBFGS[Y, Aux, _Hessian]):
    """BFGS (Broyden–Fletcher–Goldfarb–Shanno) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information. Uses the Broyden-Fletcher-Goldfarb-Shanno formula to compute the
    updates to the Hessian and or to the Hessian inverse.
    See [https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm).

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: BacktrackingArmijo
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: bool | Callable[..., None] = False,
        search: AbstractSearch = Zoom(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        self.search = search
        self.verbose = default_verbose(verbose)


BFGS.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `use_inverse`: If `True` (default), approximate the inverse Hessian. If `False`,
    approximate the Hessian.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Can either be `False` to print out nothing, or `True` to print out all information,
    or (for customisation) a callable `**kwargs -> None`. If provided as a callable then
    each value will be a 2-tuple of `(str, jax.Array)` providing a human-readable name
    and its corresponding value.
"""


# =============================================================================
# SS-DFP (self-scaling with thetak = 1)
# =============================================================================


class AbstractSSDFP(AbstractSSBroydenFamily[Y, Aux, _Hessian]):
    """Abstract version of the Self-Scaling DFP minimisation algorithm.

    This method sets thetak = 1 (DFP-like update) but computes tauk dynamically
    for self-scaling. This provides automatic scaling of the DFP update.

    When thetak = 1:
    - phik = (1 - 1) / (1 + ak * 1) = 0

    Since phik = 0, the term with vk disappears from the update formula,
    simplifying the computation (no need to compute vk).

    The DFP update formula is:
    H_{k+1} = (H_k + s s^T / inner - Hy Hy^T / yHy) / tauk

    This class may be subclassed to implement custom solvers with alternative searches
    and descent methods.
    """

    def _compute_thetak(
        self,
        ak: Scalar,
        bk: Scalar,
        hk: Scalar,
        is_first: Bool[Array, ""],
    ) -> Scalar:
        """Return thetak = 1 (DFP update)."""
        return jnp.array(1.0)

    def _hessian_update_term(self, **kwargs) -> PyTree:
        """Compute the SS-DFP Hessian update term.

        For DFP (thetak=1), phik=0, and the formula simplifies to
        a Woodbury identity form (dual of BFGS inverse Hessian update).
        """
        hessian_pytree = kwargs["hessian_pytree"]
        grad_diff = kwargs["grad_diff"]
        Bs = kwargs["Bs"]
        sBs = kwargs["sBs"]
        rho = kwargs["rho"]
        tauk = kwargs["tauk"]

        inner = 1.0 / rho

        # DFP Hessian update using Woodbury identity
        diff_outer = _outer(grad_diff, grad_diff)
        mvp_outer = _outer(grad_diff, Bs)

        # term1 = ((inner + sBs/tauk) * yk yk^T) / inner^2
        term1 = (((inner + sBs * tauk) * (diff_outer**ω)) / (inner**2)).ω
        # term2 = (Bs yk^T + yk Bs^T) / inner
        term2 = ((_outer(Bs, grad_diff) ** ω + mvp_outer**ω) / inner).ω

        return ((hessian_pytree**ω - term2**ω) * tauk + term1**ω).ω

    def _invhessian_update_term(self, **kwargs) -> PyTree:
        """Compute the SS-DFP inverse Hessian update term.

        For DFP (thetak=1), phik=0, so the vk term disappears.
        """
        hessian_pytree = kwargs["hessian_pytree"]
        y_diff = kwargs["y_diff"]
        Hy = kwargs["Hy"]
        inner = kwargs["inner"]
        yHy = kwargs["yHy"]
        tauk = kwargs["tauk"]

        # DFP update: H_{k+1} = (H_k + s s^T / inner - Hy Hy^T / yHy) / tauk
        term1 = (_outer(Hy, Hy) ** ω / yHy).ω
        term2 = (_outer(y_diff, y_diff) ** ω / inner).ω

        # Apply self-scaling
        return ((hessian_pytree**ω - term1**ω) / tauk + term2**ω).ω


# =============================================================================
# DFP (thetak = 1, tauk = 1 - inherits from SS-DFP)
# =============================================================================


class AbstractDFP(AbstractSSDFP[Y, Aux, _Hessian]):
    """Abstract version of the DFP (Davidon–Fletcher–Powell) minimisation algorithm.

    This method uses thetak = 1 and tauk = 1, which corresponds to the classic DFP
    update. Inherits the simplified DFP formula from AbstractSSDFP.

    When thetak = 1:
    - phik = (1 - 1) / (1 + ak * 1) = 0

    When tauk = 1 (no self-scaling) and phik = 0:
    - The update becomes the standard DFP formula.

    This class may be subclassed to implement custom solvers with alternative searches
    and descent methods that use the DFP update to approximate the Hessian or the
    inverse Hessian.
    """

    _self_scaled: bool = False  # tauk = 1, no self-scaling

    def _compute_tauk(
        self,
        thetak: Scalar,
        ak: Scalar,
        bk: Scalar,
        hk: Scalar,
        is_first: Bool[Array, ""],
        N: int,
    ) -> Scalar:
        """Return tauk = 1.0 (no self-scaling)."""
        return jnp.array(1.0)


class DFP(AbstractDFP[Y, Aux, _Hessian]):
    """DFP (Davidon–Fletcher–Powell) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information. Uses the Davidon-Fletcher-Powell formula to compute the updates to
    the Hessian and or to the Hessian inverse.
    See [https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula](https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula).

    [`optimistix.BFGS`][] is generally preferred, since it is more numerically stable on
    most problems.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: BacktrackingArmijo
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: bool | Callable[..., None] = False,
        search: AbstractSearch = Zoom(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        self.search = search
        self.verbose = default_verbose(verbose)


DFP.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `use_inverse`: If `True` (default), approximate the inverse Hessian. If `False`,
    approximate the Hessian.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Can either be `False` to print out nothing, or `True` to print out all information,
    or (for customisation) a callable `**kwargs -> None`. If provided as a callable then
    each value will be a 2-tuple of `(str, jax.Array)` providing a human-readable name
    and its corresponding value.
"""

# =============================================================================
# Concrete Implementations
# =============================================================================


class SSBroyden(AbstractSSBroyden[Y, Aux, _Hessian]):
    """Self-Scaling Broyden minimisation algorithm.

    This is a quasi-Newton optimisation algorithm that uses a self-scaling update
    for the inverse Hessian approximation. The self-scaling mechanism automatically
    adjusts the scaling of the Hessian approximation at each iteration, which can
    improve convergence in some cases.

    The self-scaling update is based on the Broyden family of quasi-Newton methods
    with automatic scaling parameter selection.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: AbstractSearch
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: bool | Callable[..., None] = False,
        search: AbstractSearch = Zoom(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        self.search = search
        self.verbose = default_verbose(verbose)


SSBroyden.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `use_inverse`: If `True` (default), approximate the inverse Hessian. If `False`,
    approximate the Hessian.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Can either be `False` to print out nothing, or `True` to print out all information,
    or (for customisation) a callable `**kwargs -> None`. If provided as a callable then
    each value will be a 2-tuple of `(str, jax.Array)` providing a human-readable name
    and its corresponding value.
"""


class Broyden(AbstractBroyden[Y, Aux, _Hessian]):
    """Broyden minimisation algorithm (without self-scaling).

    This is a quasi-Newton optimisation algorithm that uses the Broyden family update
    for the inverse Hessian approximation. The thetak parameter is computed dynamically
    to control the convex combination between BFGS-like and DFP-like updates, but
    tauk = 1 (no self-scaling).

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: AbstractSearch
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: bool | Callable[..., None] = False,
        search: AbstractSearch = Zoom(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        self.search = search
        self.verbose = default_verbose(verbose)


Broyden.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `use_inverse`: If `True` (default), approximate the inverse Hessian. If `False`,
    approximate the Hessian.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Can either be `False` to print out nothing, or `True` to print out all information,
    or (for customisation) a callable `**kwargs -> None`. If provided as a callable then
    each value will be a 2-tuple of `(str, jax.Array)` providing a human-readable name
    and its corresponding value.
"""


class SSBFGS(AbstractSSBFGS[Y, Aux, _Hessian]):
    """Self-Scaling BFGS minimisation algorithm.

    This is a quasi-Newton optimisation algorithm that uses a self-scaling BFGS update
    for the inverse Hessian approximation. It sets thetak = 0 (BFGS-like update) but
    computes tauk dynamically for self-scaling.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: AbstractSearch
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: bool | Callable[..., None] = False,
        search: AbstractSearch = Zoom(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        self.search = search
        self.verbose = default_verbose(verbose)


SSBFGS.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `use_inverse`: If `True` (default), approximate the inverse Hessian. If `False`,
    approximate the Hessian.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Can either be `False` to print out nothing, or `True` to print out all information,
    or (for customisation) a callable `**kwargs -> None`. If provided as a callable then
    each value will be a 2-tuple of `(str, jax.Array)` providing a human-readable name
    and its corresponding value.
"""


class SSDFP(AbstractSSDFP[Y, Aux, _Hessian]):
    """Self-Scaling DFP minimisation algorithm.

    This is a quasi-Newton optimisation algorithm that uses a self-scaling DFP update
    for the inverse Hessian approximation. It sets thetak = 1 (DFP-like update) but
    computes tauk dynamically for self-scaling.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: AbstractSearch
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: bool | Callable[..., None] = False,
        search: AbstractSearch = Zoom(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        self.search = search
        self.verbose = default_verbose(verbose)


SSDFP.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `use_inverse`: If `True` (default), approximate the inverse Hessian. If `False`,
    approximate the Hessian.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Can either be `False` to print out nothing, or `True` to print out all information,
    or (for customisation) a callable `**kwargs -> None`. If provided as a callable then
    each value will be a 2-tuple of `(str, jax.Array)` providing a human-readable name
    and its corresponding value.
"""
