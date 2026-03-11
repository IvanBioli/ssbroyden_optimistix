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


class AbstractBFGS(AbstractQuasiNewton[Y, Aux, _Hessian, None]):
    """Abstract version of the BFGS (Broyden–Fletcher–Goldfarb–Shanno) minimisation
    algorithm. This class may be subclassed to implement custom solvers with alternative
    searches and descent methods that use the BFGS update to approximate the Hessian or
    the inverse Hessian.
    """

    def init_hessian(self, y: Y, f: Scalar, grad: Y) -> tuple[_Hessian, None]:
        identity_operator = _identity_pytree(y)
        if self.use_inverse:
            f_info = FunctionInfo.EvalGradHessianInv(f, grad, identity_operator)
        else:
            f_info = FunctionInfo.EvalGradHessian(f, grad, identity_operator)
        return f_info, None  # pyright: ignore

    def update_hessian(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: None,
        step_size: Scalar,  # noqa: ARG002
    ) -> tuple[_Hessian, None]:
        del step_size  # Unused in BFGS
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
            *_, f_info = args
            if self.use_inverse:
                return f_info.hessian_inv
            else:
                return f_info.hessian

        def update(args):
            inner, grad_diff, y_diff, f_info = args
            if self.use_inverse:
                assert isinstance(f_info, FunctionInfo.EvalGradHessianInv)
                hessian_inv = f_info.hessian_inv
                # Use Woodbury identity for rank-1 update of approximate Hessian.
                inv_mvp = hessian_inv.mv(grad_diff)
                mvp_inner = tree_dot(grad_diff, inv_mvp)
                diff_outer = _outer(y_diff, y_diff)
                mvp_outer = _outer(y_diff, inv_mvp)
                term1 = (((inner + mvp_inner) * (diff_outer**ω)) / (inner**2)).ω
                term2 = ((_outer(inv_mvp, y_diff) ** ω + mvp_outer**ω) / inner).ω
                new_hessian_inv = lx.PyTreeLinearOperator(
                    (hessian_inv.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                    output_structure=jax.eval_shape(lambda: grad_diff),
                    tags=lx.positive_semidefinite_tag,
                )
                return new_hessian_inv
            else:
                assert isinstance(f_info, FunctionInfo.EvalGradHessian)
                hessian = f_info.hessian
                # BFGS update to the operator directly
                mvp = hessian.mv(y_diff)
                term1 = (_outer(grad_diff, grad_diff) ** ω / inner).ω
                term2 = (_outer(mvp, mvp) ** ω / tree_dot(y_diff, mvp)).ω
                new_hessian = lx.PyTreeLinearOperator(
                    (hessian.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                    output_structure=jax.eval_shape(lambda: grad_diff),
                    tags=lx.positive_semidefinite_tag,
                )
                return new_hessian

        args = (inner, grad_diff, y_diff, f_info)
        hessian = filter_cond(
            inner_nonzero,
            update,
            no_update,
            args,
        )

        # We're using type: ignore here because the type of `FunctionInfo` depends on
        # the `use_inverse` attribute.
        # https://github.com/patrick-kidger/optimistix/pull/135#discussion_r2155452558
        if self.use_inverse:
            return (
                FunctionInfo.EvalGradHessianInv(f_eval, grad, hessian),  # type: ignore
                None,
            )
        else:
            return (
                FunctionInfo.EvalGradHessian(f_eval, grad, hessian),  # type: ignore
                None,
            )


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
- `use_inverse`: The BFGS algorithm involves computing matrix-vector products of the
    form `B^{-1} g`, where `B` is an approximation to the Hessian of the function to be
    minimised. This means we can either (a) store the approximate Hessian `B`, and do a
    linear solve on every step, or (b) store the approximate Hessian inverse `B^{-1}`,
    and do a matrix-vector product on every step. Option (a) is generally cheaper for
    sparse Hessians (as the inverse may be dense). Option (b) is generally cheaper for
    dense Hessians (as matrix-vector products are cheaper than linear solves). The
    default is (b), denoted via `use_inverse=True`. Note that this is incompatible with
    searches like [`optimistix.ClassicalTrustRegion`][], which use the Hessian 
    approximation `B` as part of their computations.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Can either be `False` to print out nothing, or `True` to print out all information,
    or (for customisation) a callable `**kwargs -> None`. If provided as a callable then
    each value will be a 2-tuple of `(str, jax.Array)` providing a human-readable name
    and its corresponding value.
"""


class AbstractDFP(AbstractQuasiNewton[Y, Aux, _Hessian, None]):
    """Abstract version of the DFP (Davidon–Fletcher–Powell) minimisation algorithm.
    This class may be subclassed to implement custom solvers with alternative searches
    and descent methods that use the DFP update to approximate the Hessian or the
    inverse Hessian.
    """

    def init_hessian(self, y: Y, f: Scalar, grad: Y) -> tuple[_Hessian, None]:
        identity_operator = _identity_pytree(y)
        if self.use_inverse:
            f_info = FunctionInfo.EvalGradHessianInv(f, grad, identity_operator)
        else:
            f_info = FunctionInfo.EvalGradHessian(f, grad, identity_operator)
        return f_info, None  # pyright: ignore

    def update_hessian(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: None,
        step_size: Scalar,  # noqa: ARG002
    ) -> tuple[_Hessian, None]:
        del step_size  # Unused in DFP
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
            *_, f_info = args
            if self.use_inverse:
                return f_info.hessian_inv
            else:
                return f_info.hessian

        def update(args):
            inner, grad_diff, y_diff, f_info = args
            if self.use_inverse:
                assert isinstance(f_info, FunctionInfo.EvalGradHessianInv)
                hessian_inv = f_info.hessian_inv
                inv_mvp = hessian_inv.mv(grad_diff)
                term1 = (_outer(y_diff, y_diff) ** ω / inner).ω
                term2 = (_outer(inv_mvp, inv_mvp) ** ω / tree_dot(grad_diff, inv_mvp)).ω
                new_hessian_inv = lx.PyTreeLinearOperator(
                    (hessian_inv.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                    output_structure=jax.eval_shape(lambda: grad_diff),
                    tags=lx.positive_semidefinite_tag,
                )
                return new_hessian_inv
            else:
                assert isinstance(f_info, FunctionInfo.EvalGradHessian)
                hessian = f_info.hessian
                mvp = hessian.mv(y_diff)
                mvp_inner = tree_dot(y_diff, mvp)
                diff_outer = _outer(grad_diff, grad_diff)
                mvp_outer = _outer(grad_diff, mvp)
                term1 = (((inner + mvp_inner) * (diff_outer**ω)) / (inner**2)).ω
                term2 = ((_outer(mvp, grad_diff) ** ω + mvp_outer**ω) / inner).ω
                new_hessian = lx.PyTreeLinearOperator(
                    (hessian.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                    output_structure=jax.eval_shape(lambda: grad_diff),
                    tags=lx.positive_semidefinite_tag,
                )
                return new_hessian

        args = (inner, grad_diff, y_diff, f_info)
        hessian = filter_cond(
            inner_nonzero,
            update,
            no_update,
            args,
        )

        # We're using type: ignore here because the type of `FunctionInfo` depends on
        # the `use_inverse` attribute.
        # https://github.com/patrick-kidger/optimistix/pull/135#discussion_r2155452558
        if self.use_inverse:
            return (
                FunctionInfo.EvalGradHessianInv(f_eval, grad, hessian),  # type: ignore
                None,
            )
        else:
            return (
                FunctionInfo.EvalGradHessian(f_eval, grad, hessian),  # type: ignore
                None,
            )


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
- `use_inverse`: The DFP algorithm involves computing matrix-vector products of the
    form `B^{-1} g`, where `B` is an approximation to the Hessian of the function to be
    minimised. This means we can either (a) store the approximate Hessian `B`, and do a
    linear solve on every step, or (b) store the approximate Hessian inverse `B^{-1}`,
    and do a matrix-vector product on every step. Option (a) is generally cheaper for
    sparse Hessians (as the inverse may be dense). Option (b) is generally cheaper for
    dense Hessians (as matrix-vector products are cheaper than linear solves). The
    default is (b), denoted via `use_inverse=True`. Note that this is incompatible with
    searches like [`optimistix.ClassicalTrustRegion`][], which use the Hessian 
    approximation `B` as part of their computations.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Can either be `False` to print out nothing, or `True` to print out all information,
    or (for customisation) a callable `**kwargs -> None`. If provided as a callable then
    each value will be a 2-tuple of `(str, jax.Array)` providing a human-readable name
    and its corresponding value.
"""


class _SSBroydenUpdateState(eqx.Module):
    """State for the self-scaling Broyden update."""

    first_step: Bool[Array, ""]
    step_size: Scalar


class AbstractSSBroyden(AbstractQuasiNewton[Y, Aux, _Hessian, _SSBroydenUpdateState]):
    """Abstract version of the Self-Scaling Broyden minimisation algorithm.

    This is a quasi-Newton algorithm that uses a self-scaling update for the
    inverse Hessian approximation. The self-scaling mechanism automatically adjusts
    the scaling of the Hessian approximation at each iteration.

    This class may be subclassed to implement custom solvers with alternative searches
    and descent methods.

    Note: This method only supports `use_inverse=True` as the self-scaling update
    operates on the inverse Hessian.
    """

    use_inverse: bool = True  # Self-scaling only works with inverse Hessian

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

    def update_hessian(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: _SSBroydenUpdateState,
        step_size: Scalar,
    ) -> tuple[_Hessian, _SSBroydenUpdateState]:
        """Update the Hessian using self-scaling Broyden formula."""
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

                # Compute ck
                ck = jnp.sqrt(jnp.abs(ak / (1 + ak)))

                # Compute rhokm and theta bounds
                rhokm = jnp.minimum(1.0, hk * (1 - ck))
                thetakm = (rhokm - 1) / ak
                thetakp = 1 / rhokm
                thetak = jnp.maximum(thetakm, jnp.minimum(thetakp, (1 - bk) / bk))

                # Compute tauk based on iteration
                is_first = hessian_update_state.first_step

                def first_iter_tauk(_):
                    return hk / (1 + ak * thetak)

                def later_iter_tauk(_):
                    # Get dimension from y_diff
                    N = sum(jnp.size(leaf) for leaf in jtu.tree_leaves(y_diff))
                    rhokk = jnp.minimum(1.0, 1.0 / bk)
                    sigmak = 1 + thetak * ak
                    sigmaknm1 = jnp.abs(sigmak) ** (1.0 / (1.0 - N))
                    return jax.lax.cond(
                        thetak <= 0,
                        lambda _: jnp.minimum(rhokk * sigmaknm1, sigmak),
                        lambda _: rhokk * jnp.minimum(sigmaknm1, 1 / thetak),
                        operand=None,
                    )

                tauk = jax.lax.cond(
                    is_first, first_iter_tauk, later_iter_tauk, operand=None
                )

                # v_k = s_k * rho - H_k y_k / (y_k^T H_k y_k)
                vk = (y_diff**ω * rho - Hy**ω / yHy).ω

                # phi_k = (1 - theta_k) / (1 + a_k * theta_k)
                phik = (1 - thetak) / (1 + ak * thetak)

                # Update formula:
                # H_{k+1} = (H_k - Hy Hy^T / yHy + phik * yHy * vk vk^T) / tauk
                #           + rho * s s^T
                term1 = _outer(Hy, Hy)
                term2 = _outer(vk, vk)
                term3 = _outer(y_diff, y_diff)

                new_hessian_pytree = (
                    (hessian_inv.pytree**ω - term1**ω / yHy + term2**ω * (phik * yHy))
                    / tauk
                    + term3**ω * rho
                ).ω

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
            else:
                assert isinstance(f_info, FunctionInfo.EvalGradHessian)
                raise NotImplementedError(
                    "Self-scaling Broyden update only implemented for inverse Hessian."
                )

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
        # https://github.com/patrick-kidger/optimistix/pull/135#discussion_r2155452558
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
        verbose: bool | Callable[..., None] = False,
        search: AbstractSearch = Zoom(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = True  # Self-scaling only works with inverse Hessian
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
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Can either be `False` to print out nothing, or `True` to print out all information,
    or (for customisation) a callable `**kwargs -> None`. If provided as a callable then
    each value will be a 2-tuple of `(str, jax.Array)` providing a human-readable name
    and its corresponding value.

Note: This method always uses `use_inverse=True` as the self-scaling update operates
on the inverse Hessian approximation.
"""
