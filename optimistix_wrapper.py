"""
Wrapper for optimistix optimizers with manual step loop, JIT compilation,
and proper iteration counting.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

import optimistix


@dataclass
class OptimizationResult:
    """Result of optimization."""

    params: Any
    num_iters: int
    result: optimistix.RESULTS
    final_loss: float
    final_aux: Any = None
    stats: Any = None


def make_optimistix_loss(loss_fn: Callable) -> Callable:
    """Wrap a loss function to match optimistix signature (y, args) -> (loss, aux)."""

    def wrapped(y, args):
        return loss_fn(y), None

    return wrapped


def wrap_loss_if_needed(loss_fn: Callable, test_params: Any) -> Callable:
    """
    Wrap loss_fn for optimistix if it doesn't already have the right signature.

    Optimistix expects: (y, args) -> (loss, aux)
    Standard loss: (params) -> scalar
    """
    # FIXME: Allow more general functions that take args, but for now just check if it already has the right signature
    return make_optimistix_loss(loss_fn)


def run_optimization(
    solver: optimistix.AbstractMinimiser,
    loss_fn: Callable,
    init_params: Any,
    maxiter: int = 1000,
    callback: Optional[Callable[[int, Any, Any, float], bool]] = None,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Run optimistix solver with manual step loop for full control.

    Args:
        solver: An optimistix minimizer (BFGS, LBFGS, etc.)
        loss_fn: Loss function with signature (params) -> scalar
        init_params: Initial parameters (pytree)
        maxiter: Maximum number of iterations
        callback: Optional callback(iter, params, state, loss) -> should_stop
                  Called each accepted step. Return True to stop early.
        verbose: Print progress messages

    Returns:
        OptimizationResult with final params, iteration count, and status
    """
    # Wrap loss for optimistix if needed: (y, args) -> (loss, aux)
    loss_optimistix = wrap_loss_if_needed(loss_fn, init_params)

    # Setup solver internals
    args = None
    f_struct = jax.ShapeDtypeStruct((), jnp.float64)
    aux_struct = None
    tags = frozenset()
    options = {}

    # JIT compile step and terminate with partial application
    step = eqx.Partial(
        solver.step, fn=loss_optimistix, args=args, options=options, tags=tags
    )
    step = eqx.filter_jit(step)

    terminate = eqx.Partial(
        solver.terminate, fn=loss_optimistix, args=args, options=options, tags=tags
    )
    terminate = eqx.filter_jit(terminate)

    # Initialize
    params = deepcopy(init_params)
    state = solver.init(
        loss_optimistix, params, args, options, f_struct, aux_struct, tags
    )
    done, result = terminate(y=params, state=state)

    # Warm-up compilation
    if verbose:
        print("Compiling solver...")
    p_dummy, state_dummy, aux_dummy = step(y=params, state=state)
    _ = terminate(y=p_dummy, state=state_dummy)
    p_dummy, state_dummy, aux_dummy = step(y=p_dummy, state=state_dummy)
    _ = terminate(y=p_dummy, state=state_dummy)
    if verbose:
        print("Compilation done.")

    # Main optimization loop
    iter_count = 0
    aux = None

    while not done:
        params, state, aux = step(y=params, state=state)
        done, result = terminate(y=params, state=state)

        # Check for accepted step (line search solvers track this)
        if hasattr(state, "search_state") and hasattr(state.search_state, "accepted"):
            accepted = state.search_state.accepted
        else:
            accepted = True

        if accepted:
            iter_count += 1

            # Compute current loss for callback
            if callback is not None:
                current_loss = float(loss_optimistix(params, None)[0])
                should_stop = callback(iter_count, params, state, current_loss)
                if should_stop:
                    break

        if iter_count >= maxiter:
            done = True

    # Postprocess
    params, final_aux, stats = solver.postprocess(
        loss_optimistix, params, aux, args, options, state, tags, result
    )

    final_loss = float(loss_optimistix(params, None)[0])

    if verbose and result != optimistix.RESULTS.successful:
        print(f"Solver finished with status: {result}")

    return OptimizationResult(
        params=params,
        num_iters=iter_count,
        result=result,
        final_loss=final_loss,
        final_aux=final_aux,
        stats=stats,
    )
