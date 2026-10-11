"""A host-side optimisation loop that counts iterations as accepted steps.

``optimistix.minimise`` counts every call of ``step``, including the trial points of
the line search. :func:`run` drives ``step`` from Python instead, so that a budget
of iterations and callbacks refer to accepted steps (as in the quasi-Newton
literature), and the loop can be stopped or logged at any accepted step.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix


@dataclass
class RunResult:
    params: Any
    num_iters: int  # accepted steps
    num_steps: int  # calls of `step`
    result: optimistix.RESULTS
    final_loss: float
    stats: Any = None


def _num_accepted(state, previous: int, accepted_fallback: bool) -> int:
    if hasattr(state, "num_accepted_steps"):
        return int(state.num_accepted_steps)
    if hasattr(state, "search_state") and hasattr(state.search_state, "accepted"):
        return previous + int(state.search_state.accepted)
    return previous + int(accepted_fallback)


def run(
    solver: optimistix.AbstractMinimiser,
    loss_fn: Callable[[Any], Any],
    init_params: Any,
    maxiter: int = 1000,
    callback: Callable[[int, Any, Any, float], bool] | None = None,
    verbose: bool = False,
) -> RunResult:
    """Minimise ``loss_fn(params) -> scalar`` with ``solver`` for ``maxiter`` accepted steps.

    ``callback(iteration, params, state, loss)`` is called after every accepted step;
    returning ``True`` stops the loop. The loop also stops when the solver terminates.
    """

    def fn(y, args):
        return loss_fn(y), None

    options, tags = {}, frozenset()
    leaves = jax.tree_util.tree_leaves(init_params)
    f_struct = jax.ShapeDtypeStruct((), leaves[0].dtype if leaves else jnp.float64)
    step = eqx.filter_jit(eqx.Partial(solver.step, fn=fn, args=None, options=options, tags=tags))
    terminate = eqx.filter_jit(
        eqx.Partial(solver.terminate, fn=fn, args=None, options=options, tags=tags)
    )

    params = deepcopy(init_params)
    state = solver.init(fn, params, None, options, f_struct, None, tags)
    done, result = terminate(y=params, state=state)

    num_iters = num_steps = 0
    while not done and num_iters < maxiter:
        params, state, _ = step(y=params, state=state)
        done, result = terminate(y=params, state=state)
        num_steps += 1
        accepted = _num_accepted(state, num_iters, True)
        if accepted > num_iters:
            num_iters = accepted
            if callback is not None:
                loss = float(fn(params, None)[0])
                if callback(num_iters, params, state, loss):
                    break

    params, _, stats = solver.postprocess(fn, params, None, None, options, state, tags, result)
    final_loss = float(fn(params, None)[0])
    if verbose and result != optimistix.RESULTS.successful:
        print(f"Solver finished with status: {result}")
    return RunResult(params, num_iters, num_steps, result, final_loss, stats)
