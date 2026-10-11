# SPDX-License-Identifier: Apache-2.0
#
# Small private helpers vendored from Optimistix (https://github.com/patrick-kidger/
# optimistix, Apache-2.0, copyright the Optimistix authors), so that this package only
# relies on Optimistix' public API. Bodies unchanged; ``lin_to_grad`` is the v0.0.11
# form (Python-float cotangent).

from collections.abc import Callable
from typing import Any

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar
from lineax.internal import max_norm, sum_squares, tree_dot  # noqa: F401


def tree_full_like(struct: PyTree, fill_value: ArrayLike, allow_static: bool = False):
    fn = lambda x: jnp.full(x.shape, fill_value, x.dtype)
    if isinstance(fill_value, (int, float)):
        if fill_value == 0:
            fn = lambda x: jnp.zeros(x.shape, x.dtype)
        elif fill_value == 1:
            fn = lambda x: jnp.ones(x.shape, x.dtype)
    if allow_static:
        _fn = fn
        fn = (
            lambda x: _fn(x)
            if eqx.is_array(x) or isinstance(x, jax.ShapeDtypeStruct)
            else x
        )
    return jtu.tree_map(fn, struct)


def tree_where(pred, true, false):
    return jtu.tree_map(
        lambda p, t, f: jtu.tree_map(lambda ti, fi: jnp.where(p, ti, fi), t, f),
        pred,
        true,
        false,
    )


def lin_to_grad(lin_fn, y_eval, autodiff_mode=None):
    if autodiff_mode == "bwd":
        (grad,) = jax.linear_transpose(lin_fn, y_eval)(1.0)  # (1.0 is a scaling factor)
        return grad
    if autodiff_mode == "fwd":
        return jax.jacfwd(lin_fn)(y_eval)
    else:
        raise ValueError(
            "Only `autodiff_mode='fwd'` or `autodiff_mode='bwd'` are valid."
        )


def cauchy_termination(
    rtol: float,
    atol: float,
    norm: Callable[[PyTree], Scalar],
    y: Any,
    y_diff: Any,
    f: Any,
    f_diff: Any,
) -> Bool[Array, ""]:
    y_scale = (atol + rtol * ω(y).call(jnp.abs)).ω
    f_scale = (atol + rtol * ω(f).call(jnp.abs)).ω
    y_converged = norm((ω(y_diff).call(jnp.abs) / y_scale**ω).ω) < 1
    f_converged = norm((ω(f_diff).call(jnp.abs) / f_scale**ω).ω) < 1
    return y_converged & f_converged


def filter_cond(pred, true_fun, false_fun, *operands):
    dynamic, static = eqx.partition(operands, eqx.is_array)

    def _true_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = true_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    def _false_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = false_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    dynamic_out, static_out = lax.cond(pred, _true_fun, _false_fun, dynamic)
    return eqx.combine(dynamic_out, static_out.value)


def verbose_print(*args: tuple[bool, str, Any]) -> None:
    string_pieces = []
    arg_pieces = []
    for display, name, value in args:
        if display:
            string_pieces.append(name + ": {}")
            arg_pieces.append(value)
    if len(string_pieces) > 0:
        string = ", ".join(string_pieces)
        jax.debug.print(string, *arg_pieces)


def needs_grad_at_y_eval(search) -> bool:
    """Whether ``search`` needs the gradient at ``y_eval`` before deciding.

    Reads the ``info_needed_at_y_eval`` convention proposed upstream (optimistix#232)
    and, for searches written against the old fork, ``_needs_grad_at_y_eval``.
    """
    from optimistix import FunctionInfo

    info = getattr(type(search), "info_needed_at_y_eval", None)
    if isinstance(info, type):
        return issubclass(info, FunctionInfo.EvalGrad)
    return bool(getattr(type(search), "_needs_grad_at_y_eval", False))
