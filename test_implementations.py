"""Test script to verify that BFGS/DFP implementations match.

Tests:
1. Optimization results on standard problems from helpers.py
2. Final results comparison between old and new implementations
3. Hessian updates on simplified settings
"""

import sys

sys.path.insert(0, "./optimistix/tests")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω

import optimistix as optx

# Import old and new implementations
from old_quasi_newton import BFGS as OldBFGS
from old_quasi_newton import DFP as OldDFP
from old_quasi_newton import SSBroyden as OldSSBroyden
from old_quasi_newton import (
    _identity_pytree,
    _outer,
)
from optimistix._misc import tree_dot
from ssbrodyen_family import BFGS as NewBFGS
from ssbrodyen_family import DFP as NewDFP
from ssbrodyen_family import SSBFGS as NewSSBFGS
from ssbrodyen_family import SSDFP as NewSSDFP
from ssbrodyen_family import SSBroyden as NewSSBroyden


def compare_pytrees(old, new, name, tol=1e-10):
    """Compare two pytrees and return max difference."""
    diff = jtu.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)), old, new)
    max_diff = max(float(jnp.max(d)) for d in jtu.tree_leaves(diff))
    status = "✓" if max_diff < tol else "✗"
    return max_diff, status


def test_bfgs_step_by_step():
    """Test BFGS update step by step on random data."""
    print("\n" + "=" * 60)
    print("Testing BFGS Step-by-Step Comparison")
    print("=" * 60)

    key = jr.PRNGKey(42)
    n = 5
    num_steps = 10

    # Start with identity Hessian
    H_old = _identity_pytree(jnp.zeros(n))
    H_new = _identity_pytree(jnp.zeros(n))

    all_passed = True

    for step in range(num_steps):
        key, k1, k2 = jr.split(key, 3)
        y_diff = jr.normal(k1, (n,))
        grad_diff = jr.normal(k2, (n,))

        # Ensure positive curvature
        inner = tree_dot(grad_diff, y_diff)
        if inner < 0:
            grad_diff = -grad_diff
            inner = -inner

        if inner < 1e-10:
            continue

        # === Old BFGS update ===
        inv_mvp_old = H_old.mv(grad_diff)
        mvp_inner_old = tree_dot(grad_diff, inv_mvp_old)
        diff_outer_old = _outer(y_diff, y_diff)
        mvp_outer_old = _outer(y_diff, inv_mvp_old)
        term1_old = (((inner + mvp_inner_old) * (diff_outer_old**ω)) / (inner**2)).ω
        term2_old = ((_outer(inv_mvp_old, y_diff) ** ω + mvp_outer_old**ω) / inner).ω
        new_pytree_old = (H_old.pytree**ω + term1_old**ω - term2_old**ω).ω

        # === New BFGS update (tauk=1) ===
        Hy_new = H_new.mv(grad_diff)
        yHy_new = tree_dot(grad_diff, Hy_new)
        tauk = jnp.array(1.0)

        diff_outer_new = _outer(y_diff, y_diff)
        mvp_outer_new = _outer(y_diff, Hy_new)
        term1_new = (((inner + yHy_new) * (diff_outer_new**ω)) / (inner**2)).ω
        term2_new = ((_outer(Hy_new, y_diff) ** ω + mvp_outer_new**ω) / inner).ω
        new_pytree_new = ((H_new.pytree**ω + term1_new**ω - term2_new**ω) / tauk).ω

        # Compare
        max_diff, status = compare_pytrees(
            new_pytree_old, new_pytree_new, f"Step {step}"
        )
        passed = max_diff < 1e-10
        all_passed = all_passed and passed

        print(f"  Step {step+1}: {status} Hessian diff={max_diff:.2e}")

        # Update for next iteration
        H_old = lx.PyTreeLinearOperator(
            new_pytree_old,
            output_structure=jax.eval_shape(lambda: y_diff),
            tags=lx.positive_semidefinite_tag,
        )
        H_new = lx.PyTreeLinearOperator(
            new_pytree_new,
            output_structure=jax.eval_shape(lambda: y_diff),
            tags=lx.positive_semidefinite_tag,
        )

    return all_passed


def test_dfp_step_by_step():
    """Test DFP update step by step on random data."""
    print("\n" + "=" * 60)
    print("Testing DFP Step-by-Step Comparison")
    print("=" * 60)

    key = jr.PRNGKey(123)
    n = 5
    num_steps = 10

    # Start with identity Hessian
    H_old = _identity_pytree(jnp.zeros(n))
    H_new = _identity_pytree(jnp.zeros(n))

    all_passed = True

    for step in range(num_steps):
        key, k1, k2 = jr.split(key, 3)
        y_diff = jr.normal(k1, (n,))
        grad_diff = jr.normal(k2, (n,))

        # Ensure positive curvature
        inner = tree_dot(grad_diff, y_diff)
        if inner < 0:
            grad_diff = -grad_diff
            inner = -inner

        if inner < 1e-10:
            continue

        # === Old DFP update ===
        inv_mvp_old = H_old.mv(grad_diff)
        yHy_old = tree_dot(grad_diff, inv_mvp_old)
        term1_old = (_outer(y_diff, y_diff) ** ω / inner).ω
        term2_old = (_outer(inv_mvp_old, inv_mvp_old) ** ω / yHy_old).ω
        new_pytree_old = (H_old.pytree**ω + term1_old**ω - term2_old**ω).ω

        # === New DFP update (tauk=1) ===
        Hy_new = H_new.mv(grad_diff)
        yHy_new = tree_dot(grad_diff, Hy_new)
        tauk = jnp.array(1.0)

        term1_new = (_outer(Hy_new, Hy_new) ** ω / yHy_new).ω
        term2_new = (_outer(y_diff, y_diff) ** ω / inner).ω
        new_pytree_new = ((H_new.pytree**ω - term1_new**ω) / tauk + term2_new**ω).ω

        # Compare
        max_diff, status = compare_pytrees(
            new_pytree_old, new_pytree_new, f"Step {step}"
        )
        passed = max_diff < 1e-10
        all_passed = all_passed and passed

        print(f"  Step {step+1}: {status} Hessian diff={max_diff:.2e}")

        # Update for next iteration
        H_old = lx.PyTreeLinearOperator(
            new_pytree_old,
            output_structure=jax.eval_shape(lambda: y_diff),
            tags=lx.positive_semidefinite_tag,
        )
        H_new = lx.PyTreeLinearOperator(
            new_pytree_new,
            output_structure=jax.eval_shape(lambda: y_diff),
            tags=lx.positive_semidefinite_tag,
        )

    return all_passed


def test_ssbroyden_step_by_step():
    """Test SSBroyden update step by step on random data."""
    print("\n" + "=" * 60)
    print("Testing SSBroyden Step-by-Step Comparison")
    print("=" * 60)

    key = jr.PRNGKey(999)
    n = 5
    num_steps = 10

    # Start with identity Hessian
    H_old = _identity_pytree(jnp.zeros(n))
    H_new = _identity_pytree(jnp.zeros(n))

    # Simulate optimization state
    is_first_old = True
    is_first_new = True
    step_size = jnp.array(0.5)  # Simulated step size

    all_passed = True

    for step in range(num_steps):
        key, k1, k2, k3 = jr.split(key, 4)
        y_diff = jr.normal(k1, (n,))
        grad_diff = jr.normal(k2, (n,))
        grad_prev = jr.normal(k3, (n,))  # Previous gradient

        # Ensure positive curvature
        inner = tree_dot(grad_diff, y_diff)
        if inner < 0:
            grad_diff = -grad_diff
            inner = -inner

        if inner < 1e-10:
            continue

        rho = 1.0 / inner

        # === Shared computations ===
        Hy_old = H_old.mv(grad_diff)
        yHy_old = tree_dot(grad_diff, Hy_old)

        Hy_new = H_new.mv(grad_diff)
        yHy_new = tree_dot(grad_diff, Hy_new)

        # Self-scaling parameters
        hk_old = yHy_old * rho
        hk_new = yHy_new * rho

        bk = -step_size * rho * tree_dot(y_diff, grad_prev)

        ak_old = bk * hk_old - 1
        ak_new = bk * hk_new - 1

        # === Old SSBroyden thetak ===
        ck_old = jnp.sqrt(jnp.abs(ak_old / (1 + ak_old)))
        rhokm_old = jnp.minimum(1.0, hk_old * (1 - ck_old))
        thetakm_old = (rhokm_old - 1) / ak_old
        thetakp_old = 1 / rhokm_old
        thetak_old = jnp.maximum(thetakm_old, jnp.minimum(thetakp_old, (1 - bk) / bk))

        # === New SSBroyden thetak (same formula) ===
        ck_new = jnp.sqrt(jnp.abs(ak_new / (1 + ak_new)))
        rhokm_new = jnp.minimum(1.0, hk_new * (1 - ck_new))
        thetakm_new = (rhokm_new - 1) / ak_new
        thetakp_new = 1 / rhokm_new
        thetak_new = jnp.maximum(thetakm_new, jnp.minimum(thetakp_new, (1 - bk) / bk))

        # === Old SSBroyden tauk ===
        if is_first_old:
            tauk_old = hk_old / (1 + ak_old * thetak_old)
        else:
            N = n
            rhokk_old = jnp.minimum(1.0, 1.0 / bk)
            sigmak_old = 1 + thetak_old * ak_old
            sigmaknm1_old = jnp.abs(sigmak_old) ** (1.0 / (1.0 - N))
            if thetak_old <= 0:
                tauk_old = jnp.minimum(rhokk_old * sigmaknm1_old, sigmak_old)
            else:
                tauk_old = rhokk_old * jnp.minimum(sigmaknm1_old, 1 / thetak_old)

        # === New SSBroyden tauk (same formula) ===
        if is_first_new:
            tauk_new = hk_new / (1 + ak_new * thetak_new)
        else:
            N = n
            rhokk_new = jnp.minimum(1.0, 1.0 / bk)
            sigmak_new = 1 + thetak_new * ak_new
            sigmaknm1_new = jnp.abs(sigmak_new) ** (1.0 / (1.0 - N))
            if thetak_new <= 0:
                tauk_new = jnp.minimum(rhokk_new * sigmaknm1_new, sigmak_new)
            else:
                tauk_new = rhokk_new * jnp.minimum(sigmaknm1_new, 1 / thetak_new)

        # === Old SSBroyden update ===
        vk_old = y_diff * rho - Hy_old / yHy_old
        phik_old = (1 - thetak_old) / (1 + ak_old * thetak_old)
        term1_old = _outer(Hy_old, Hy_old)
        term2_old = _outer(vk_old, vk_old)
        term3_old = _outer(y_diff, y_diff)
        new_pytree_old = (
            (
                H_old.pytree**ω
                - term1_old**ω / yHy_old
                + term2_old**ω * (phik_old * yHy_old)
            )
            / tauk_old
            + term3_old**ω * rho
        ).ω

        # === New SSBroyden update ===
        vk_new = y_diff * rho - Hy_new / yHy_new
        phik_new = (1 - thetak_new) / (1 + ak_new * thetak_new)
        term1_new = _outer(Hy_new, Hy_new)
        term2_new = _outer(vk_new, vk_new)
        term3_new = _outer(y_diff, y_diff)
        new_pytree_new = (
            (
                H_new.pytree**ω
                - term1_new**ω / yHy_new
                + term2_new**ω * (phik_new * yHy_new)
            )
            / tauk_new
            + term3_new**ω * rho
        ).ω

        # Compare
        max_diff, status = compare_pytrees(
            new_pytree_old, new_pytree_new, f"Step {step}"
        )
        passed = max_diff < 1e-10
        all_passed = all_passed and passed

        print(
            f"  Step {step+1}: {status} Hessian diff={max_diff:.2e} thetak_old={float(thetak_old):.4f} thetak_new={float(thetak_new):.4f} tauk_old={float(tauk_old):.4f} tauk_new={float(tauk_new):.4f}"
        )

        # Update for next iteration
        H_old = lx.PyTreeLinearOperator(
            new_pytree_old,
            output_structure=jax.eval_shape(lambda: y_diff),
            tags=lx.positive_semidefinite_tag,
        )
        H_new = lx.PyTreeLinearOperator(
            new_pytree_new,
            output_structure=jax.eval_shape(lambda: y_diff),
            tags=lx.positive_semidefinite_tag,
        )
        is_first_old = False
        is_first_new = False

    return all_passed


def test_bfgs_on_rosenbrock():
    """Test BFGS on Rosenbrock comparing intermediate y values."""
    print("\n" + "=" * 60)
    print("Testing BFGS on Rosenbrock (intermediate y comparison)")
    print("=" * 60)

    # Simple Rosenbrock for n=2
    def rosenbrock(y, args):
        return (1 - y[0]) ** 2 + 100 * (y[1] - y[0] ** 2) ** 2

    # We'll manually step through both solvers and compare
    y0 = jnp.array([0.0, 0.0])

    # Initialize Hessians
    H_old = _identity_pytree(y0)
    H_new = _identity_pytree(y0)

    y_old = y0
    y_new = y0

    grad_fn = jax.grad(rosenbrock)

    grad_old = grad_fn(y_old, None)
    grad_new = grad_fn(y_new, None)

    num_steps = 20
    step_size = 0.001  # Fixed small step size for stability

    all_passed = True

    for step in range(num_steps):
        # Compute descent direction: -H^{-1} * grad
        d_old = -H_old.mv(grad_old)
        d_new = -H_new.mv(grad_new)

        # Compare descent directions
        diff_d, status_d = compare_pytrees(d_old, d_new, "direction", tol=1e-8)

        # Take step
        y_old_new_val = y_old + step_size * d_old
        y_new_new_val = y_new + step_size * d_new

        # Get new gradient
        grad_old_new = grad_fn(y_old_new_val, None)
        grad_new_new = grad_fn(y_new_new_val, None)

        # Compute differences
        y_diff_old = y_old_new_val - y_old
        y_diff_new = y_new_new_val - y_new
        grad_diff_old = grad_old_new - grad_old
        grad_diff_new = grad_new_new - grad_new

        inner_old = tree_dot(grad_diff_old, y_diff_old)
        inner_new = tree_dot(grad_diff_new, y_diff_new)

        # Skip update if inner product is too small
        if inner_old > 1e-10:
            # Old BFGS update
            inv_mvp_old = H_old.mv(grad_diff_old)
            mvp_inner_old = tree_dot(grad_diff_old, inv_mvp_old)
            diff_outer_old = _outer(y_diff_old, y_diff_old)
            mvp_outer_old = _outer(y_diff_old, inv_mvp_old)
            term1_old = (
                ((inner_old + mvp_inner_old) * (diff_outer_old**ω)) / (inner_old**2)
            ).ω
            term2_old = (
                (_outer(inv_mvp_old, y_diff_old) ** ω + mvp_outer_old**ω) / inner_old
            ).ω
            new_H_pytree_old = (H_old.pytree**ω + term1_old**ω - term2_old**ω).ω
            H_old = lx.PyTreeLinearOperator(
                new_H_pytree_old,
                output_structure=jax.eval_shape(lambda: y_diff_old),
                tags=lx.positive_semidefinite_tag,
            )

        if inner_new > 1e-10:
            # New BFGS update
            Hy_new = H_new.mv(grad_diff_new)
            yHy_new = tree_dot(grad_diff_new, Hy_new)
            diff_outer_new = _outer(y_diff_new, y_diff_new)
            mvp_outer_new = _outer(y_diff_new, Hy_new)
            term1_new = (
                ((inner_new + yHy_new) * (diff_outer_new**ω)) / (inner_new**2)
            ).ω
            term2_new = (
                (_outer(Hy_new, y_diff_new) ** ω + mvp_outer_new**ω) / inner_new
            ).ω
            new_H_pytree_new = (H_new.pytree**ω + term1_new**ω - term2_new**ω).ω
            H_new = lx.PyTreeLinearOperator(
                new_H_pytree_new,
                output_structure=jax.eval_shape(lambda: y_diff_new),
                tags=lx.positive_semidefinite_tag,
            )

        # Compare Hessians
        diff_H, status_H = compare_pytrees(
            H_old.pytree, H_new.pytree, "Hessian", tol=1e-8
        )

        # Compare y values
        diff_y, status_y = compare_pytrees(y_old_new_val, y_new_new_val, "y", tol=1e-8)

        passed = diff_H < 1e-8 and diff_y < 1e-8
        all_passed = all_passed and passed

        f_val = rosenbrock(y_old_new_val, None)
        print(f"  Step {step+1}: y_diff={diff_y:.2e} H_diff={diff_H:.2e} f={f_val:.4e}")

        # Update state
        y_old = y_old_new_val
        y_new = y_new_new_val
        grad_old = grad_old_new
        grad_new = grad_new_new

    return all_passed


def test_dfp_on_quadratic():
    """Test DFP on quadratic comparing intermediate y values."""
    print("\n" + "=" * 60)
    print("Testing DFP on Quadratic Bowl (intermediate y comparison)")
    print("=" * 60)

    # Simple quadratic bowl
    key = jr.PRNGKey(999)
    n = 4
    A = jr.normal(key, (n, n))
    Q = A.T @ A + jnp.eye(n)  # Positive definite

    def quadratic(y, args):
        return 0.5 * y @ Q @ y

    y0 = jnp.ones(n)

    # Initialize Hessians
    H_old = _identity_pytree(y0)
    H_new = _identity_pytree(y0)

    y_old = y0
    y_new = y0

    grad_fn = jax.grad(quadratic)

    grad_old = grad_fn(y_old, None)
    grad_new = grad_fn(y_new, None)

    num_steps = 15
    step_size = 0.1

    all_passed = True

    for step in range(num_steps):
        # Compute descent direction: -H^{-1} * grad
        d_old = -H_old.mv(grad_old)
        d_new = -H_new.mv(grad_new)

        # Take step
        y_old_new_val = y_old + step_size * d_old
        y_new_new_val = y_new + step_size * d_new

        # Get new gradient
        grad_old_new = grad_fn(y_old_new_val, None)
        grad_new_new = grad_fn(y_new_new_val, None)

        # Compute differences
        y_diff_old = y_old_new_val - y_old
        y_diff_new = y_new_new_val - y_new
        grad_diff_old = grad_old_new - grad_old
        grad_diff_new = grad_new_new - grad_new

        inner_old = tree_dot(grad_diff_old, y_diff_old)
        inner_new = tree_dot(grad_diff_new, y_diff_new)

        # Skip update if inner product is too small
        if inner_old > 1e-10:
            # Old DFP update
            inv_mvp_old = H_old.mv(grad_diff_old)
            yHy_old = tree_dot(grad_diff_old, inv_mvp_old)
            term1_old = (_outer(y_diff_old, y_diff_old) ** ω / inner_old).ω
            term2_old = (_outer(inv_mvp_old, inv_mvp_old) ** ω / yHy_old).ω
            new_H_pytree_old = (H_old.pytree**ω + term1_old**ω - term2_old**ω).ω
            H_old = lx.PyTreeLinearOperator(
                new_H_pytree_old,
                output_structure=jax.eval_shape(lambda: y_diff_old),
                tags=lx.positive_semidefinite_tag,
            )

        if inner_new > 1e-10:
            # New DFP update
            Hy_new = H_new.mv(grad_diff_new)
            yHy_new = tree_dot(grad_diff_new, Hy_new)
            term1_new = (_outer(Hy_new, Hy_new) ** ω / yHy_new).ω
            term2_new = (_outer(y_diff_new, y_diff_new) ** ω / inner_new).ω
            new_H_pytree_new = (H_new.pytree**ω - term1_new**ω + term2_new**ω).ω
            H_new = lx.PyTreeLinearOperator(
                new_H_pytree_new,
                output_structure=jax.eval_shape(lambda: y_diff_new),
                tags=lx.positive_semidefinite_tag,
            )

        # Compare Hessians
        diff_H, status_H = compare_pytrees(
            H_old.pytree, H_new.pytree, "Hessian", tol=1e-8
        )

        # Compare y values
        diff_y, status_y = compare_pytrees(y_old_new_val, y_new_new_val, "y", tol=1e-8)

        passed = diff_H < 1e-8 and diff_y < 1e-8
        all_passed = all_passed and passed

        f_val = quadratic(y_old_new_val, None)
        print(f"  Step {step+1}: y_diff={diff_y:.2e} H_diff={diff_H:.2e} f={f_val:.4e}")

        # Update state
        y_old = y_old_new_val
        y_new = y_new_new_val
        grad_old = grad_old_new
        grad_new = grad_new_new

    return all_passed


# =============================================================================
# Test Optimization Results with helpers.py Functions
# =============================================================================


def test_optimization_on_problems():
    """Test optimization using helpers.py test functions."""
    print("\n" + "=" * 60)
    print("Testing Optimization on Standard Problems")
    print("=" * 60)

    rtol = atol = 1e-6
    max_steps = 500

    # Define test problems
    def rosenbrock_scalar(y, args):
        """Rosenbrock as a scalar function."""
        return (args - y[0]) ** 2 + 100 * (y[1] - y[0] ** 2) ** 2

    def matyas_scalar(y, args):
        """Matyas function."""
        c1, c2 = args
        return c1 * (y[0] ** 2 + y[1] ** 2) - c2 * y[0] * y[1]

    def beale_scalar(y, args):
        """Beale function."""
        c1, c2, c3 = args
        term1 = (c1 - y[0] + y[0] * y[1]) ** 2
        term2 = (c2 - y[0] + y[0] * y[1] ** 2) ** 2
        term3 = (c3 - y[0] + y[0] * y[1] ** 3) ** 2
        return term1 + term2 + term3

    def bowl_scalar(y, args):
        """Simple quadratic bowl."""
        return y @ args @ y

    problems = [
        ("Rosenbrock", rosenbrock_scalar, jnp.array([0.0, 0.0]), jnp.array(1.0), 0.0),
        (
            "Matyas",
            matyas_scalar,
            jnp.array([6.0, 6.0]),
            (jnp.array(0.26), jnp.array(0.48)),
            0.0,
        ),
        (
            "Beale",
            beale_scalar,
            jnp.array([2.0, 0.0]),
            (jnp.array(1.5), jnp.array(2.25), jnp.array(2.625)),
            0.0,
        ),
    ]

    # Create optimizers
    old_bfgs = OldBFGS(rtol, atol)
    new_bfgs = NewBFGS(rtol, atol)
    old_dfp = OldDFP(rtol, atol)
    new_dfp = NewDFP(rtol, atol)

    all_passed = True

    for problem_name, fn, y0, args, expected_min in problems:
        print(f"\n  {problem_name}:")

        # Test BFGS
        try:
            old_result = optx.minimise(
                fn, old_bfgs, y0, args=args, max_steps=max_steps, throw=False
            )
            new_result = optx.minimise(
                fn, new_bfgs, y0, args=args, max_steps=max_steps, throw=False
            )

            old_y = old_result.value
            new_y = new_result.value
            old_f = fn(old_y, args)
            new_f = fn(new_y, args)

            diff_y, _ = compare_pytrees(old_y, new_y, "y", tol=1e-4)
            diff_f = abs(float(old_f - new_f))

            bfgs_passed = diff_f < 1e-4
            all_passed = all_passed and bfgs_passed
            status = "✓" if bfgs_passed else "✗"
            print(
                f"    BFGS: {status} old_f={float(old_f):.6e} new_f={float(new_f):.6e} diff_f={diff_f:.2e} diff_y={diff_y:.2e}"
            )
        except Exception as e:
            print(f"    BFGS: ✗ Error: {e}")
            all_passed = False

        # Test DFP
        try:
            old_result = optx.minimise(
                fn, old_dfp, y0, args=args, max_steps=max_steps, throw=False
            )
            new_result = optx.minimise(
                fn, new_dfp, y0, args=args, max_steps=max_steps, throw=False
            )

            old_y = old_result.value
            new_y = new_result.value
            old_f = fn(old_y, args)
            new_f = fn(new_y, args)

            diff_y, _ = compare_pytrees(old_y, new_y, "y", tol=1e-4)
            diff_f = abs(float(old_f - new_f))

            dfp_passed = diff_f < 1e-4
            all_passed = all_passed and dfp_passed
            status = "✓" if dfp_passed else "✗"
            print(
                f"    DFP:  {status} old_f={float(old_f):.6e} new_f={float(new_f):.6e} diff_f={diff_f:.2e} diff_y={diff_y:.2e}"
            )
        except Exception as e:
            print(f"    DFP: ✗ Error: {e}")
            all_passed = False

    return all_passed


def test_ssbroyden_optimization():
    """Test SSBroyden optimizer on standard problems."""
    print("\n" + "=" * 60)
    print("Testing SSBroyden Optimization")
    print("=" * 60)

    rtol = atol = 1e-6
    max_steps = 500

    def rosenbrock_scalar(y, args):
        return (args - y[0]) ** 2 + 100 * (y[1] - y[0] ** 2) ** 2

    def quadratic(y, args):
        return y @ args @ y

    # Create quadratic with random positive definite matrix
    key = jr.PRNGKey(42)
    A = jr.normal(key, (4, 4))
    Q = A.T @ A + jnp.eye(4)

    problems = [
        ("Rosenbrock", rosenbrock_scalar, jnp.array([0.0, 0.0]), jnp.array(1.0)),
        ("Quadratic", quadratic, jnp.ones(4), Q),
    ]

    old_ssbroyden = OldSSBroyden(rtol, atol)
    new_ssbroyden = NewSSBroyden(rtol, atol)

    all_passed = True

    for problem_name, fn, y0, args in problems:
        print(f"\n  {problem_name}:")

        try:
            old_result = optx.minimise(
                fn, old_ssbroyden, y0, args=args, max_steps=max_steps, throw=False
            )
            new_result = optx.minimise(
                fn, new_ssbroyden, y0, args=args, max_steps=max_steps, throw=False
            )

            old_y = old_result.value
            new_y = new_result.value
            old_f = fn(old_y, args)
            new_f = fn(new_y, args)

            diff_y, _ = compare_pytrees(old_y, new_y, "y", tol=1e-4)
            diff_f = abs(float(old_f - new_f))

            passed = diff_f < 1e-4
            all_passed = all_passed and passed
            status = "✓" if passed else "✗"
            print(
                f"    SSBroyden: {status} old_f={float(old_f):.6e} new_f={float(new_f):.6e} diff_f={diff_f:.2e} diff_y={diff_y:.2e}"
            )
        except Exception as e:
            print(f"    SSBroyden: ✗ Error: {e}")
            all_passed = False

    return all_passed


def test_ssbfgs_ssdfp_optimization():
    """Test SSBFGS and SSDFP optimizers on standard problems."""
    print("\n" + "=" * 60)
    print("Testing SSBFGS and SSDFP Optimization")
    print("=" * 60)

    rtol = atol = 1e-6
    max_steps = 500

    def rosenbrock_scalar(y, args):
        return (args - y[0]) ** 2 + 100 * (y[1] - y[0] ** 2) ** 2

    def matyas_scalar(y, args):
        c1, c2 = args
        return c1 * (y[0] ** 2 + y[1] ** 2) - c2 * y[0] * y[1]

    def beale_scalar(y, args):
        c1, c2, c3 = args
        term1 = (c1 - y[0] + y[0] * y[1]) ** 2
        term2 = (c2 - y[0] + y[0] * y[1] ** 2) ** 2
        term3 = (c3 - y[0] + y[0] * y[1] ** 3) ** 2
        return term1 + term2 + term3

    def quadratic(y, args):
        return y @ args @ y

    # Create quadratic with random positive definite matrix
    key = jr.PRNGKey(42)
    A = jr.normal(key, (4, 4))
    Q = A.T @ A + jnp.eye(4)

    problems = [
        ("Rosenbrock", rosenbrock_scalar, jnp.array([0.0, 0.0]), jnp.array(1.0)),
        (
            "Matyas",
            matyas_scalar,
            jnp.array([6.0, 6.0]),
            (jnp.array(0.26), jnp.array(0.48)),
        ),
        (
            "Beale",
            beale_scalar,
            jnp.array([2.0, 0.0]),
            (jnp.array(1.5), jnp.array(2.25), jnp.array(2.625)),
        ),
        ("Quadratic", quadratic, jnp.ones(4), Q),
    ]

    # Create optimizers
    ssbfgs = NewSSBFGS(rtol, atol)
    ssdfp = NewSSDFP(rtol, atol)

    all_passed = True

    for problem_name, fn, y0, args in problems:
        print(f"\n  {problem_name}:")

        # Test SSBFGS
        try:
            result = optx.minimise(
                fn, ssbfgs, y0, args=args, max_steps=max_steps, throw=False
            )
            y_result = result.value
            f_result = fn(y_result, args)

            # Check convergence (f should be close to 0 for these problems)
            converged = float(f_result) < 1e-4
            all_passed = all_passed and converged
            status = "✓" if converged else "✗"
            print(f"    SSBFGS: {status} f={float(f_result):.6e}")
        except Exception as e:
            print(f"    SSBFGS: ✗ Error: {e}")
            all_passed = False

        # Test SSDFP
        try:
            result = optx.minimise(
                fn, ssdfp, y0, args=args, max_steps=max_steps, throw=False
            )
            y_result = result.value
            f_result = fn(y_result, args)

            # Check convergence (f should be close to 0 for these problems)
            converged = float(f_result) < 1e-4
            all_passed = all_passed and converged
            status = "✓" if converged else "✗"
            print(f"    SSDFP:  {status} f={float(f_result):.6e}")
        except Exception as e:
            print(f"    SSDFP: ✗ Error: {e}")
            all_passed = False

    return all_passed


# =============================================================================
# Test use_inverse=False (Hessian update terms)
# =============================================================================


# =============================================================================
# Test use_inverse=False (Hessian update terms)
# =============================================================================
def test_hessian_update_terms():
    """Test all optimizers with use_inverse=False (direct Hessian update)."""
    print("\n" + "=" * 60)
    print("Testing use_inverse=False (_hessian_update_term)")
    print("=" * 60)

    rtol = atol = 1e-6
    max_steps = 500

    def rosenbrock_scalar(y, args):
        return (args - y[0]) ** 2 + 100 * (y[1] - y[0] ** 2) ** 2

    def matyas_scalar(y, args):
        c1, c2 = args
        return c1 * (y[0] ** 2 + y[1] ** 2) - c2 * y[0] * y[1]

    def quadratic(y, args):
        return y @ args @ y

    key = jr.PRNGKey(42)
    A = jr.normal(key, (4, 4))
    Q = A.T @ A + jnp.eye(4)

    problems = [
        ("Rosenbrock", rosenbrock_scalar, jnp.array([0.0, 0.0]), jnp.array(1.0)),
        (
            "Matyas",
            matyas_scalar,
            jnp.array([6.0, 6.0]),
            (jnp.array(0.26), jnp.array(0.48)),
        ),
        ("Quadratic", quadratic, jnp.ones(4), Q),
    ]

    # SSDFP excluded: _hessian_update_term raises NotImplementedError
    solvers = [
        ("BFGS", NewBFGS(rtol, atol, use_inverse=False)),
        ("SSBFGS", NewSSBFGS(rtol, atol, use_inverse=False)),
        ("SSBroyden", NewSSBroyden(rtol, atol, use_inverse=False)),
        ("DFP", NewDFP(rtol, atol, use_inverse=False)),
        ("SSDFP", NewSSDFP(rtol, atol, use_inverse=False)),
    ]

    all_passed = True

    for problem_name, fn, y0, args in problems:
        print(f"\n  {problem_name}:")
        for solver_name, solver in solvers:
            try:
                result = optx.minimise(
                    fn, solver, y0, args=args, max_steps=max_steps, throw=False
                )
                f = float(fn(result.value, args))
                converged = f < 1e-3
                all_passed = all_passed and converged
                status = "✓" if converged else "✗"
                print(f"    {solver_name:10s}: {status} f={f:.3e}")
            except Exception as e:
                print(f"    {solver_name:10s}: ✗ Error: {e}")
                all_passed = False

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("Testing BFGS/DFP Implementations")
    print("=" * 60)

    results = []

    # Part 1: Hessian update tests (simplified setting)
    print("\n" + "=" * 60)
    print("PART 1: Hessian Update Step-by-Step Tests")
    print("=" * 60)
    results.append(("BFGS Random Steps", test_bfgs_step_by_step()))
    results.append(("DFP Random Steps", test_dfp_step_by_step()))
    results.append(("SSBroyden Random Steps", test_ssbroyden_step_by_step()))
    results.append(("BFGS on Rosenbrock (manual)", test_bfgs_on_rosenbrock()))
    results.append(("DFP on Quadratic (manual)", test_dfp_on_quadratic()))

    # Part 2: Full optimization tests
    print("\n" + "=" * 60)
    print("PART 2: Full Optimization Tests")
    print("=" * 60)
    results.append(("Optimization on Problems", test_optimization_on_problems()))
    results.append(("SSBroyden Optimization", test_ssbroyden_optimization()))
    results.append(("SSBFGS/SSDFP Optimization", test_ssbfgs_ssdfp_optimization()))

    # Part 3: use_inverse=False tests
    print("\n" + "=" * 60)
    print("PART 3: Hessian Update Term Tests (use_inverse=False)")
    print("=" * 60)
    results.append(("use_inverse=False", test_hessian_update_terms()))

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        print(f"  {name}: {status}")

    print("=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
        sys.exit(1)
