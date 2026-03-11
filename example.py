# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: optimistix_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Imports

import time

# %%
from copy import deepcopy

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap

from optimistix_wrapper import run_optimization
from ssbrodyen_family import BFGS, SSBFGS, Broyden, SSBroyden, Zoom

# %%
# Configure JAX
dev = "gpu"
jax.config.update("jax_platform_name", dev)
jax.config.update("jax_enable_x64", True)

# Set random seed for reproducibility
seed = 1234

# %% [markdown]
# ## Utilities


# %%
def random_interior_points(key, intervals, N):
    """N uniformly drawn collocation points in a hyperrectangle defined by intervals."""
    intervals = jnp.array(intervals)
    l_bounds = intervals[:, 0]
    r_bounds = intervals[:, 1]
    dim = len(l_bounds)
    return random.uniform(
        key,
        shape=(N, dim),
        minval=jnp.broadcast_to(l_bounds, (N, dim)),
        maxval=jnp.broadcast_to(r_bounds, (N, dim)),
    )


def random_boundary_points(key, intervals, N):
    """N uniformly drawn collocation points on the boundary of a hyperrectangle."""
    intervals = jnp.array(intervals)
    l_bounds = intervals[:, 0]
    r_bounds = intervals[:, 1]
    dim = len(l_bounds)
    lengths = r_bounds - l_bounds
    total_length = jnp.sum(lengths)

    # Start with random interior points, then project one coordinate to a face
    x = random_interior_points(key, intervals, N)

    for i in range(N):
        key, key_loc, key_side = random.split(key, 3)

        # Pick which face dimension proportional to side length
        location = random.uniform(key_loc, (), minval=0.0, maxval=total_length)
        cumulative = jnp.cumsum(lengths)
        rand_dim = int(jnp.searchsorted(cumulative, location))
        rand_dim = min(rand_dim, dim - 1)

        # Pick left (0) or right (1) face
        side_idx = random.randint(key_side, shape=(), minval=0, maxval=2)
        rand_side = intervals[rand_dim, side_idx]

        x = x.at[i, rand_dim].set(rand_side)

    return x


def laplace(func, argnum=0):
    """Laplacian of func with respect to argument argnum (trace of Hessian)."""
    hesse = jax.hessian(func, argnums=argnum)
    return lambda *args, **kwargs: jnp.trace(hesse(*args, **kwargs), axis1=-2, axis2=-1)


class MLP(eqx.Module):
    layers: list

    def __init__(self, layer_sizes, *, key):
        keys = random.split(key, len(layer_sizes) - 1)
        self.layers = [
            eqx.nn.Linear(in_f, out_f, key=k)
            for in_f, out_f, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


# %% [markdown]
# ## Problem setup

# %%
# Number of dimensions
dim = 3

# Layer sizes
layer_sizes = [dim, 32, 32, 1]

# Collocation point counts
N_Omega = 5000
N_Gamma = 800
N_eval = 10 * N_Omega

# Domain intervals
intervals = [(0.0, 1.0) for _ in range(dim)]

# Collocation points
x_Omega = random_interior_points(random.PRNGKey(seed), intervals, N_Omega)
x_Gamma = random_boundary_points(random.PRNGKey(seed + 1), intervals, N_Gamma)
x_eval = random_interior_points(random.PRNGKey(seed + 2), intervals, N_eval)

# %%
# Create model
net = MLP(layer_sizes, key=random.PRNGKey(seed))
init_net = deepcopy(net)


# Functional interface: (params, x) -> output
# Split into (params, static) so we can pass params as a pytree to optimisers
@eqx.filter_jit
def model(params, x):
    net = eqx.combine(params, static)
    return net(x)


init_params, static = eqx.partition(net, eqx.is_array)
v_model = vmap(model, (None, 0))


# %%
# Solution and right-hand side
def u_star(x):
    return jnp.prod(jnp.sin(jnp.pi * x), keepdims=True)


def f(x):
    return dim * jnp.pi**2 * u_star(x)


# Residuals
interior_res = lambda params, x: laplace(model, argnum=1)(params, x) + f(x)
v_interior_res = vmap(interior_res, (None, 0))

boundary_res = lambda params, x: model(params, x) - u_star(x)
v_boundary_res = vmap(boundary_res, (None, 0))


# Loss
def interior_loss(params):
    return 0.5 * jnp.mean(v_interior_res(params, x_Omega) ** 2)


def boundary_loss(params):
    return 0.5 * jnp.mean(v_boundary_res(params, x_Gamma) ** 2)


def loss(params):
    return interior_loss(params) + boundary_loss(params)


# Errors
def compute_errors(params):
    """Compute relative L2 and H1 errors on evaluation points."""
    error = lambda x: model(params, x) - u_star(x)
    v_error = vmap(error)
    v_error_abs_grad = vmap(lambda x: jnp.sum(jax.jacrev(error)(x) ** 2.0) ** 0.5)

    l2_norm = lambda f: jnp.mean(f(x_eval) ** 2.0) ** 0.5

    # Reference norms for relative errors
    v_u_star = vmap(u_star)
    v_grad_u_star = vmap(lambda x: jnp.sum(jax.jacrev(u_star)(x) ** 2.0) ** 0.5)
    u_star_l2 = l2_norm(v_u_star)
    u_star_h1 = u_star_l2 + l2_norm(v_grad_u_star)

    l2_error = l2_norm(v_error) / u_star_l2
    h1_error = (l2_norm(v_error) + l2_norm(v_error_abs_grad)) / u_star_h1
    return l2_error, h1_error


# %% [markdown]
# ## Training

# %%
# Solver configs: (name, solver_instance)
solvers = {
    "BFGS": BFGS(
        rtol=1e-12, atol=1e-12, search=Zoom(initial_guess_strategy="increase")
    ),
    "SSBFGS": SSBFGS(
        rtol=1e-12, atol=1e-12, search=Zoom(initial_guess_strategy="increase")
    ),
    "Broyden": Broyden(
        rtol=1e-12, atol=1e-12, search=Zoom(initial_guess_strategy="increase")
    ),
    "SSBroyden": SSBroyden(
        rtol=1e-12, atol=1e-12, search=Zoom(initial_guess_strategy="increase")
    ),
}

maxiter = 10_000
log_every = 10

# {name: {"iters": [...], "l2": [...], "h1": [...], "loss": [...]}}
results = {}

for name, solver in solvers.items():
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    logs = {"iters": [], "l2": [], "h1": [], "loss": []}

    def callback(it, p, state, cur_loss, _logs=logs):
        if it % log_every == 0 or it == 1:
            l2, h1 = compute_errors(p)
            _logs["iters"].append(it)
            _logs["l2"].append(float(l2))
            _logs["h1"].append(float(h1))
            _logs["loss"].append(float(cur_loss))
        if it % (log_every * 100) == 0 or it == 1:
            print(
                f"  iter {it:5d} | loss {cur_loss:.6e} | L2 {float(l2):.6e} | H1 {float(h1):.6e}"
            )
        return False

    init = deepcopy(init_params)
    t0 = time.time()
    res = run_optimization(
        solver, loss, init, maxiter=maxiter, callback=callback, verbose=True
    )
    elapsed = time.time() - t0

    # Final errors
    l2_final, h1_final = compute_errors(res.params)
    logs["iters"].append(res.num_iters)
    logs["l2"].append(float(l2_final))
    logs["h1"].append(float(h1_final))
    logs["loss"].append(res.final_loss)

    results[name] = logs
    print(
        f"  Finished: {res.num_iters} iters, loss={res.final_loss:.6e}, "
        f"L2={float(l2_final):.6e}, H1={float(h1_final):.6e}, time={elapsed:.1f}s"
    )

# %%
plt.rcParams.update(
    {
        "font.size": 18,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for name, logs in results.items():
    axes[0].semilogy(logs["iters"], logs["loss"], label=name)
    axes[1].semilogy(logs["iters"], logs["l2"], label=name)
    axes[2].semilogy(logs["iters"], logs["h1"], label=name)

for ax, ylabel, title in zip(
    axes,
    ["Loss", r"Relative $\mathrm{L}^2$ error", r"Relative $\mathrm{H}^1$ error"],
    ["Loss", r"$\mathrm{L}^2$ Error Convergence", r"$\mathrm{H}^1$ Error Convergence"],
):
    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=22)
    ax.legend(fontsize=16)
    ax.tick_params(labelsize=16)
    ax.grid(True, which="both", ls="--", alpha=0.5)

fig.tight_layout()
fig.savefig("example_output.pdf", bbox_inches="tight")
fig.savefig("example_output.png", dpi=150, bbox_inches="tight")
plt.show()
