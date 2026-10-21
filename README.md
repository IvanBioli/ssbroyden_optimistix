# ssbroyden-optimistix

**Self-scaled Broyden, BFGS, DFP and L-BFGS minimisers for JAX, ready to drop into [Optimistix](https://github.com/patrick-kidger/optimistix).**

[![PyPI](https://img.shields.io/pypi/v/ssbroyden-optimistix?color=0B1A33)](https://pypi.org/project/ssbroyden-optimistix/)
[![Python](https://img.shields.io/pypi/pyversions/ssbroyden-optimistix?color=0B1A33)](https://pypi.org/project/ssbroyden-optimistix/)
[![Tests](https://github.com/IvanBioli/ssbroyden_optimistix/actions/workflows/tests.yml/badge.svg)](https://github.com/IvanBioli/ssbroyden_optimistix/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-0B1A33)](https://github.com/IvanBioli/ssbroyden_optimistix/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2603.10599-b31b1b.svg)](https://arxiv.org/abs/2603.10599)

```bash
pip install ssbroyden-optimistix
```

`ssbroyden-optimistix` gives you quasi-Newton optimisers for smooth, deterministic
(full-batch) problems in JAX: the **self-scaled Broyden family** (`SSBroyden`,
`SSBFGS`, `SSDFP`, `Broyden`) together with `BFGS`, `DFP` and `LBFGS`, all with a
strong-Wolfe **Zoom line search**. Every solver is a regular `optimistix` minimiser:
pass it to `optx.minimise`, use it on any pytree of parameters (arrays, Equinox
models, ...), and `jit`/`vmap` through it as usual.

Use it when you need to push a loss to **high accuracy**, and first-order methods
(Adam & co.) stall: training neural networks for PDEs and surrogate models, curve
fitting and parameter estimation, calibration of physical models. On these problems
the self-scaled updates often reach a far lower loss than plain BFGS for the same
number of iterations:

![Loss and error of BFGS, SSBFGS, Broyden and SSBroyden on a neural network solving the 3D Poisson equation](https://raw.githubusercontent.com/IvanBioli/ssbroyden_optimistix/main/example_output.png)

<sub>Neural network solving the 3D Poisson equation, 10,000 iterations; reproduce with
[`example.py`](https://github.com/IvanBioli/ssbroyden_optimistix/blob/main/example.py).</sub>

## Quick start

```python
import jax.numpy as jnp
import optimistix as optx
import ssbroyden_optimistix as sso  # note: enables float64 in JAX


def rosenbrock(y, args):
    return jnp.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2)


solver = sso.SSBroyden(rtol=1e-10, atol=1e-10)
sol = optx.minimise(rosenbrock, solver, jnp.zeros(10), max_steps=2000)
print(sol.value)  # [1. 1. 1. ... 1.]
```

Switching optimiser is a one-word change: `sso.BFGS(...)`, `sso.LBFGS(...)`,
`sso.SSBFGS(...)`, ...

### Training a model

The solvers work on any pytree. With an Equinox model, split the trainable arrays from
the rest and optimise those:

```python
import equinox as eqx

params, static = eqx.partition(model, eqx.is_array)

def loss(params, args):
    model = eqx.combine(params, static)
    return jnp.mean((jax.vmap(model)(x) - y) ** 2)

sol = optx.minimise(loss, sso.SSBroyden(rtol=1e-12, atol=1e-12), params,
                    max_steps=10_000, throw=False)
model = eqx.combine(sol.value, static)
```

`optx.minimise` counts every line-search evaluation as a step and raises an error
when `max_steps` is reached (unless `throw=False`). For a training loop with a budget
of **accepted iterations**, logging and early stopping, use the helper
`ssbroyden_optimistix.contrib.run`:

```python
from ssbroyden_optimistix.contrib import run

def callback(it, params, state, loss):
    if it % 100 == 0:
        print(f"iter {it:5d}  loss {loss:.3e}")
    return False  # return True to stop

res = run(sso.SSBroyden(rtol=1e-12, atol=1e-12), lambda p: loss(p, None), params,
          maxiter=10_000, callback=callback)
res.params, res.num_iters, res.final_loss
```

[`example.py`](https://github.com/IvanBioli/ssbroyden_optimistix/blob/main/example.py)
is a complete script (neural network for the 3D Poisson equation, four solvers
compared, the plot above).

## Which solver should I use?

| Solver | Use it for |
| --- | --- |
| `SSBroyden` | **Recommended default.** Self-scaled Broyden: θ and τ chosen at every step. |
| `SSBFGS` | Self-scaled BFGS (θ = 0); a robust alternative to `SSBroyden`. |
| `BFGS` | Classic BFGS with a Zoom line search, as a baseline. |
| `LBFGS` | Large problems: stores `history_length` vector pairs instead of an n × n matrix. |
| `SSDFP`, `DFP`, `Broyden` | Other members of the family, mainly for comparisons. |

`BFGS`, `DFP` and the self-scaled family store a dense n × n matrix, so they suit
problems with up to tens of thousands of parameters; beyond that, use `LBFGS`.

### Options

All solvers take `rtol`, `atol` and `search=` (default `sso.Zoom()`); all but `LBFGS`
take `use_inverse=` to approximate the inverse Hessian (default) or the Hessian. The
self-scaled family also takes:

- `alpha`: how the step length entering the update is computed. `"geometric"`
  (default) is exact for any line search; `"search"` uses the step size reported by
  the line search.
- `initial_scaling`: scale the first update with `τ = h / (1 + aθ)` (default `True`).

`sso.Zoom(initial_guess_strategy="keep" | "increase")` controls the first trial step
of each line search.

### Precision

Importing the package switches JAX to **double precision**, because quasi-Newton
updates on ill-conditioned problems are sensitive to rounding. To keep JAX's default,
set the environment variable `SSBROYDEN_ENABLE_X64=0` before the import.

## How it works

After a step `s = x_new - x` with gradient change `y = g_new - g`, the self-scaled
Broyden family updates the inverse-Hessian approximation `H` as

```text
H_new = (H - H y y^T H / (y^T H y) + φ (y^T H y) v v^T) / τ + s s^T / (y^T s)

v = s / (y^T s) - H y / (y^T H y)
φ = (1 - θ) / (1 + a θ),   a = b h - 1
h = y^T H y / (y^T s),     b = s^T B s / (y^T s),   B = H^(-1)
```

θ interpolates between BFGS (θ = 0) and DFP (θ = 1), and τ rescales the
approximation. The choices of θ and τ, and the derivation, are in the
[technical note](https://arxiv.org/abs/2603.10599).

| Solver | θ | τ |
| --- | --- | --- |
| `BFGS` | 0 | 1 |
| `DFP` | 1 | 1 |
| `SSBFGS` | 0 | computed |
| `SSDFP` | 1 | computed |
| `Broyden` | computed | 1 |
| `SSBroyden` | computed | computed |

**Your own variant.** The class hierarchy (`AbstractQuasiNewton` →
`AbstractSSBroydenFamily` → `AbstractSSBroyden`, `AbstractBroyden`, `AbstractSSBFGS`,
`AbstractSSDFP`) lets you define other members of the family by overriding
`_compute_thetak` and `_compute_tauk`.

## Compatibility

Python ≥ 3.11, Optimistix 0.0.11 and 0.1. The package uses only Optimistix' public
API, so no fork is needed. Install JAX for your hardware first (e.g.
`pip install "jax[cuda13]"`); see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

Development version: `pip install git+https://github.com/IvanBioli/ssbroyden_optimistix.git`

## Contributing and tests

Issues and pull requests are welcome. To run the tests:

```bash
pip install -e ".[test]"
JAX_ENABLE_X64=1 pytest tests
```

The oracle tests need a checkout of the fork used in the paper (below) at tag
`cmame-2026`; point `OPTX_ORACLE_PATH` to its inner `optimistix/` package directory
(they are skipped otherwise).

## Citation

If you use this package, please cite the technical note:

```bibtex
@misc{bioli2026selfscaledbroydenfamilyquasinewton,
  title         = {Self-Scaled Broyden Family of Quasi-Newton Methods in {JAX}},
  author        = {Bioli, Ivan and Mendibe Abarrategi, Mikel},
  year          = {2026},
  eprint        = {2603.10599},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MS},
  url           = {https://arxiv.org/abs/2603.10599}
}
```

The code is archived on Zenodo: version 0.1.0 is
[doi:10.5281/zenodo.23297645](https://doi.org/10.5281/zenodo.23297645); all versions:
[doi:10.5281/zenodo.23297644](https://doi.org/10.5281/zenodo.23297644).

<details>
<summary><b>Reproducing published results</b></summary>

**Settings of Bioli, Marcati, Sangalli (2025).** The quasi-Newton baselines of
[arXiv:2505.11638](https://arxiv.org/abs/2505.11638) were run with the Optimistix fork
[IvanBioli/optimistix](https://github.com/IvanBioli/optimistix) (tag `cmame-2026`).
These constructions give bitwise the same iterates (checked by
`tests/test_oracle_equivalence.py`):

```python
sso.BFGS(rtol=0.0, atol=0.0)                                     # optx.BFGS(0, 0, search=optx.Zoom())
sso.BFGS(rtol=0.0, atol=0.0, search=sso.Zoom(initial_guess_strategy="increase"))
sso.LBFGS(rtol=0.0, atol=0.0, history_length=500)
sso.SSBroyden(rtol=0.0, atol=0.0, alpha="search", initial_scaling=False)
```

**Changes since the technical note.** The code described in
[arXiv:2603.10599](https://arxiv.org/abs/2603.10599) (v1) is at tag `arxiv-v1`. Since
then:

- the package no longer needs a fork of Optimistix: it builds only on Optimistix'
  public API (0.0.11 and 0.1) and ships the Zoom line search;
- `ssbroyden_optimistix.contrib.run` replaces `optimistix_wrapper.py` to count
  quasi-Newton iterations separately from line search steps;
- `LBFGS` with Zoom, and the options `alpha` and `initial_scaling` of the family, are
  new;
- by default, the step length in the H-form updates is now computed from the step
  (`alpha="geometric"`), and the first update is scaled (`initial_scaling=True`). In
  v1 it was the step size reported by the line search, which is the accepted step
  length only with `Zoom(initial_guess_strategy="keep")`: with `"increase"`, as in
  `example.py`, it is 1.5 times the accepted step. Results of the self-scaled
  methods in the H-form therefore differ from v1; `alpha="search",
  initial_scaling=False` gives the v1 updates.

</details>

## Credits

- The Hessian-form updates of the family and the first convergence tests are by
  Mikel Mendibe; the rest by Ivan Bioli.
- The Zoom line search is adapted from
  [bagibence/optimistix@zoom_linesearch](https://github.com/bagibence/optimistix/tree/zoom_linesearch),
  and the quasi-Newton step and L-BFGS from Optimistix (both Apache-2.0).

## License

Apache-2.0, see [LICENSE](https://github.com/IvanBioli/ssbroyden_optimistix/blob/main/LICENSE).
