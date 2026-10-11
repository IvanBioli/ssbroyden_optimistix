# ssbroyden-optimistix

Quasi-Newton minimisers for [Optimistix](https://github.com/patrick-kidger/optimistix)
in [JAX](https://github.com/jax-ml/jax): BFGS, DFP, L-BFGS and the **self-scaling
Broyden family** (SSBroyden, Broyden, SSBFGS, SSDFP), all with a strong-Wolfe "Zoom"
line search by default. The package uses only Optimistix' public API and works with
Optimistix 0.0.11 and 0.1. Importing it switches JAX to double precision (set
`SSBROYDEN_ENABLE_X64=0` before the import to keep JAX's default).

## Installation

```bash
pip install ssbroyden-optimistix
# or the development version:
pip install git+https://github.com/IvanBioli/ssbroyden_optimistix.git
```

Install JAX first for your hardware (e.g. `pip install "jax[cuda13]"`); see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Quick start

```python
import jax
import jax.numpy as jnp
import optimistix as optx
import ssbroyden_optimistix as sso  # enables float64


def rosenbrock(y, args):
    return jnp.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2)


solver = sso.SSBroyden(rtol=1e-10, atol=1e-10)
sol = optx.minimise(rosenbrock, solver, jnp.zeros(10), max_steps=2000)
print(sol.value, sol.state.num_accepted_steps)
```

`optx.minimise` counts every evaluation of the line search as a step. To give a
budget of accepted iterations, or to log at each iteration, drive the solver with
`ssbroyden_optimistix.contrib.run(solver, loss_fn, params, maxiter, callback)`; see
[example.py](example.py) (a PINN for the 3D Poisson equation).

## Solvers

The self-scaling Broyden family updates the inverse Hessian approximation as

$$H_{k+1} = \frac{1}{\tau_k}\Big(H_k - \frac{H_k y_k y_k^\top H_k}{y_k^\top H_k y_k}
+ \phi_k\, (y_k^\top H_k y_k)\, v_k v_k^\top\Big) + \frac{s_k s_k^\top}{y_k^\top s_k},$$

where $\phi_k$ is determined by the family parameter $\theta_k$ (BFGS: $\theta_k = 0$,
DFP: $\theta_k = 1$) and $\tau_k$ scales the approximation.

| Solver | $\theta_k$ | $\tau_k$ | |
| --- | --- | --- | --- |
| `BFGS` | 0 | 1 | classic BFGS, with Zoom |
| `DFP` | 1 | 1 | classic DFP, with Zoom |
| `LBFGS` | — | — | limited-memory BFGS, with Zoom |
| `SSBFGS` | 0 | computed | self-scaling BFGS |
| `SSDFP` | 1 | computed | self-scaling DFP |
| `Broyden` | computed | 1 | Broyden family, unscaled |
| `SSBroyden` | computed | computed | self-scaling Broyden |

All of them take `search=` (default `Zoom()`), and all but `LBFGS` take
`use_inverse=` to approximate the inverse Hessian (default) or the Hessian. The family
also takes:

- `alpha`: the step length that enters the choice of $\theta_k$ and $\tau_k$.
  `"geometric"` (default) computes it from the step and the search direction, so it is
  exact for any line search; `"search"` uses the step size reported by the line
  search, which is the length of the accepted step only for searches that report it
  (such as `Zoom(initial_guess_strategy="keep")`, the default).
- `initial_scaling`: whether the first update uses the initial scaling
  $\tau = h_k / (1 + a_k \theta_k)$ (default `True`).

The class hierarchy (`AbstractQuasiNewton` → `AbstractSSBroydenFamily` →
`AbstractSSBroyden`, `AbstractBroyden`, `AbstractSSBFGS`, `AbstractSSDFP`) lets you
define other members of the family by overriding `_compute_thetak` and
`_compute_tauk`.

## Settings used in the paper

The quasi-Newton baselines of
[Bioli, Marcati, Sangalli (2025)](https://arxiv.org/abs/2505.11638) were run with the
Optimistix fork [IvanBioli/optimistix](https://github.com/IvanBioli/optimistix) (tag
`cmame-2026`). These constructions give bitwise the same iterates as the solvers of
the fork (checked by `tests/test_oracle_equivalence.py`):

```python
sso.BFGS(rtol=0.0, atol=0.0)                                     # optx.BFGS(0, 0, search=optx.Zoom())
sso.BFGS(rtol=0.0, atol=0.0, search=sso.Zoom(initial_guess_strategy="increase"))
sso.LBFGS(rtol=0.0, atol=0.0, history_length=500)
sso.SSBroyden(rtol=0.0, atol=0.0, alpha="search", initial_scaling=False)
```

## Changes since the technical note

The code described in [arXiv:2603.10599](https://arxiv.org/abs/2603.10599) (v1) is at
tag `arxiv-v1`. Since then:

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

## Tests

```bash
pip install -e ".[test]"
JAX_ENABLE_X64=1 pytest tests
```

The oracle tests need a checkout of the fork at tag `cmame-2026`; point
`OPTX_ORACLE_PATH` to its inner `optimistix/` package directory (they are skipped
otherwise).

## Credits

- The Hessian-form updates of the family and the first convergence tests are by
  Mikel Mendibe; the rest by Ivan Bioli.
- The Zoom line search is adapted from
  [bagibence/optimistix@zoom_linesearch](https://github.com/bagibence/optimistix/tree/zoom_linesearch),
  and the quasi-Newton step and L-BFGS from Optimistix (both Apache-2.0).

## License

Apache-2.0, see [LICENSE](LICENSE).

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
