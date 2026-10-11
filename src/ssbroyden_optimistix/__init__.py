"""ssbroyden-optimistix: quasi-Newton solvers with a Zoom line search for Optimistix.

Provides BFGS, DFP, L-BFGS and the self-scaling Broyden family (SSBroyden, Broyden,
SSBFGS, SSDFP), with a strong-Wolfe "Zoom" line search by default, built on Optimistix'
public API.
"""

import os as _os

import jax as _jax

# Double precision by default (quasi-Newton updates for scientific problems are
# sensitive to rounding). Set SSBROYDEN_ENABLE_X64=0 before importing to keep JAX's
# default.
if _os.environ.get("SSBROYDEN_ENABLE_X64", "1") != "0":
    _jax.config.update("jax_enable_x64", True)

from ._lbfgs import AbstractLBFGS, LBFGS  # noqa: E402
from ._quasi_newton import AbstractBFGS, AbstractDFP, AbstractQuasiNewton, BFGS, DFP  # noqa: E402
from ._ssbroyden import (  # noqa: E402
    AbstractBroyden,
    AbstractSSBFGS,
    AbstractSSBroyden,
    AbstractSSBroydenFamily,
    AbstractSSDFP,
    Broyden,
    SSBFGS,
    SSBroyden,
    SSDFP,
)
from ._zoom import Zoom  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "AbstractBFGS",
    "AbstractBroyden",
    "AbstractDFP",
    "AbstractLBFGS",
    "AbstractQuasiNewton",
    "AbstractSSBFGS",
    "AbstractSSBroyden",
    "AbstractSSBroydenFamily",
    "AbstractSSDFP",
    "BFGS",
    "Broyden",
    "DFP",
    "LBFGS",
    "SSBFGS",
    "SSBroyden",
    "SSDFP",
    "Zoom",
]
