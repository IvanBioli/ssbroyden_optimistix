"""ssbroyden_optimistix enables double precision on import (opt-out by env var)."""

import os
import subprocess
import sys


def _x64_after_import(**env):
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"} | env
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ssbroyden_optimistix, jax; print(jax.config.jax_enable_x64)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_import_enables_x64():
    assert _x64_after_import(JAX_PLATFORMS="cpu") == "True"


def test_opt_out():
    assert _x64_after_import(JAX_PLATFORMS="cpu", SSBROYDEN_ENABLE_X64="0") == "False"
