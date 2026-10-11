import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
# Bitwise comparisons need reproducible CPU reductions (single-threaded Eigen).
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
# Results also depend on how many cores XLA's CPU thread pool sees, so pin the test
# process to one core before JAX starts (set SSBROYDEN_TEST_NO_PIN=1 to opt out).
if hasattr(os, "sched_setaffinity") and not os.environ.get("SSBROYDEN_TEST_NO_PIN"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
