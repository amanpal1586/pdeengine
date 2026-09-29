"""
PDE Engine
==========

A unified framework for defining, solving, and comparing numerical
methods for Partial Differential Equations (finite differences, finite
elements, and related schemes).

Typical usage:

    from pde_engine.equations import HeatEquation
    from pde_engine.solvers import ExplicitFD

    problem = HeatEquation(alpha=0.01, length=1.0)
    solver = ExplicitFD(problem, dx=0.01, dt=0.0001)
    solution = solver.solve(t_final=1.0)
"""

__version__ = "1.0.0"
__author__ = "AMAN PAL"

# --- Public API -------------------------------------------------------
# Import order matters: equations and solvers are the core building
# blocks; error/benchmark/visualize depend on them. Keep these imports
# lightweight (no heavy computation at import time).

from . import equations
from . import solvers
from . import error
from . import benchmark
from . import visualize

__all__ = [
    "equations",
    "solvers",
    "error",
    "benchmark",
    "visualize",
]
