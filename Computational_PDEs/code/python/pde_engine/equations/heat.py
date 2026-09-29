"""
heat.py
=======

The 1D heat (diffusion) equation:

    u_t = alpha * u_xx + f(x, t),   x in [0, L], t > 0

with homogeneous Dirichlet boundary conditions u(0,t) = u(L,t) = 0 by
default, and a sinusoidal initial condition whose exact solution is
known in closed form -- this makes HeatEquation the natural starting
point for verifying any new solver against ground truth.
"""

import numpy as np
from .base import PDEBase


class HeatEquation(PDEBase):
    """
    1D heat equation u_t = alpha * u_xx + f(x, t).

    Parameters
    ----------
    alpha : float
        Thermal diffusivity (must be positive).
    length : float
        Length of the spatial domain [0, length].
    mode : int
        Which sine mode to use for the default initial condition,
        u(x,0) = sin(mode * pi * x / L). mode=1 (default) gives the
        simplest case with a known exact solution.
    source : callable, optional
        Custom source term f(x, t). If provided, the exact solution
        is no longer available (set to None) since the closed form
        derivation would need to change with the forcing.
    initial_condition_fn : callable, optional
        Custom initial condition u(x, 0). If provided, overrides the
        default sine mode and disables the exact solution as well,
        since the analytical solution below only holds for the sine
        initial condition.
    """

    def __init__(self, alpha: float = 0.01, length: float = 1.0,
                 mode: int = 1, source=None, initial_condition_fn=None):
        super().__init__(length=length, boundary_type="dirichlet")

        if alpha <= 0:
            raise ValueError("alpha (diffusivity) must be positive.")
        self.alpha = alpha
        self.mode = mode

        self._custom_source = source
        self._custom_ic = initial_condition_fn

        # Exact solution only holds for the default homogeneous,
        # sine-initial-condition case.
        self._exact_available = (source is None and initial_condition_fn is None)

    # --- Required interface --------------------------------------------

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        if self._custom_ic is not None:
            return self._custom_ic(x)
        return np.sin(self.mode * np.pi * x / self.length)

    def boundary_values(self, t: float):
        # Homogeneous Dirichlet: u(0,t) = u(L,t) = 0
        return 0.0, 0.0

    # --- Optional interface ----------------------------------------------

    def source_term(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._custom_source is not None:
            return self._custom_source(x, t)
        return np.zeros_like(x, dtype=float)

    def exact_solution(self, x: np.ndarray, t: float):
        if not self._exact_available:
            return None
        k = self.mode * np.pi / self.length
        return np.sin(k * x) * np.exp(-self.alpha * k**2 * t)

    def stability_limit(self, dx: float) -> float:
        """
        CFL-type stability bound for explicit (Forward Euler) time
        stepping applied to this equation: dt <= dx^2 / (2 * alpha).

        Returns the maximum stable dt for a given spatial step dx.
        Useful for experiments that deliberately probe stability.
        """
        return dx**2 / (2.0 * self.alpha)

    def __repr__(self):
        return (f"HeatEquation(alpha={self.alpha}, length={self.length}, "
                f"mode={self.mode})")
