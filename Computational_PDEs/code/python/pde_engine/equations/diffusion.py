"""
diffusion.py
============

The 1D advection-diffusion equation:

    u_t + v * u_x = D * u_xx + f(x, t),   x in [0, L], t > 0

with periodic boundary conditions by default. This generalizes the
pure HeatEquation (which is the D-only, v=0 case) by adding an
advection (transport) term -- useful for testing how well a scheme
handles combined transport + diffusion, which is where many schemes
(e.g. central-difference advection terms) can develop spurious
oscillations if not handled carefully.

Periodic boundaries are used here (rather than Dirichlet) because they
admit a simple, exact traveling-wave solution for verification:

    u(x,t) = sin(k * (x - v*t)) * exp(-D * k^2 * t)

which is an exact solution of the PDE above with f = 0, for any wave
number k = mode * 2*pi / L (2*pi, not pi, since periodic BCs require a
full-period sine wave, not a half-period one).
"""

import numpy as np
from .base import PDEBase


class DiffusionEquation(PDEBase):
    """
    1D advection-diffusion equation u_t + v*u_x = D*u_xx + f(x, t).

    Parameters
    ----------
    D : float
        Diffusion coefficient (must be positive).
    v : float
        Advection (transport) velocity. Defaults to 0, which reduces
        this to pure diffusion (equivalent in behaviour to
        HeatEquation, but with periodic rather than Dirichlet BCs).
    length : float
        Length of the spatial domain [0, length].
    mode : int
        Which periodic sine mode to use for the default initial
        condition, u(x,0) = sin(mode * 2*pi * x / L).
    source : callable, optional
        Custom source term f(x, t). If provided, disables the exact
        solution.
    initial_condition_fn : callable, optional
        Custom initial condition u(x, 0). If provided, disables the
        exact solution.
    """

    def __init__(self, D: float = 0.01, v: float = 0.0, length: float = 1.0,
                 mode: int = 1, source=None, initial_condition_fn=None):
        super().__init__(length=length, boundary_type="periodic")

        if D <= 0:
            raise ValueError("D (diffusion coefficient) must be positive.")
        self.D = D
        self.v = v
        self.mode = mode

        self._custom_source = source
        self._custom_ic = initial_condition_fn

        self._exact_available = (source is None and initial_condition_fn is None)

    # --- Required interface --------------------------------------------

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        if self._custom_ic is not None:
            return self._custom_ic(x)
        k = self.mode * 2.0 * np.pi / self.length
        return np.sin(k * x)

    def boundary_values(self, t: float):
        """
        Periodic BCs don't use fixed edge values; solvers should wrap
        indices instead. Returned here only for interface uniformity.
        """
        return None, None

    # --- Optional interface ----------------------------------------------

    def source_term(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._custom_source is not None:
            return self._custom_source(x, t)
        return np.zeros_like(x, dtype=float)

    def exact_solution(self, x: np.ndarray, t: float):
        if not self._exact_available:
            return None
        k = self.mode * 2.0 * np.pi / self.length
        return np.sin(k * (x - self.v * t)) * np.exp(-self.D * k**2 * t)

    def peclet_number(self, dx: float) -> float:
        """
        Grid (cell) Peclet number Pe = v * dx / D, a standard
        diagnostic for advection-diffusion schemes: Pe > 2 typically
        signals that a central-difference advection term will produce
        non-physical oscillations, and an upwind scheme is preferred.
        """
        if self.D == 0:
            return np.inf
        return abs(self.v) * dx / self.D

    def __repr__(self):
        return (f"DiffusionEquation(D={self.D}, v={self.v}, "
                f"length={self.length}, mode={self.mode})")
