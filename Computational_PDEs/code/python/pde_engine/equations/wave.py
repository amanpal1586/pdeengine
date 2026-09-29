"""
wave.py
=======

The 1D wave equation:

    u_tt = c^2 * u_xx,   x in [0, L], t > 0

with homogeneous Dirichlet boundary conditions u(0,t) = u(L,t) = 0,
an initial displacement u(x,0), and an initial velocity u_t(x,0).

Because the wave equation is second order in time, it needs both an
initial displacement and an initial velocity -- unlike the heat
equation. Solvers must account for this (e.g. by tracking two state
arrays, or by converting to a first-order system).
"""

import numpy as np
from .base import PDEBase


class WaveEquation(PDEBase):
    """
    1D wave equation u_tt = c^2 * u_xx.

    Parameters
    ----------
    c : float
        Wave propagation speed (must be positive).
    length : float
        Length of the spatial domain [0, length].
    mode : int
        Which sine mode to use for the default initial displacement,
        u(x,0) = sin(mode * pi * x / L), with zero initial velocity.
        This combination has a known exact (standing wave) solution.
    initial_displacement_fn : callable, optional
        Custom u(x, 0). Overrides the default sine mode and disables
        the exact solution.
    initial_velocity_fn : callable, optional
        Custom u_t(x, 0). Defaults to zero (the string released from
        rest). Overriding this also disables the exact solution.
    """

    def __init__(self, c: float = 1.0, length: float = 1.0, mode: int = 1,
                 initial_displacement_fn=None, initial_velocity_fn=None):
        super().__init__(length=length, boundary_type="dirichlet")

        if c <= 0:
            raise ValueError("c (wave speed) must be positive.")
        self.c = c
        self.mode = mode

        self._custom_disp = initial_displacement_fn
        self._custom_vel = initial_velocity_fn

        # Exact standing-wave solution only holds for the default
        # sine displacement with zero initial velocity.
        self._exact_available = (initial_displacement_fn is None
                                  and initial_velocity_fn is None)

    # --- Required interface --------------------------------------------

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """Initial displacement u(x, 0)."""
        if self._custom_disp is not None:
            return self._custom_disp(x)
        return np.sin(self.mode * np.pi * x / self.length)

    def initial_velocity(self, x: np.ndarray) -> np.ndarray:
        """Initial velocity u_t(x, 0). Defaults to zero (at rest)."""
        if self._custom_vel is not None:
            return self._custom_vel(x)
        return np.zeros_like(x, dtype=float)

    def boundary_values(self, t: float):
        # Homogeneous Dirichlet: u(0,t) = u(L,t) = 0
        return 0.0, 0.0

    # --- Optional interface ----------------------------------------------

    def exact_solution(self, x: np.ndarray, t: float):
        if not self._exact_available:
            return None
        k = self.mode * np.pi / self.length
        return np.sin(k * x) * np.cos(self.c * k * t)

    def stability_limit(self, dx: float) -> float:
        """
        CFL stability bound for explicit central-difference time
        stepping applied to the wave equation: dt <= dx / c.

        Returns the maximum stable dt for a given spatial step dx.
        """
        return dx / self.c

    def __repr__(self):
        return f"WaveEquation(c={self.c}, length={self.length}, mode={self.mode})"
