"""
base.py
=======

Abstract base class for all PDE problem definitions in the engine.

Every concrete equation (HeatEquation, WaveEquation, DiffusionEquation,
...) must implement this common interface so that solvers can treat
any PDE the same way:

    problem.initial_condition(x)
    problem.boundary_values(t)
    problem.source_term(x, t)      # optional, defaults to 0
    problem.exact_solution(x, t)   # optional, for validation only

Keeping this interface small and uniform is what lets a single solver
(ExplicitFD, CrankNicolson, IMEX, ...) work against any PDE defined
here without special-casing.
"""

from abc import ABC, abstractmethod
import numpy as np


class PDEBase(ABC):
    """
    Abstract base class for a 1D time-dependent PDE problem on [0, L].

    Parameters
    ----------
    length : float
        Length of the spatial domain [0, length].
    boundary_type : str
        'dirichlet' (default), 'neumann', or 'periodic'. Concrete
        classes may restrict which types they actually support.
    """

    def __init__(self, length: float = 1.0, boundary_type: str = "dirichlet"):
        if length <= 0:
            raise ValueError("Domain length must be positive.")
        self.length = length

        boundary_type = boundary_type.lower()
        if boundary_type not in ("dirichlet", "neumann", "periodic"):
            raise ValueError(
                "boundary_type must be 'dirichlet', 'neumann', or 'periodic'.")
        self.boundary_type = boundary_type

    # --- Required interface --------------------------------------------

    @abstractmethod
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """Return u(x, 0) for an array of spatial points x."""
        raise NotImplementedError

    @abstractmethod
    def boundary_values(self, t: float):
        """
        Return the boundary condition values at time t.

        For Dirichlet problems, returns a tuple (u_left, u_right) giving
        u(0, t) and u(L, t). For Neumann problems, returns
        (du/dx at 0, du/dx at L).
        """
        raise NotImplementedError

    # --- Optional interface (sensible defaults provided) ----------------

    def source_term(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Return the forcing/source term f(x, t), if any.

        Defaults to zero everywhere (homogeneous PDE). Override in a
        subclass or pass a custom source function to enable forcing.
        """
        return np.zeros_like(x, dtype=float)

    def exact_solution(self, x: np.ndarray, t: float):
        """
        Return the known analytical solution u(x, t), if one exists.

        Returns None by default. Provided by concrete classes only
        when a closed-form solution is available (used for verifying
        solver correctness and computing convergence rates).
        """
        return None

    def has_exact_solution(self) -> bool:
        """Whether this problem instance can provide ground truth."""
        return self.exact_solution(np.array([0.0]), 0.0) is not None

    # --- Shared utilities -------------------------------------------------

    def spatial_grid(self, n_points: int) -> np.ndarray:
        """Convenience helper: evenly spaced grid of n_points on [0, L]."""
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")
        return np.linspace(0.0, self.length, n_points)

    def __repr__(self):
        return (f"{self.__class__.__name__}(length={self.length}, "
                f"boundary_type='{self.boundary_type}')")
