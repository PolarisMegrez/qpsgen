"""Exception hierarchy for qpsgen."""

from __future__ import annotations

__all__ = [
    "QPSGenError",
    "RepresentationError",
    "ModeError",
    "MasterEquationError",
    "ProbabilityConservationError",
    "CovarianceClosureError",
    "NumericalModelError",
    "ReductionError",
]


class QPSGenError(Exception):
    """Base class for all qpsgen errors."""


class RepresentationError(QPSGenError):
    """Unknown or unsupported phase-space representation."""


class ModeError(QPSGenError):
    """An expression references a mode outside the declared mode order."""


class MasterEquationError(QPSGenError):
    """Malformed master equation input."""


class ProbabilityConservationError(QPSGenError):
    """The zero-order Kramers–Moyal term is nonzero.

    Raised when converting to a Langevin equation although the underlying
    master equation does not conserve probability (the Fokker–Planck
    equation contains a zeroth-order term).
    """


class CovarianceClosureError(QPSGenError):
    """A requested second-moment layout or closure is not symbolically valid."""


class NumericalModelError(QPSGenError):
    """A symbolic dynamics object cannot satisfy the numerical API contract."""


class ReductionError(QPSGenError):
    """A requested structural reduction is invalid or cannot be materialized."""
