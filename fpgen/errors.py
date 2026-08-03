"""Exception hierarchy for fpgen."""

from __future__ import annotations

__all__ = [
    "FPGenError",
    "RepresentationError",
    "ModeError",
    "MasterEquationError",
    "ProbabilityConservationError",
    "CovarianceClosureError",
]


class FPGenError(Exception):
    """Base class for all fpgen errors."""


class RepresentationError(FPGenError):
    """Unknown or unsupported phase-space representation."""


class ModeError(FPGenError):
    """An expression references a mode outside the declared mode order."""


class MasterEquationError(FPGenError):
    """Malformed master equation input."""


class ProbabilityConservationError(FPGenError):
    """The zero-order Kramers–Moyal term is nonzero.

    Raised when converting to a Langevin equation although the underlying
    master equation does not conserve probability (the Fokker–Planck
    equation contains a zeroth-order term).
    """


class CovarianceClosureError(FPGenError):
    """The Hermitian second-moment dynamics is not symbolically closed."""
