"""fpgen: structured Kramers–Moyal / Fokker–Planck / Langevin toolkit.

Given a structured quantum master equation, fpgen derives the corresponding
Kramers–Moyal series in the Wigner, P, Q or positive-P phase-space
representation, truncates it to a Fokker–Planck equation, and converts that
to an Itô c-number Langevin equation on the augmented complex state.

Typical usage::

    from fpgen import boson_modes, MasterEquation, LindbladChannel, derive_kramers_moyal

    a, = boson_modes("a")
    master = MasterEquation(modes=[a], hamiltonian=..., channels=[...])
    km = derive_kramers_moyal(master, representation="positive-p")
    fpe = km.truncate(2)
    langevin = fpe.to_langevin()

The public API is the names listed in ``__all__``; everything else is an
internal implementation detail.  Version 0.2.0 removed the legacy
``opr2fp``/``to_fp_normal_form`` string-parsing API.
"""

__version__ = "0.2.0"

from fpgen.derive import derive_kramers_moyal
from fpgen.errors import (
    FPGenError,
    MasterEquationError,
    ModeError,
    ProbabilityConservationError,
    RepresentationError,
)
from fpgen.master import LindbladChannel, LiouvillianTerm, MasterEquation
from fpgen.operators import Op, boson_modes
from fpgen.results import (
    FokkerPlanckEquation,
    KramersMoyalSeries,
    LangevinEquation,
)

__all__ = [
    "boson_modes",
    "Op",
    "MasterEquation",
    "LindbladChannel",
    "LiouvillianTerm",
    "derive_kramers_moyal",
    "KramersMoyalSeries",
    "FokkerPlanckEquation",
    "LangevinEquation",
    "FPGenError",
    "RepresentationError",
    "ModeError",
    "MasterEquationError",
    "ProbabilityConservationError",
]
