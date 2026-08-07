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

__version__ = "0.5.1"

from fpgen.covariance import (
    AugmentedLangevinBlocks,
    AugmentedSecondMomentDynamics,
    CovarianceDynamics,
    HermitianCovarianceJacobian,
    IndependentSecondMomentDynamics,
    LinearReductionCandidate,
    LinearReductionPlan,
    MaterializationFailure,
    MaterializedReduction,
    MomentLayoutDecision,
    ReductionSearchResult,
    RejectedPartition,
)
from fpgen.derive import derive_kramers_moyal
from fpgen.errors import (
    CovarianceClosureError,
    FPGenError,
    MasterEquationError,
    ModeError,
    NumericalModelError,
    ProbabilityConservationError,
    ReductionError,
    RepresentationError,
)
from fpgen.master import LindbladChannel, LiouvillianTerm, MasterEquation
from fpgen.numerical import (
    AUGMENTED_SECOND_MOMENT_LAYOUT_VERSION,
    INDEPENDENT_SECOND_MOMENT_LAYOUT_VERSION,
    MODEL_SCHEMA_VERSION,
    MOMENT_DYNAMICS_API_VERSION,
    REDUCTION_API_VERSION,
    STATE_LAYOUT_VERSION,
    CompiledMomentDynamics,
    DerivationProvenance,
    MomentDynamicsSpec,
    NumericalModelSpec,
    ParameterSpec,
    SparsityPattern,
    StateVariableSpec,
)
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
    "CovarianceDynamics",
    "AugmentedLangevinBlocks",
    "AugmentedSecondMomentDynamics",
    "IndependentSecondMomentDynamics",
    "MomentLayoutDecision",
    "HermitianCovarianceJacobian",
    "LinearReductionCandidate",
    "ReductionSearchResult",
    "RejectedPartition",
    "LinearReductionPlan",
    "MaterializationFailure",
    "MaterializedReduction",
    "MODEL_SCHEMA_VERSION",
    "MOMENT_DYNAMICS_API_VERSION",
    "REDUCTION_API_VERSION",
    "STATE_LAYOUT_VERSION",
    "AUGMENTED_SECOND_MOMENT_LAYOUT_VERSION",
    "INDEPENDENT_SECOND_MOMENT_LAYOUT_VERSION",
    "StateVariableSpec",
    "ParameterSpec",
    "SparsityPattern",
    "DerivationProvenance",
    "NumericalModelSpec",
    "MomentDynamicsSpec",
    "CompiledMomentDynamics",
    "FPGenError",
    "RepresentationError",
    "ModeError",
    "MasterEquationError",
    "ProbabilityConservationError",
    "CovarianceClosureError",
    "NumericalModelError",
    "ReductionError",
]
