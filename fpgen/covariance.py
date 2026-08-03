"""Closed second-moment dynamics and structural reductions.

For a c-number Langevin equation ``dot(alpha) = -I H alpha + xi``, the
normal moment ``R_ij = <alpha_i alpha_j^*>`` obeys

    dot(R) = -I (H(R) R - R H(R)^dagger) + D(R)

when the selected closure maps every phase-space dependence in ``H`` and
``D`` to the retained moments. No Gaussian/Wick closure is applied implicitly.

Linear augmented drifts can additionally be closed on the combined normal and
anomalous moments ``R=<alpha alpha^dagger>`` and ``C=<alpha alpha^T>``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from itertools import combinations

import sympy as sp

from fpgen.errors import (
    CovarianceClosureError,
    NumericalModelError,
    ReductionError,
)
from fpgen.numerical import (
    AUGMENTED_SECOND_MOMENT_LAYOUT_VERSION,
    INDEPENDENT_SECOND_MOMENT_LAYOUT_VERSION,
    MODEL_SCHEMA_VERSION,
    STATE_LAYOUT_VERSION,
    DerivationProvenance,
    MomentDynamicsSpec,
    ParameterSpec,
    SparsityPattern,
    StateVariableSpec,
    make_fingerprint,
    matrix_sparsity,
)
from fpgen.representations import conjugate_variable_map

__all__ = [
    "AugmentedLangevinBlocks",
    "MomentLayoutDecision",
    "CovarianceDynamics",
    "AugmentedSecondMomentDynamics",
    "IndependentSecondMomentDynamics",
    "HermitianCovarianceJacobian",
    "LinearReductionCandidate",
    "LinearReductionPlan",
    "MaterializedReduction",
]


def _simplified_matrix(matrix) -> sp.Matrix:
    return sp.Matrix(matrix).applyfunc(sp.simplify)


def _display_coordinate(kind: str, mode_i: str, mode_j: str) -> sp.Symbol:
    base = f"R_{{{mode_i}{mode_j}}}"
    name = base if kind == "diag" else rf"\operatorname{{{kind}}}({base})"
    return sp.Symbol(name, real=True)


def _hermitian_state(mode_names: tuple[str, ...]):
    """Build a machine-safe Hermitian state in declared-index order."""
    n = len(mode_names)
    pair_indices = tuple(combinations(range(n), 2))
    specs: list[StateVariableSpec] = []

    for i, mode in enumerate(mode_names):
        index = len(specs)
        specs.append(StateVariableSpec(
            id=f"r_diag_{i}",
            index=index,
            symbol=sp.Symbol(f"x{index}", real=True),
            latex_name=f"R_{{{mode}{mode}}}",
            kind="diag",
            matrix_index=(i, i),
            mode_names=(mode, mode),
        ))
    for kind, prefix, latex_kind in (
        ("offdiag_re", "r_re", "Re"),
        ("offdiag_im", "r_im", "Im"),
    ):
        for i, j in pair_indices:
            index = len(specs)
            left, right = mode_names[i], mode_names[j]
            specs.append(StateVariableSpec(
                id=f"{prefix}_{i}_{j}",
                index=index,
                symbol=sp.Symbol(f"x{index}", real=True),
                latex_name=rf"\operatorname{{{latex_kind}}}(R_{{{left}{right}}})",
                kind=kind,
                matrix_index=(i, j),
                mode_names=(left, right),
            ))

    R = sp.zeros(n)
    for item in specs[:n]:
        i, _ = item.matrix_index
        R[i, i] = item.symbol
    pair_count = len(pair_indices)
    real_specs = specs[n:n + pair_count]
    imag_specs = specs[n + pair_count:]
    for (i, j), real_item, imag_item in zip(
        pair_indices, real_specs, imag_specs, strict=True
    ):
        R[i, j] = real_item.symbol + sp.I * imag_item.symbol
        R[j, i] = real_item.symbol - sp.I * imag_item.symbol

    pair_order = tuple((mode_names[i], mode_names[j]) for i, j in pair_indices)
    return R, tuple(specs), pair_indices, pair_order


def _formal_conjugate(expr: sp.Expr, variables, real_symbols=()) -> sp.Expr:
    """Conjugate an expression and map each phase variable to its partner."""
    partner_map = conjugate_variable_map(tuple(variables))
    replacements = {
        sp.conjugate(variable): partner
        for variable, partner in partner_map.items()
    }
    replacements.update({
        sp.conjugate(symbol): symbol for symbol in real_symbols
    })
    return sp.simplify(sp.conjugate(expr).xreplace(replacements))


def _formal_conjugate_matrix(matrix, variables, real_symbols=()) -> sp.Matrix:
    return sp.Matrix(matrix).applyfunc(
        lambda entry: _formal_conjugate(entry, variables, real_symbols)
    )


def _exchange_matrix(n: int) -> sp.Matrix:
    return sp.zeros(n).row_join(sp.eye(n)).col_join(
        sp.eye(n).row_join(sp.zeros(n))
    )


def _augmented_second_moment_state(mode_names: tuple[str, ...]):
    """Build Hermitian ``R_tilde=<z z^dagger>`` from normal ``R`` and ``C``."""
    R, normal_specs, pair_indices, pair_order = _hermitian_state(mode_names)
    n = len(mode_names)
    symmetric_indices = tuple(
        (i, j) for i in range(n) for j in range(i, n)
    )
    specs = list(normal_specs)
    real_anomalous = []
    imag_anomalous = []
    for kind, prefix, latex_kind, target in (
        ("anomalous_re", "c_re", "Re", real_anomalous),
        ("anomalous_im", "c_im", "Im", imag_anomalous),
    ):
        for i, j in symmetric_indices:
            index = len(specs)
            left, right = mode_names[i], mode_names[j]
            item = StateVariableSpec(
                id=f"{prefix}_{i}_{j}",
                index=index,
                symbol=sp.Symbol(f"x{index}", real=True),
                latex_name=rf"\operatorname{{{latex_kind}}}(C_{{{left}{right}}})",
                kind=kind,
                matrix_index=(i, j),
                mode_names=(left, right),
            )
            specs.append(item)
            target.append(item)

    C = sp.zeros(n)
    for (i, j), real_item, imag_item in zip(
        symmetric_indices, real_anomalous, imag_anomalous, strict=True
    ):
        value = real_item.symbol + sp.I * imag_item.symbol
        C[i, j] = value
        C[j, i] = value
    augmented = R.row_join(C).col_join(
        C.conjugate().row_join(R.conjugate())
    )
    return (
        R,
        C,
        augmented,
        tuple(specs),
        pair_indices,
        pair_order,
        symmetric_indices,
    )


def _close_augmented_bilinears(
    expr: sp.Expr,
    phase_variables: tuple[sp.Symbol, ...],
    normal_moment: sp.Matrix,
    anomalous_moment: sp.Matrix,
) -> sp.Expr:
    """Map every quadratic alpha/abar monomial to an ``R`` or ``C`` entry."""
    expr = sp.expand(expr)
    try:
        polynomial = sp.Poly(expr, *phase_variables)
    except sp.PolynomialError as exc:
        raise CovarianceClosureError(
            f"Expression {expr} is not polynomial in the phase-space variables."
        ) from exc

    n = len(phase_variables) // 2
    closed = sp.Integer(0)
    for powers, coefficient in polynomial.terms():
        degree = sum(powers)
        if degree == 0:
            closed += coefficient
            continue
        if degree != 2:
            raise CovarianceClosureError(
                "Second-moment substitution accepts only constants and "
                f"quadratic phase-space monomials; found powers {powers} in {expr}."
            )

        alpha_degree = sum(powers[:n])
        partner_degree = sum(powers[n:])
        if alpha_degree == 2:
            axes = [i for i, power in enumerate(powers[:n]) for _ in range(power)]
            closed += coefficient * anomalous_moment[axes[0], axes[1]]
        elif partner_degree == 2:
            axes = [
                i for i, power in enumerate(powers[n:]) for _ in range(power)
            ]
            closed += coefficient * sp.conjugate(
                anomalous_moment[axes[0], axes[1]]
            )
        elif alpha_degree == partner_degree == 1:
            i = next(i for i, power in enumerate(powers[:n]) if power)
            j = next(i for i, power in enumerate(powers[n:]) if power)
            closed += coefficient * normal_moment[i, j]
        else:  # pragma: no cover - degree accounting makes this unreachable
            raise CovarianceClosureError(
                f"Unsupported quadratic phase-space monomial {powers}."
            )
    return sp.simplify(closed)


def _close_augmented_matrix(
    matrix,
    phase_variables,
    normal_moment,
    anomalous_moment,
) -> sp.Matrix:
    return sp.Matrix(matrix).applyfunc(
        lambda entry: _close_augmented_bilinears(
            entry, phase_variables, normal_moment, anomalous_moment
        )
    )


def _independent_second_moment_state(mode_names: tuple[str, ...]):
    """Build a complex-symmetric ``S=<z z^T>`` for independent partners."""
    labels = tuple(
        f"{family}_{mode}"
        for family in ("alpha", "beta")
        for mode in mode_names
    )
    d = len(labels)
    symmetric_indices = tuple(
        (i, j) for i in range(d) for j in range(i, d)
    )
    specs = []
    real_specs = []
    imag_specs = []
    for kind, prefix, latex_kind, target in (
        ("independent_re", "s_re", "Re", real_specs),
        ("independent_im", "s_im", "Im", imag_specs),
    ):
        for i, j in symmetric_indices:
            index = len(specs)
            item = StateVariableSpec(
                id=f"{prefix}_{i}_{j}",
                index=index,
                symbol=sp.Symbol(f"x{index}", real=True),
                latex_name=(
                    rf"\operatorname{{{latex_kind}}}(S_{{{labels[i]},{labels[j]}}})"
                ),
                kind=kind,
                matrix_index=(i, j),
                mode_names=(labels[i], labels[j]),
            )
            specs.append(item)
            target.append(item)
    S = sp.zeros(d)
    for (i, j), real_item, imag_item in zip(
        symmetric_indices, real_specs, imag_specs, strict=True
    ):
        value = real_item.symbol + sp.I * imag_item.symbol
        S[i, j] = value
        S[j, i] = value
    return S, tuple(specs), symmetric_indices


def _close_independent_bilinears(
    expr: sp.Expr,
    phase_variables: tuple[sp.Symbol, ...],
    second_moment: sp.Matrix,
) -> sp.Expr:
    expr = sp.expand(expr)
    try:
        polynomial = sp.Poly(expr, *phase_variables)
    except sp.PolynomialError as exc:
        raise CovarianceClosureError(
            f"Expression {expr} is not polynomial in the phase-space variables."
        ) from exc
    closed = sp.Integer(0)
    for powers, coefficient in polynomial.terms():
        degree = sum(powers)
        if degree == 0:
            closed += coefficient
            continue
        if degree != 2:
            raise CovarianceClosureError(
                "Independent second-moment substitution accepts only constants "
                f"and quadratic monomials; found powers {powers} in {expr}."
            )
        axes = [i for i, power in enumerate(powers) for _ in range(power)]
        closed += coefficient * second_moment[axes[0], axes[1]]
    return sp.simplify(closed)


def _close_independent_matrix(matrix, phase_variables, second_moment) -> sp.Matrix:
    return sp.Matrix(matrix).applyfunc(
        lambda entry: _close_independent_bilinears(
            entry, phase_variables, second_moment
        )
    )


def _close_bilinears(
    expr: sp.Expr,
    phase_variables: tuple[sp.Symbol, ...],
    covariance: sp.Matrix,
) -> sp.Expr:
    """Map ``alpha_j abar_i`` to ``R_ji`` without higher-moment closure."""
    expr = sp.expand(expr)
    try:
        polynomial = sp.Poly(expr, *phase_variables)
    except sp.PolynomialError as exc:
        raise CovarianceClosureError(
            f"Expression {expr} is not polynomial in the phase-space variables."
        ) from exc

    n = len(phase_variables) // 2
    closed = sp.Integer(0)
    for powers, coefficient in polynomial.terms():
        degree = sum(powers)
        if degree == 0:
            closed += coefficient
            continue
        alpha_axes = [i for i, power in enumerate(powers[:n]) if power]
        partner_axes = [i for i, power in enumerate(powers[n:]) if power]
        is_single_bilinear = (
            degree == 2
            and len(alpha_axes) == 1
            and len(partner_axes) == 1
            and powers[alpha_axes[0]] == 1
            and powers[n + partner_axes[0]] == 1
        )
        if not is_single_bilinear:
            raise CovarianceClosureError(
                "Second-moment closure requires constants or one bilinear "
                f"alpha_j*abar_i; found monomial powers {powers} in {expr}."
            )
        j, i = alpha_axes[0], partner_axes[0]
        closed += coefficient * covariance[j, i]
    return sp.simplify(closed)


def _close_matrix(matrix, phase_variables, covariance) -> sp.Matrix:
    return sp.Matrix(matrix).applyfunc(
        lambda entry: _close_bilinears(entry, phase_variables, covariance)
    )


def _real_part(expr: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.expand_complex(sp.re(expr)))


def _imag_part(expr: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.expand_complex(sp.im(expr)))


def _parameter_domain(symbol: sp.Symbol, overrides) -> str:
    override = overrides.get(symbol, overrides.get(str(symbol)))
    if override is not None:
        return str(override)
    if symbol.is_positive is True:
        return "positive"
    if symbol.is_nonnegative is True:
        return "nonnegative"
    if symbol.is_real is True:
        return "real"
    return "unknown"


def _parameter_specs(parameters, overrides=None) -> tuple[ParameterSpec, ...]:
    overrides = {} if overrides is None else dict(overrides)
    parameters = tuple(parameters)
    if len(set(parameters)) != len(parameters):
        raise NumericalModelError("Parameter order contains duplicates.")
    specs = []
    for index, symbol in enumerate(parameters):
        if not isinstance(symbol, sp.Symbol):
            raise NumericalModelError(
                f"Parameter {symbol!r} is not a SymPy Symbol."
            )
        domain = _parameter_domain(symbol, overrides)
        if domain == "unknown":
            raise NumericalModelError(
                f"Parameter {symbol} has no real-domain assumption; declare it "
                "with real=True or provide parameter_domains."
            )
        if domain not in {"real", "positive", "nonnegative"}:
            raise NumericalModelError(
                f"Unsupported real-kernel domain {domain!r} for parameter {symbol}."
            )
        specs.append(ParameterSpec(
            id=f"p{index}",
            index=index,
            symbol=symbol,
            kernel_symbol=sp.Symbol(f"p{index}", real=True),
            name=str(symbol),
            latex_name=sp.latex(symbol),
            domain=domain,
        ))
    return tuple(specs)


def _enforce_real_parameter_domains(matrix, specs) -> sp.Matrix:
    replacements = {}
    for item in specs:
        if item.domain in {"real", "positive", "nonnegative"}:
            replacements.update({
                sp.conjugate(item.symbol): item.symbol,
                sp.re(item.symbol): item.symbol,
                sp.im(item.symbol): sp.Integer(0),
            })
    return sp.Matrix(matrix).xreplace(replacements).applyfunc(sp.simplify)


def _structural_rank(pattern: SparsityPattern) -> int:
    """Maximum bipartite matching size for a matrix sparsity pattern."""
    rows, _ = pattern.shape
    adjacency = {row: [] for row in range(rows)}
    for row, col in pattern.entries:
        adjacency[row].append(col)
    matched_row_for_col: dict[int, int] = {}

    def augment(row: int, seen: set[int]) -> bool:
        for col in adjacency[row]:
            if col in seen:
                continue
            seen.add(col)
            if col not in matched_row_for_col or augment(
                matched_row_for_col[col], seen
            ):
                matched_row_for_col[col] = row
                return True
        return False

    return sum(augment(row, set()) for row in range(rows))


def _degree_estimate(expressions, retained_symbols) -> int | None:
    if not retained_symbols:
        return 0
    degrees = []
    for expression in expressions:
        try:
            poly = sp.Poly(expression, *retained_symbols)
        except sp.PolynomialError:
            return None
        degrees.append(poly.total_degree())
    return max(degrees, default=0)


@dataclass(frozen=True)
class AugmentedLangevinBlocks:
    """Conjugate-structured drift and noise blocks on ``z=(alpha, alpha*)``.

    External symbols are treated as real by default, matching fpgen's model
    convention. Pass an explicit ``real_parameters`` iterable to restrict
    that assumption when inspecting genuinely complex symbolic parameters.
    """

    variables: tuple[sp.Symbol, ...]
    physical_hamiltonian: sp.Matrix
    pairing_hamiltonian: sp.Matrix
    augmented_hamiltonian: sp.Matrix
    normal_diffusion: sp.Matrix
    anomalous_diffusion: sp.Matrix
    augmented_diffusion: sp.Matrix
    transpose_diffusion: sp.Matrix
    exchange_matrix: sp.Matrix
    drift_residual: sp.Matrix
    hamiltonian_structure_residual: sp.Matrix
    diffusion_structure_residual: sp.Matrix

    @classmethod
    def from_langevin(cls, langevin, *, real_parameters=None) -> AugmentedLangevinBlocks:
        if langevin.representation == "positive-p":
            raise CovarianceClosureError(
                "Conjugate augmented blocks require Wigner, P, or Q variables; "
                "positive-P partners are independent."
            )
        if langevin.hamiltonian is None:
            raise CovarianceClosureError(
                "The augmented drift has no Hamiltonian factorization."
            )

        variables = tuple(langevin.variables)
        n = len(variables) // 2
        raw_hamiltonian = sp.Matrix(langevin.hamiltonian)
        if raw_hamiltonian.shape != (2 * n, 2 * n):
            raise CovarianceClosureError(
                "The augmented Hamiltonian dimension does not match the state."
            )
        if real_parameters is None:
            phase_set = set(variables)
            real_parameters = set().union(*(
                entry.free_symbols
                for entry in (
                    tuple(raw_hamiltonian)
                    + tuple(langevin.diffusion)
                    + tuple(langevin.drift)
                )
            )) - phase_set
        real_parameters = tuple(real_parameters)
        H = raw_hamiltonian[:n, :n]
        V = raw_hamiltonian[:n, n:]
        H_bar = _formal_conjugate_matrix(H, variables, real_parameters)
        V_bar = _formal_conjugate_matrix(V, variables, real_parameters)
        augmented_hamiltonian = H.row_join(V).col_join(
            (-V_bar).row_join(-H_bar)
        )
        hamiltonian_structure_residual = _simplified_matrix(
            raw_hamiltonian - augmented_hamiltonian
        )
        drift_residual = _simplified_matrix(
            sp.Matrix(langevin.drift)
            + sp.I * augmented_hamiltonian * sp.Matrix(variables)
        )
        if drift_residual != sp.zeros(2 * n, 1):
            raise CovarianceClosureError(
                "The drift cannot be represented by conjugate blocks "
                "[[H,V],[-V*, -H*]]; residual: "
                f"{drift_residual}."
            )

        transpose_diffusion = sp.Matrix(langevin.diffusion)
        if transpose_diffusion.shape != (2 * n, 2 * n):
            raise CovarianceClosureError(
                "The augmented FPE diffusion dimension does not match the state."
            )
        M = transpose_diffusion[:n, :n]
        D = transpose_diffusion[:n, n:]
        M_bar = _formal_conjugate_matrix(M, variables, real_parameters)
        D_bar = _formal_conjugate_matrix(D, variables, real_parameters)
        expected_transpose = M.row_join(D).col_join(
            D.T.row_join(M_bar)
        )
        diffusion_structure_residual = _simplified_matrix(
            transpose_diffusion - expected_transpose
        )
        normal_residual = _simplified_matrix(D - D_bar.T)
        anomalous_residual = _simplified_matrix(M - M.T)
        if (
            diffusion_structure_residual != sp.zeros(2 * n)
            or normal_residual != sp.zeros(n)
            or anomalous_residual != sp.zeros(n)
        ):
            raise CovarianceClosureError(
                "The FPE diffusion does not have conjugate D/M block structure; "
                f"full residual: {diffusion_structure_residual}, "
                f"D-D^dagger: {normal_residual}, M-M^T: {anomalous_residual}."
            )
        augmented_diffusion = D.row_join(M).col_join(
            M_bar.row_join(D_bar)
        )
        exchange = _exchange_matrix(n)
        conversion_residual = _simplified_matrix(
            transpose_diffusion * exchange - augmented_diffusion
        )
        if conversion_residual != sp.zeros(2 * n):
            raise CovarianceClosureError(
                "Could not convert transpose diffusion to Hermitian diffusion; "
                f"residual: {conversion_residual}."
            )
        return cls(
            variables=variables,
            physical_hamiltonian=_simplified_matrix(H),
            pairing_hamiltonian=_simplified_matrix(V),
            augmented_hamiltonian=_simplified_matrix(augmented_hamiltonian),
            normal_diffusion=_simplified_matrix(D),
            anomalous_diffusion=_simplified_matrix(M),
            augmented_diffusion=_simplified_matrix(augmented_diffusion),
            transpose_diffusion=_simplified_matrix(transpose_diffusion),
            exchange_matrix=exchange,
            drift_residual=drift_residual,
            hamiltonian_structure_residual=hamiltonian_structure_residual,
            diffusion_structure_residual=diffusion_structure_residual,
        )


@dataclass(frozen=True)
class MomentLayoutDecision:
    """Symbolic evidence behind normal or augmented state selection."""

    requested_layout: str
    selected_layout: str
    normal_autonomous: bool
    anomalous_dependencies: tuple[str, ...] = ()
    projection_residual: sp.Matrix | None = None
    reasons: tuple[str, ...] = ()


class CovarianceDynamics:
    """Closed Hermitian normal-moment flow on a machine-safe real state."""

    def __init__(
        self,
        *,
        representation,
        modes,
        state_spec,
        covariance,
        pair_indices,
        pair_order,
        hamiltonian,
        diffusion,
        rhs_matrix,
        rhs,
        langevin,
        parameter_spec=(),
        parameter_order_explicit=False,
        closure_policy="factorized_bilinear",
        layout_decision=None,
    ):
        self.representation = representation
        self.modes = tuple(modes)
        self.state_spec = tuple(state_spec)
        self.coordinates = tuple(item.symbol for item in state_spec)
        self.covariance = covariance
        self.pair_indices = tuple(pair_indices)
        self.pair_order = tuple(pair_order)
        self.hamiltonian = hamiltonian
        self.diffusion = diffusion
        self.rhs_matrix = rhs_matrix
        self.rhs = rhs
        self.langevin = langevin
        self.parameter_spec = tuple(parameter_spec)
        self.parameter_order_explicit = parameter_order_explicit
        self.state_layout_version = STATE_LAYOUT_VERSION
        self.layout_kind = "normal"
        self.closure_policy = closure_policy
        self.layout_decision = layout_decision or MomentLayoutDecision(
            requested_layout="normal",
            selected_layout="normal",
            normal_autonomous=True,
            reasons=("normal layout requested explicitly",),
        )
        self.normal_moment = covariance
        self.anomalous_moment = None
        self.moment_matrix = covariance
        self.normal_rhs = rhs_matrix
        self.anomalous_rhs = None
        self.physical_hamiltonian = hamiltonian
        self.pairing_hamiltonian = sp.zeros(hamiltonian.rows)
        self.normal_diffusion = diffusion
        self.anomalous_diffusion = sp.zeros(diffusion.rows)

    @classmethod
    def from_langevin(
        cls,
        langevin,
        *,
        parameters=None,
        parameter_domains=None,
        closure_policy="factorized_bilinear",
    ) -> CovarianceDynamics:
        if closure_policy not in {"exact", "factorized_bilinear"}:
            raise CovarianceClosureError(
                "closure_policy must be 'exact' or 'factorized_bilinear'."
            )
        if langevin.representation == "positive-p":
            raise CovarianceClosureError(
                "A Hermitian covariance dynamics requires conjugate-augmented "
                "Wigner, P, or Q variables; positive-P partners are independent."
            )
        if langevin.physical_hamiltonian is None:
            raise CovarianceClosureError(
                "The drift has no closed physical Hamiltonian block."
            )
        anomalous_h = _simplified_matrix(langevin.anomalous_hamiltonian)
        if anomalous_h != sp.zeros(*anomalous_h.shape):
            raise CovarianceClosureError(
                "The anomalous Hamiltonian block is nonzero, so the Hermitian "
                "normal covariance R is not closed without anomalous moments."
            )
        master = getattr(langevin.fokker_planck, "master", None)
        if master is None:
            raise CovarianceClosureError(
                "Mode names are unavailable because the originating master "
                "equation was not retained."
            )

        modes = tuple(master.modes)
        covariance, state_spec, pair_indices, pair_order = _hermitian_state(modes)
        phase_variables = tuple(langevin.variables)
        if closure_policy == "exact":
            phase_dependence = set().union(*(
                entry.free_symbols
                for entry in tuple(langevin.physical_hamiltonian)
            )) & set(phase_variables)
            if phase_dependence:
                raise CovarianceClosureError(
                    "Exact second-moment closure requires a state-independent "
                    "Hamiltonian block; use closure='factorized_bilinear' to "
                    "request the explicit nonlinear moment substitution."
                )
        H = _close_matrix(
            langevin.physical_hamiltonian, phase_variables, covariance
        )
        D = _close_matrix(langevin.normal_correlation, phase_variables, covariance)

        explicit = parameters is not None
        specs = ()
        if explicit:
            specs = _parameter_specs(parameters, parameter_domains)
            H = _enforce_real_parameter_domains(H, specs)
            D = _enforce_real_parameter_domains(D, specs)
        rhs_matrix = _simplified_matrix(
            -sp.I * (H * covariance - covariance * H.conjugate().T) + D
        )
        if explicit:
            rhs_matrix = _enforce_real_parameter_domains(rhs_matrix, specs)
        hermitian_residual = _simplified_matrix(
            rhs_matrix - rhs_matrix.conjugate().T
        )
        if explicit:
            hermitian_residual = _enforce_real_parameter_domains(
                hermitian_residual, specs
            )
        if hermitian_residual != sp.zeros(*rhs_matrix.shape):
            raise CovarianceClosureError(
                "The derived covariance flow is not Hermitian; residual: "
                f"{hermitian_residual}."
            )

        diagonal_flow = tuple(_real_part(rhs_matrix[i, i]) for i in range(len(modes)))
        real_flow = tuple(_real_part(rhs_matrix[i, j]) for i, j in pair_indices)
        imag_flow = tuple(_imag_part(rhs_matrix[i, j]) for i, j in pair_indices)
        rhs = sp.Matrix(diagonal_flow + real_flow + imag_flow)
        if explicit:
            rhs = _enforce_real_parameter_domains(rhs, specs)

        if parameters is None:
            state_symbols = {item.symbol for item in state_spec}
            free = set().union(*(entry.free_symbols for entry in rhs)) - state_symbols
            parameters = tuple(sorted(free, key=sp.default_sort_key))
            # Inferred parameters are metadata for display only.  Their unknown
            # assumptions must not prevent construction of the symbolic flow.
            specs = tuple(ParameterSpec(
                id=f"p{index}",
                index=index,
                symbol=symbol,
                kernel_symbol=sp.Symbol(f"p{index}", real=True),
                name=str(symbol),
                latex_name=sp.latex(symbol),
                domain=_parameter_domain(symbol, {}),
            ) for index, symbol in enumerate(parameters))
        else:
            state_symbols = {item.symbol for item in state_spec}
            exported = tuple(rhs) + tuple(H) + tuple(D)
            free = set().union(
                *(entry.free_symbols for entry in exported)
            ) - state_symbols
            declared = {item.symbol for item in specs}
            missing = free - declared
            extra = declared - free
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing {sorted(map(str, missing))}")
                if extra:
                    details.append(f"unused {sorted(map(str, extra))}")
                raise NumericalModelError(
                    "Explicit parameter order does not match RHS symbols: "
                    + "; ".join(details)
                )

        return cls(
            representation=langevin.representation,
            modes=modes,
            state_spec=state_spec,
            covariance=covariance,
            pair_indices=pair_indices,
            pair_order=pair_order,
            hamiltonian=H,
            diffusion=D,
            rhs_matrix=rhs_matrix,
            rhs=rhs,
            langevin=langevin,
            parameter_spec=specs,
            parameter_order_explicit=explicit,
            closure_policy=closure_policy,
        )

    @classmethod
    def from_augmented(
        cls,
        augmented,
        *,
        layout_decision: MomentLayoutDecision,
    ) -> CovarianceDynamics:
        """Project a symbolically verified autonomous normal block."""
        n = len(augmented.modes)
        normal_size = n * n
        return cls(
            representation=augmented.representation,
            modes=augmented.modes,
            state_spec=augmented.state_spec[:normal_size],
            covariance=augmented.normal_moment,
            pair_indices=augmented.pair_indices,
            pair_order=augmented.pair_order,
            hamiltonian=augmented.physical_hamiltonian,
            diffusion=augmented.normal_diffusion,
            rhs_matrix=augmented.normal_rhs,
            rhs=augmented.rhs[:normal_size, :],
            langevin=augmented.langevin,
            parameter_spec=augmented.parameter_spec,
            parameter_order_explicit=augmented.parameter_order_explicit,
            closure_policy=augmented.closure_policy,
            layout_decision=layout_decision,
        )

    @cached_property
    def _jacobian(self) -> sp.Matrix:
        return self.rhs.jacobian(sp.Matrix(self.coordinates)).applyfunc(sp.simplify)

    def jacobian(self, *, sparse: bool = False):
        if not sparse:
            return self._jacobian
        pattern = matrix_sparsity(self._jacobian)
        values = tuple(self._jacobian[i, j] for i, j in pattern.entries)
        return pattern, values

    def jacobian_sparsity(self) -> SparsityPattern:
        return matrix_sparsity(self._jacobian)

    @cached_property
    def _parameter_jacobian(self) -> sp.Matrix:
        parameters = sp.Matrix([item.symbol for item in self.parameter_spec])
        if not self.parameter_spec:
            return sp.zeros(len(self.coordinates), 0)
        return self.rhs.jacobian(parameters).applyfunc(sp.simplify)

    def parameter_jacobian(self) -> sp.Matrix:
        return self._parameter_jacobian

    def jvp(self, direction) -> sp.Matrix:
        direction = sp.Matrix(direction)
        if direction.shape != (len(self.coordinates), 1):
            raise NumericalModelError("State direction has the wrong dimension.")
        return (self._jacobian * direction).applyfunc(sp.simplify)

    def vjp(self, cotangent) -> sp.Matrix:
        cotangent = sp.Matrix(cotangent)
        if cotangent.shape != (len(self.coordinates), 1):
            raise NumericalModelError("State cotangent has the wrong dimension.")
        return (self._jacobian.T * cotangent).applyfunc(sp.simplify)

    def parameter_jvp(self, direction) -> sp.Matrix:
        direction = sp.Matrix(direction)
        if direction.shape != (len(self.parameter_spec), 1):
            raise NumericalModelError("Parameter direction has the wrong dimension.")
        return (self._parameter_jacobian * direction).applyfunc(sp.simplify)

    def directional(
        self,
        order: int,
        state_directions=(),
        parameter_directions=(),
    ) -> sp.Matrix:
        """Contract an exact mixed derivative without materializing its tensor."""
        state_directions = tuple(sp.Matrix(item) for item in state_directions)
        parameter_directions = tuple(
            sp.Matrix(item) for item in parameter_directions
        )
        if order < 1 or len(state_directions) + len(parameter_directions) != order:
            raise NumericalModelError(
                "order must equal the number of state and parameter directions."
            )
        expression = self.rhs
        for direction in state_directions:
            if direction.shape != (len(self.coordinates), 1):
                raise NumericalModelError("State direction has the wrong dimension.")
            expression = expression.jacobian(sp.Matrix(self.coordinates)) * direction
        parameter_symbols = sp.Matrix([item.symbol for item in self.parameter_spec])
        for direction in parameter_directions:
            if direction.shape != (len(self.parameter_spec), 1):
                raise NumericalModelError(
                    "Parameter direction has the wrong dimension."
                )
            expression = expression.jacobian(parameter_symbols) * direction
        return expression.applyfunc(sp.simplify)

    @property
    def provenance(self) -> DerivationProvenance:
        fpe = self.langevin.fokker_planck
        return DerivationProvenance(
            representation=self.representation,
            phase_space_variable_order=tuple(str(v) for v in self.langevin.variables),
            fpe_truncation_order=fpe.truncation_order,
            original_kramers_moyal_order=fpe.original_max_order,
            discarded_term_count=len(fpe.discarded),
            fpe_is_exact=fpe.is_exact,
            stochastic_interpretation="ito",
            moment_closure=self.closure_policy,
            closure_assumptions=(
                (
                    "H is state independent; D contains only constants or one "
                    "alpha_j*abar_i bilinear"
                    if self.closure_policy == "exact"
                    else "bilinear phase-space factors in H and D are replaced "
                    "by second moments before matrix multiplication"
                ),
                "no implicit Gaussian/Wick factorization",
            ),
            moment_layout="normal",
            layout_selection=(
                "auto_verified"
                if self.layout_decision.requested_layout == "auto"
                else "explicit"
            ),
        )

    def to_model_spec(self, *, name: str) -> MomentDynamicsSpec:
        if not self.parameter_order_explicit:
            raise NumericalModelError(
                "Numerical export requires an explicit parameter order in "
                "to_covariance_dynamics(parameters=[...])."
            )
        substitutions = {
            item.symbol: item.kernel_symbol for item in self.parameter_spec
        }
        rhs = self.rhs.xreplace(substitutions)
        jacobian = self._jacobian.xreplace(substitutions)
        parameter_jacobian = self._parameter_jacobian.xreplace(substitutions)
        covariance = self.covariance.xreplace(substitutions)
        hamiltonian = self.hamiltonian.xreplace(substitutions)
        diffusion = self.diffusion.xreplace(substitutions)
        provenance = self.provenance
        fingerprint = make_fingerprint(
            model_kind="moment_dynamics",
            state=self.state_spec,
            parameters=self.parameter_spec,
            expressions=(
                tuple(rhs)
                + tuple(covariance)
                + tuple(hamiltonian)
                + tuple(diffusion)
            ),
            provenance=provenance,
        )
        return MomentDynamicsSpec(
            schema_version=MODEL_SCHEMA_VERSION,
            model_kind="moment_dynamics",
            name=name,
            state=self.state_spec,
            parameters=self.parameter_spec,
            rhs=rhs,
            jacobian=jacobian,
            parameter_jacobian=parameter_jacobian,
            jacobian_sparsity=matrix_sparsity(jacobian),
            provenance=provenance,
            fingerprint=fingerprint,
            covariance=covariance,
            hamiltonian=hamiltonian,
            diffusion=diffusion,
        )

    def find_linear_reductions(
        self,
        *,
        retained_dimension: int = 1,
        max_candidates: int | None = None,
        equation_partitions: str = "all",
    ) -> tuple[LinearReductionCandidate, ...]:
        n = len(self.coordinates)
        if not 1 <= retained_dimension < n:
            raise ReductionError("retained_dimension must lie between 1 and n-1.")
        if equation_partitions not in {"all", "matching"}:
            raise ReductionError(
                "equation_partitions must be 'all' or 'matching'."
            )
        candidates = []
        for retained in combinations(range(n), retained_dimension):
            eliminated = tuple(i for i in range(n) if i not in retained)
            y = sp.Matrix([self.coordinates[i] for i in eliminated])
            equation_options = (
                (retained,)
                if equation_partitions == "matching"
                else combinations(range(n), retained_dimension)
            )
            for retained_equations in equation_options:
                eliminated_equations = tuple(
                    i for i in range(n) if i not in retained_equations
                )
                h = sp.Matrix([self.rhs[i] for i in eliminated_equations])
                A = h.jacobian(y)
                zero_y = {symbol: 0 for symbol in y}
                b = h.subs(zero_y)
                residual = (h - A * y - b).applyfunc(sp.simplify)
                if residual != sp.zeros(len(eliminated), 1):
                    continue
                pattern = matrix_sparsity(A)
                structural_rank = _structural_rank(pattern)
                if structural_rank < len(eliminated):
                    continue
                retained_symbols = tuple(self.coordinates[i] for i in retained)
                retained_residual = tuple(
                    self.rhs[i] for i in retained_equations
                )
                degree = _degree_estimate(
                    tuple(A) + tuple(b) + retained_residual,
                    retained_symbols,
                )
                operation_count = sum(
                    int(sp.count_ops(expr))
                    for expr in tuple(A) + tuple(b) + retained_residual
                )
                candidates.append(LinearReductionCandidate(
                    retained_indices=retained,
                    retained_ids=tuple(self.state_spec[i].id for i in retained),
                    retained_equations=tuple(retained_equations),
                    eliminated_indices=eliminated,
                    eliminated_ids=tuple(
                        self.state_spec[i].id for i in eliminated
                    ),
                    eliminated_equations=eliminated_equations,
                    A_sparsity=pattern,
                    structural_rank=structural_rank,
                    required_rank=len(eliminated),
                    operation_count=operation_count,
                    degree_estimate=degree,
                    reduced_degree=None,
                    denominator_operation_count=None,
                    diagnostics=(),
                ))
        enriched = []
        for candidate in candidates:
            if len(candidate.eliminated_indices) > 3:
                enriched.append(candidate)
                continue
            try:
                materialized = LinearReductionPlan.from_candidate(
                    self, candidate
                ).materialize()
                reduced_degree = None
                if len(candidate.retained_indices) == 1:
                    reduced_degree = max(
                        sp.Poly(expr, self.coordinates[candidate.retained_indices[0]]).degree()
                        for expr in materialized.numerators
                    )
                denominator_cost = sum(
                    int(sp.count_ops(expr)) for expr in materialized.denominators
                )
                candidate = replace(
                    candidate,
                    reduced_degree=reduced_degree,
                    denominator_operation_count=denominator_cost,
                )
            except (ReductionError, sp.PolynomialError):
                pass
            enriched.append(candidate)
        candidates = enriched
        candidates.sort(key=lambda item: (
            item.structural_rank < item.required_rank,
            item.operation_count,
            item.A_sparsity.nnz,
            item.reduced_degree if item.reduced_degree is not None else 10**9,
            item.denominator_operation_count
            if item.denominator_operation_count is not None else 10**9,
            item.degree_estimate if item.degree_estimate is not None else 10**9,
            item.retained_indices,
        ))
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
        return tuple(candidates)

    def linear_reduce(
        self,
        *,
        candidate: LinearReductionCandidate | None = None,
        order_parameters=None,
    ) -> LinearReductionPlan:
        if (candidate is None) == (order_parameters is None):
            raise ReductionError(
                "Provide exactly one of candidate or order_parameters."
            )
        if candidate is None:
            lookup = {item.id: item.index for item in self.state_spec}
            lookup.update({item.symbol: item.index for item in self.state_spec})
            try:
                retained = tuple(lookup[item] for item in order_parameters)
            except KeyError as exc:
                raise ReductionError(f"Unknown order parameter {exc.args[0]!r}.") from exc
            matches = [
                item for item in self.find_linear_reductions(
                    retained_dimension=len(retained)
                )
                if item.retained_indices == retained
            ]
            if not matches:
                raise ReductionError(
                    f"No matching linear reduction exists for indices {retained}."
                )
            candidate = matches[0]
        return LinearReductionPlan.from_candidate(self, candidate)


class AugmentedSecondMomentDynamics(CovarianceDynamics):
    """Hermitian normal/anomalous dynamics on ``R_tilde=<z z^dagger>``."""

    @classmethod
    def from_langevin(
        cls,
        langevin,
        *,
        parameters=None,
        parameter_domains=None,
        closure_policy="exact",
    ) -> AugmentedSecondMomentDynamics:
        if closure_policy not in {"exact", "factorized_bilinear"}:
            raise CovarianceClosureError(
                "closure_policy must be 'exact' or 'factorized_bilinear'."
            )
        blocks = AugmentedLangevinBlocks.from_langevin(
            langevin, real_parameters=parameters
        )
        master = getattr(langevin.fokker_planck, "master", None)
        if master is None:
            raise CovarianceClosureError(
                "Mode names are unavailable because the master equation was not retained."
            )
        modes = tuple(master.modes)
        n = len(modes)
        (
            R,
            C,
            augmented_moment,
            state_spec,
            pair_indices,
            pair_order,
            symmetric_indices,
        ) = _augmented_second_moment_state(modes)
        phase_variables = tuple(langevin.variables)

        if closure_policy == "exact":
            phase_dependence = set().union(*(
                entry.free_symbols
                for entry in tuple(blocks.augmented_hamiltonian)
            )) & set(phase_variables)
            if phase_dependence:
                raise CovarianceClosureError(
                    "Exact second-moment closure requires a state-independent "
                    "augmented Hamiltonian; use closure='factorized_bilinear' "
                    "for the explicit nonlinear moment substitution."
                )

        H_aug = _close_augmented_matrix(
            blocks.augmented_hamiltonian,
            phase_variables,
            R,
            C,
        )
        D_aug = _close_augmented_matrix(
            blocks.augmented_diffusion,
            phase_variables,
            R,
            C,
        )

        explicit = parameters is not None
        parameter_spec = ()
        if explicit:
            parameter_spec = _parameter_specs(parameters, parameter_domains)
            H_aug = _enforce_real_parameter_domains(H_aug, parameter_spec)
            D_aug = _enforce_real_parameter_domains(D_aug, parameter_spec)

        rhs_matrix = _simplified_matrix(
            -sp.I * H_aug * augmented_moment
            + sp.I * augmented_moment * H_aug.conjugate().T
            + D_aug
        )
        if explicit:
            rhs_matrix = _enforce_real_parameter_domains(rhs_matrix, parameter_spec)
        hermitian_residual = _simplified_matrix(
            rhs_matrix - rhs_matrix.conjugate().T
        )
        if explicit:
            hermitian_residual = _enforce_real_parameter_domains(
                hermitian_residual, parameter_spec
            )
        if hermitian_residual != sp.zeros(2 * n):
            raise CovarianceClosureError(
                "The augmented second-moment flow is not Hermitian; residual: "
                f"{hermitian_residual}."
            )

        R_flow = rhs_matrix[:n, :n]
        C_flow = rhs_matrix[:n, n:]
        normal_symmetry_residual = _simplified_matrix(
            R_flow - R_flow.conjugate().T
        )
        if explicit:
            normal_symmetry_residual = _enforce_real_parameter_domains(
                normal_symmetry_residual, parameter_spec
            )
        if normal_symmetry_residual != sp.zeros(n):
            raise CovarianceClosureError(
                "The normal block of the augmented moment flow is not Hermitian."
            )
        if _simplified_matrix(C_flow - C_flow.T) != sp.zeros(n):
            raise CovarianceClosureError(
                "The anomalous block of the augmented moment flow is not symmetric."
            )

        normal_diagonal = tuple(_real_part(R_flow[i, i]) for i in range(n))
        normal_real = tuple(_real_part(R_flow[i, j]) for i, j in pair_indices)
        normal_imag = tuple(_imag_part(R_flow[i, j]) for i, j in pair_indices)
        anomalous_real = tuple(
            _real_part(C_flow[i, j]) for i, j in symmetric_indices
        )
        anomalous_imag = tuple(
            _imag_part(C_flow[i, j]) for i, j in symmetric_indices
        )
        rhs = sp.Matrix(
            normal_diagonal
            + normal_real
            + normal_imag
            + anomalous_real
            + anomalous_imag
        )
        if explicit:
            rhs = _enforce_real_parameter_domains(rhs, parameter_spec)

        state_symbols = {item.symbol for item in state_spec}
        free = set().union(*(
            entry.free_symbols
            for entry in tuple(rhs) + tuple(H_aug) + tuple(D_aug)
        )) - state_symbols
        if not explicit:
            inferred = tuple(sorted(free, key=sp.default_sort_key))
            parameter_spec = tuple(ParameterSpec(
                id=f"p{index}",
                index=index,
                symbol=symbol,
                kernel_symbol=sp.Symbol(f"p{index}", real=True),
                name=str(symbol),
                latex_name=sp.latex(symbol),
                domain=_parameter_domain(symbol, {}),
            ) for index, symbol in enumerate(inferred))
        else:
            declared = {item.symbol for item in parameter_spec}
            if free != declared:
                raise NumericalModelError(
                    "Explicit parameter order does not match augmented moment symbols: "
                    f"expected {sorted(map(str, free))}, "
                    f"got {sorted(map(str, declared))}."
                )

        result = cls(
            representation=langevin.representation,
            modes=modes,
            state_spec=state_spec,
            covariance=augmented_moment,
            pair_indices=pair_indices,
            pair_order=pair_order,
            hamiltonian=H_aug,
            diffusion=D_aug,
            rhs_matrix=rhs_matrix,
            rhs=rhs,
            langevin=langevin,
            parameter_spec=parameter_spec,
            parameter_order_explicit=explicit,
            closure_policy=closure_policy,
            layout_decision=MomentLayoutDecision(
                requested_layout="augmented",
                selected_layout="augmented",
                normal_autonomous=False,
                reasons=("augmented layout requested explicitly",),
            ),
        )
        result.state_layout_version = AUGMENTED_SECOND_MOMENT_LAYOUT_VERSION
        result.layout_kind = "augmented"
        result.normal_moment = R
        result.anomalous_moment = C
        result.moment_matrix = augmented_moment
        result.normal_rhs = R_flow
        result.anomalous_rhs = C_flow
        result.physical_hamiltonian = H_aug[:n, :n]
        result.pairing_hamiltonian = H_aug[:n, n:]
        result.normal_diffusion = D_aug[:n, :n]
        result.anomalous_diffusion = D_aug[:n, n:]
        result.augmented_blocks = blocks
        return result

    def normal_layout_decision(
        self, *, requested_layout="auto"
    ) -> MomentLayoutDecision:
        n = len(self.modes)
        normal_size = n * n
        anomalous_symbols = set(self.coordinates[normal_size:])
        dependencies = set().union(*(
            entry.free_symbols for entry in self.normal_rhs
        )) & anomalous_symbols
        projected_formula = _simplified_matrix(
            -sp.I * (
                self.physical_hamiltonian * self.normal_moment
                - self.normal_moment * self.physical_hamiltonian.conjugate().T
            )
            + self.normal_diffusion
        )
        if self.parameter_order_explicit:
            projected_formula = _enforce_real_parameter_domains(
                projected_formula, self.parameter_spec
            )
        projection_residual = _simplified_matrix(
            self.normal_rhs - projected_formula
        )
        if self.parameter_order_explicit:
            projection_residual = _enforce_real_parameter_domains(
                projection_residual, self.parameter_spec
            )
        autonomous = (
            not dependencies and projection_residual == sp.zeros(n)
        )
        reasons = []
        if dependencies:
            reasons.append("normal flow depends on anomalous coordinates")
        if projection_residual != sp.zeros(n):
            reasons.append("normal projection does not match the closed R equation")
        if autonomous:
            reasons.append("normal block is autonomous and projection-equivalent")
        return MomentLayoutDecision(
            requested_layout=requested_layout,
            selected_layout="normal" if autonomous else "augmented",
            normal_autonomous=autonomous,
            anomalous_dependencies=tuple(
                sorted(map(str, dependencies))
            ),
            projection_residual=projection_residual,
            reasons=tuple(reasons),
        )

    def project_normal(self, *, requested_layout="auto") -> CovarianceDynamics:
        decision = self.normal_layout_decision(requested_layout=requested_layout)
        if not decision.normal_autonomous:
            raise CovarianceClosureError(
                "The augmented dynamics cannot be reduced to the normal block: "
                + "; ".join(decision.reasons)
            )
        return CovarianceDynamics.from_augmented(
            self, layout_decision=decision
        )

    @property
    def provenance(self) -> DerivationProvenance:
        fpe = self.langevin.fokker_planck
        return DerivationProvenance(
            representation=self.representation,
            phase_space_variable_order=tuple(str(v) for v in self.langevin.variables),
            fpe_truncation_order=fpe.truncation_order,
            original_kramers_moyal_order=fpe.original_max_order,
            discarded_term_count=len(fpe.discarded),
            fpe_is_exact=fpe.is_exact,
            stochastic_interpretation="ito",
            moment_closure=self.closure_policy,
            closure_assumptions=(
                (
                    "H_aug is state independent; D_aug contains only constants "
                    "or quadratic phase-space monomials"
                    if self.closure_policy == "exact"
                    else "quadratic phase-space factors in H_aug and D_aug are "
                    "replaced by R/C moments before matrix multiplication"
                ),
                "R_tilde=<z z^dagger> with Hermitian R and symmetric C",
                "no implicit Gaussian/Wick factorization",
            ),
            state_layout_version=AUGMENTED_SECOND_MOMENT_LAYOUT_VERSION,
            moment_layout="augmented",
            layout_selection=(
                "auto_verified"
                if self.layout_decision.requested_layout == "auto"
                else "explicit"
            ),
        )


class IndependentSecondMomentDynamics(CovarianceDynamics):
    """Complex-symmetric second moments for positive-P independent partners."""

    @classmethod
    def from_langevin(
        cls,
        langevin,
        *,
        parameters=None,
        parameter_domains=None,
        closure_policy="exact",
    ) -> IndependentSecondMomentDynamics:
        if langevin.representation != "positive-p":
            raise CovarianceClosureError(
                "Independent second moments are reserved for positive-P variables."
            )
        if langevin.hamiltonian is None:
            raise CovarianceClosureError(
                "The independent augmented drift has no Hamiltonian factorization."
            )
        if closure_policy not in {"exact", "factorized_bilinear"}:
            raise CovarianceClosureError(
                "closure_policy must be 'exact' or 'factorized_bilinear'."
            )
        master = getattr(langevin.fokker_planck, "master", None)
        if master is None:
            raise CovarianceClosureError(
                "Mode names are unavailable because the master equation was not retained."
            )
        modes = tuple(master.modes)
        variables = tuple(langevin.variables)
        S, state_spec, symmetric_indices = _independent_second_moment_state(modes)
        raw_hamiltonian = sp.Matrix(langevin.hamiltonian)
        raw_diffusion = sp.Matrix(langevin.diffusion)
        phase_variable_set = set(variables)
        if closure_policy == "exact":
            phase_dependence = set().union(*(
                entry.free_symbols for entry in raw_hamiltonian
            )) & phase_variable_set
            if phase_dependence:
                raise CovarianceClosureError(
                    "Exact independent second-moment closure requires a "
                    "state-independent augmented Hamiltonian; use "
                    "closure='factorized_bilinear' for nonlinear substitution."
                )
        H = _close_independent_matrix(raw_hamiltonian, variables, S)
        G = _close_independent_matrix(raw_diffusion, variables, S)

        explicit = parameters is not None
        parameter_spec = ()
        if explicit:
            parameter_spec = _parameter_specs(parameters, parameter_domains)
            H = _enforce_real_parameter_domains(H, parameter_spec)
            G = _enforce_real_parameter_domains(G, parameter_spec)
        rhs_matrix = _simplified_matrix(
            -sp.I * (H * S + S * H.T) + G
        )
        if explicit:
            rhs_matrix = _enforce_real_parameter_domains(rhs_matrix, parameter_spec)
        symmetry_residual = _simplified_matrix(rhs_matrix - rhs_matrix.T)
        if symmetry_residual != sp.zeros(len(variables)):
            raise CovarianceClosureError(
                "The positive-P second-moment flow is not complex symmetric; "
                f"residual: {symmetry_residual}."
            )
        real_flow = tuple(
            _real_part(rhs_matrix[i, j]) for i, j in symmetric_indices
        )
        imag_flow = tuple(
            _imag_part(rhs_matrix[i, j]) for i, j in symmetric_indices
        )
        rhs = sp.Matrix(real_flow + imag_flow)
        if explicit:
            rhs = _enforce_real_parameter_domains(rhs, parameter_spec)

        state_symbols = {item.symbol for item in state_spec}
        free = set().union(*(
            entry.free_symbols
            for entry in tuple(rhs) + tuple(H) + tuple(G)
        )) - state_symbols
        if not explicit:
            inferred = tuple(sorted(free, key=sp.default_sort_key))
            parameter_spec = tuple(ParameterSpec(
                id=f"p{index}",
                index=index,
                symbol=symbol,
                kernel_symbol=sp.Symbol(f"p{index}", real=True),
                name=str(symbol),
                latex_name=sp.latex(symbol),
                domain=_parameter_domain(symbol, {}),
            ) for index, symbol in enumerate(inferred))
        else:
            declared = {item.symbol for item in parameter_spec}
            if free != declared:
                raise NumericalModelError(
                    "Explicit parameter order does not match independent moment "
                    f"symbols: expected {sorted(map(str, free))}, "
                    f"got {sorted(map(str, declared))}."
                )

        labels = tuple(item.mode_names for item in state_spec[:len(symmetric_indices)])
        result = cls(
            representation=langevin.representation,
            modes=modes,
            state_spec=state_spec,
            covariance=S,
            pair_indices=symmetric_indices,
            pair_order=labels,
            hamiltonian=H,
            diffusion=G,
            rhs_matrix=rhs_matrix,
            rhs=rhs,
            langevin=langevin,
            parameter_spec=parameter_spec,
            parameter_order_explicit=explicit,
            closure_policy=closure_policy,
            layout_decision=MomentLayoutDecision(
                requested_layout="independent",
                selected_layout="independent",
                normal_autonomous=False,
                reasons=("positive-P partners are independent",),
            ),
        )
        result.state_layout_version = INDEPENDENT_SECOND_MOMENT_LAYOUT_VERSION
        result.layout_kind = "independent"
        result.normal_moment = None
        result.anomalous_moment = None
        result.moment_matrix = S
        result.normal_rhs = None
        result.anomalous_rhs = None
        result.physical_hamiltonian = None
        result.pairing_hamiltonian = None
        result.normal_diffusion = None
        result.anomalous_diffusion = None
        return result

    @property
    def provenance(self) -> DerivationProvenance:
        fpe = self.langevin.fokker_planck
        return DerivationProvenance(
            representation=self.representation,
            phase_space_variable_order=tuple(str(v) for v in self.langevin.variables),
            fpe_truncation_order=fpe.truncation_order,
            original_kramers_moyal_order=fpe.original_max_order,
            discarded_term_count=len(fpe.discarded),
            fpe_is_exact=fpe.is_exact,
            stochastic_interpretation="ito",
            moment_closure=self.closure_policy,
            closure_assumptions=(
                "positive-P alpha/beta variables are independent",
                (
                    "H_aug is state independent; G contains only constants or "
                    "quadratic phase-space monomials"
                    if self.closure_policy == "exact"
                    else "quadratic phase-space factors in H_aug and G are "
                    "replaced by complex-symmetric second moments"
                ),
                "no conjugacy or Hermitian constraint is imposed",
            ),
            state_layout_version=INDEPENDENT_SECOND_MOMENT_LAYOUT_VERSION,
            moment_layout="independent",
            layout_selection=(
                "auto_verified"
                if self.layout_decision.requested_layout == "auto"
                else "explicit"
            ),
        )


@dataclass(frozen=True)
class LinearReductionCandidate:
    """A matching equation/state partition that is affine in eliminated state."""

    retained_indices: tuple[int, ...]
    retained_ids: tuple[str, ...]
    retained_equations: tuple[int, ...]
    eliminated_indices: tuple[int, ...]
    eliminated_ids: tuple[str, ...]
    eliminated_equations: tuple[int, ...]
    A_sparsity: SparsityPattern
    structural_rank: int
    required_rank: int
    operation_count: int
    degree_estimate: int | None
    reduced_degree: int | None
    denominator_operation_count: int | None
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class LinearReductionPlan:
    """Exact affine elimination structure ``h(q,y,p) = A(q,p)y + b(q,p)``."""

    dynamics: CovarianceDynamics
    candidate: LinearReductionCandidate
    retained_symbols: tuple[sp.Symbol, ...]
    eliminated_symbols: tuple[sp.Symbol, ...]
    A: sp.Matrix
    b: sp.Matrix
    retained_residual: sp.Matrix
    linearity_residual: sp.Matrix

    @classmethod
    def from_candidate(cls, dynamics, candidate) -> LinearReductionPlan:
        retained = candidate.retained_indices
        eliminated = candidate.eliminated_indices
        q = tuple(dynamics.coordinates[i] for i in retained)
        y = tuple(dynamics.coordinates[i] for i in eliminated)
        y_matrix = sp.Matrix(y)
        h = sp.Matrix([dynamics.rhs[i] for i in candidate.eliminated_equations])
        A = h.jacobian(y_matrix)
        b = h.subs({symbol: 0 for symbol in y})
        residual = (h - A * y_matrix - b).applyfunc(sp.simplify)
        if residual != sp.zeros(len(y), 1):
            raise ReductionError(
                "The selected eliminated equations are not affine in the "
                f"eliminated state; residual is {residual}."
            )
        retained_residual = sp.Matrix([
            dynamics.rhs[i] for i in candidate.retained_equations
        ])
        return cls(
            dynamics=dynamics,
            candidate=candidate,
            retained_symbols=q,
            eliminated_symbols=y,
            A=A,
            b=b,
            retained_residual=retained_residual,
            linearity_residual=residual,
        )

    @property
    def rank_condition(self) -> str:
        return f"rank(A) == {len(self.eliminated_symbols)}"

    @property
    def singular_branch(self) -> str:
        return f"rank(A) < {len(self.eliminated_symbols)}"

    def assemble_A_b(self) -> tuple[sp.Matrix, sp.Matrix]:
        """Return the exact affine block for a downstream linear solver."""
        return self.A, self.b

    def materialize(self, *, method: str = "fraction_free") -> MaterializedReduction:
        if method != "fraction_free":
            raise ReductionError("Only fraction_free materialization is supported.")
        augmented = self.A.row_join(-self.b)
        solution = sp.solve_linear_system(augmented, *self.eliminated_symbols)
        if solution is None or any(
            symbol not in solution for symbol in self.eliminated_symbols
        ):
            raise ReductionError(
                "The elimination block has no unique symbolic solution on the "
                "regular branch."
            )
        eliminated_solution = sp.Matrix([
            sp.factor(solution[symbol]) for symbol in self.eliminated_symbols
        ])
        substitution = dict(zip(
            self.eliminated_symbols, eliminated_solution, strict=True
        ))
        reduced = self.retained_residual.subs(substitution).applyfunc(
            lambda expr: sp.factor(sp.cancel(expr))
        )
        numerators = []
        denominators = []
        for expression in reduced:
            numerator, denominator = sp.fraction(sp.together(expression))
            numerators.append(sp.factor(numerator))
            denominators.append(sp.factor(denominator))
        cleared = []
        for denominator in denominators:
            factored = sp.factor(denominator)
            factors = factored.args if isinstance(factored, sp.Mul) else (factored,)
            for factor in factors:
                if factor.is_number:
                    continue
                base = factor.base if isinstance(factor, sp.Pow) else factor
                if base not in cleared:
                    cleared.append(base)
        cleared_factors = tuple(cleared)
        determinant = sp.factor(self.A.det())
        full_residual = self.dynamics.rhs.subs(substitution).applyfunc(
            lambda expr: sp.factor(sp.cancel(expr))
        )
        return MaterializedReduction(
            plan=self,
            eliminated_solution=eliminated_solution,
            reduced_residual=reduced,
            numerators=sp.Matrix(numerators),
            denominators=sp.Matrix(denominators),
            cleared_factors=cleared_factors,
            regularity_determinant=determinant,
            full_residual=full_residual,
        )


@dataclass(frozen=True)
class MaterializedReduction:
    """A fraction-free symbolic reduction on the regular branch."""

    plan: LinearReductionPlan
    eliminated_solution: sp.Matrix
    reduced_residual: sp.Matrix
    numerators: sp.Matrix
    denominators: sp.Matrix
    cleared_factors: tuple[sp.Expr, ...]
    regularity_determinant: sp.Expr
    full_residual: sp.Matrix

    def reconstruct_full_state(self) -> sp.Matrix:
        """Reconstruct the full state in the canonical fpgen ordering."""
        values = {}
        values.update(zip(
            self.plan.candidate.retained_indices,
            self.plan.retained_symbols,
            strict=True,
        ))
        values.update(zip(
            self.plan.candidate.eliminated_indices,
            self.eliminated_solution,
            strict=True,
        ))
        return sp.Matrix([values[i] for i in range(len(values))])

    def validation_residual(self) -> sp.Matrix:
        """Return the complete RHS after reconstruction on the regular branch."""
        return self.full_residual

    def derivative(self, order: int = 1) -> sp.Matrix:
        if len(self.plan.retained_symbols) != 1:
            raise ReductionError("Scalar derivatives require one retained variable.")
        if order < 0:
            raise ReductionError("Derivative order must be non-negative.")
        q = self.plan.retained_symbols[0]
        return self.numerators.applyfunc(lambda expr: sp.diff(expr, q, order))

    def multiplicity_conditions(self, order: int) -> tuple[sp.Matrix, ...]:
        if order < 1:
            raise ReductionError("Multiplicity order must be positive.")
        return tuple(self.derivative(k) for k in range(order))


@dataclass(frozen=True)
class HermitianCovarianceJacobian:
    """Notebook-compatible display facade for :class:`CovarianceDynamics`."""

    modes: tuple[str, ...]
    covariance: sp.Matrix
    coordinates: tuple[sp.Symbol, ...]
    pair_order: tuple[tuple[str, str], ...]
    pair_indices: tuple[tuple[int, int], ...]
    state_spec: tuple[StateVariableSpec, ...]
    state_layout_version: str
    hamiltonian: sp.Matrix
    diffusion: sp.Matrix
    flow_matrix: sp.Matrix
    flow_vector: sp.Matrix
    jacobian: sp.Matrix
    dynamics: CovarianceDynamics

    @classmethod
    def from_dynamics(cls, dynamics) -> HermitianCovarianceJacobian:
        if dynamics.layout_kind != "normal":
            raise CovarianceClosureError(
                "HermitianCovarianceJacobian requires the normal moment layout."
            )
        display_coordinates = tuple(
            _display_coordinate(
                "diag" if item.kind == "diag" else (
                    "Re" if item.kind == "offdiag_re" else "Im"
                ),
                *item.mode_names,
            )
            for item in dynamics.state_spec
        )
        substitutions = dict(zip(
            dynamics.coordinates, display_coordinates, strict=True
        ))
        convert = lambda matrix: sp.Matrix(matrix).xreplace(substitutions)
        return cls(
            modes=dynamics.modes,
            covariance=convert(dynamics.covariance),
            coordinates=display_coordinates,
            pair_order=dynamics.pair_order,
            pair_indices=dynamics.pair_indices,
            state_spec=dynamics.state_spec,
            state_layout_version=dynamics.state_layout_version,
            hamiltonian=convert(dynamics.hamiltonian),
            diffusion=convert(dynamics.diffusion),
            flow_matrix=convert(dynamics.rhs_matrix),
            flow_vector=convert(dynamics.rhs),
            jacobian=convert(dynamics.jacobian()),
            dynamics=dynamics,
        )

    @classmethod
    def from_langevin(cls, langevin) -> HermitianCovarianceJacobian:
        dynamics = langevin.to_second_moment_dynamics(
            layout="normal",
            closure="factorized_bilinear",
        )
        return cls.from_dynamics(dynamics)

    def __repr__(self) -> str:
        return (
            f"HermitianCovarianceJacobian(modes={self.modes!r}, "
            f"coordinates={tuple(str(v) for v in self.coordinates)!r})\n"
            f"R = {self.covariance}\n"
            f"dot(r) = {self.flow_vector}\n"
            f"J = {self.jacobian}"
        )

    def to_latex(self) -> str:
        return r" \\ ".join([
            r"\mathbf r = " + sp.latex(sp.Matrix(self.coordinates)),
            r"\dot{\mathbf r} = " + sp.latex(self.flow_vector),
            r"\mathbf J(\mathbf R) = " + sp.latex(self.jacobian),
        ])

    def _repr_latex_(self) -> str:
        return f"${self.to_latex()}$"
