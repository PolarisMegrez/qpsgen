"""Closed Hermitian covariance dynamics and structural reductions.

For a c-number Langevin equation ``dot(alpha) = -I H alpha + xi``, the
normal covariance ``R_ij = <alpha_i alpha_j^*>`` obeys

    dot(R) = -I (H(R) R - R H(R)^dagger) + D(R)

when every phase-space dependence in ``H`` and ``D`` is constant or one
bilinear ``alpha_j abar_i``.  No Gaussian/Wick closure is applied implicitly.
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

__all__ = [
    "CovarianceDynamics",
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


class CovarianceDynamics:
    """Exact closed Hermitian covariance flow on a machine-safe real state."""

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

    @classmethod
    def from_langevin(
        cls,
        langevin,
        *,
        parameters=None,
        parameter_domains=None,
    ) -> CovarianceDynamics:
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
        if hermitian_residual != sp.zeros(*rhs_matrix.shape):
            raise CovarianceClosureError(
                "The derived covariance flow is not Hermitian; residual: "
                f"{hermitian_residual}."
            )

        diagonal_flow = tuple(_real_part(rhs_matrix[i, i]) for i in range(len(modes)))
        real_flow = tuple(_real_part(rhs_matrix[i, j]) for i, j in pair_indices)
        imag_flow = tuple(_imag_part(rhs_matrix[i, j]) for i, j in pair_indices)
        rhs = sp.Matrix(diagonal_flow + real_flow + imag_flow)

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
            moment_closure="exact_bilinear",
            closure_assumptions=(
                "H and D contain only constants or one alpha_j*abar_i bilinear",
                "no implicit Gaussian/Wick factorization",
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
        return cls.from_dynamics(CovarianceDynamics.from_langevin(langevin))

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
