"""Backend-neutral numerical model specifications.

The specifications in this module contain exact SymPy expressions and stable
array layouts.  They deliberately do not depend on a numerical runtime.  A
NumPy reference module can be generated explicitly for downstream validation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import sympy as sp
from sympy.printing.numpy import NumPyPrinter

from qpsgen.errors import NumericalModelError

__all__ = [
    "MOMENT_DYNAMICS_API_VERSION",
    "REDUCTION_API_VERSION",
    "MODEL_SCHEMA_VERSION",
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
]


MOMENT_DYNAMICS_API_VERSION = "1.0"
REDUCTION_API_VERSION = "1.1"
MODEL_SCHEMA_VERSION = "2.0"
STATE_LAYOUT_VERSION = "hermitian-declared-index-v1"
AUGMENTED_SECOND_MOMENT_LAYOUT_VERSION = "hermitian-normal-anomalous-declared-index-v2"
INDEPENDENT_SECOND_MOMENT_LAYOUT_VERSION = (
    "independent-complex-symmetric-declared-index-v1"
)


@dataclass(frozen=True)
class StateVariableSpec:
    """One real state coordinate in a stable flat-array layout."""

    id: str
    index: int
    symbol: sp.Symbol
    latex_name: str
    kind: str
    matrix_index: tuple[int, int]
    mode_names: tuple[str, str]
    real: bool = True

    def manifest(self) -> dict[str, Any]:
        data = asdict(self)
        data["symbol"] = str(self.symbol)
        data["matrix_index"] = list(self.matrix_index)
        data["mode_names"] = list(self.mode_names)
        return data


@dataclass(frozen=True)
class ParameterSpec:
    """One explicitly ordered model parameter."""

    id: str
    index: int
    symbol: sp.Symbol
    kernel_symbol: sp.Symbol
    name: str
    latex_name: str
    domain: str

    def manifest(self) -> dict[str, Any]:
        data = asdict(self)
        data["symbol"] = str(self.symbol)
        data["kernel_symbol"] = str(self.kernel_symbol)
        return data


@dataclass(frozen=True)
class SparsityPattern:
    """Coordinate sparsity of a matrix expression."""

    shape: tuple[int, int]
    entries: tuple[tuple[int, int], ...]

    @property
    def nnz(self) -> int:
        return len(self.entries)

    def manifest(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "entries": [list(entry) for entry in self.entries],
        }


@dataclass(frozen=True)
class DerivationProvenance:
    """Machine-readable assumptions behind a numerical dynamics model."""

    representation: str
    phase_space_variable_order: tuple[str, ...]
    fpe_truncation_order: int
    original_kramers_moyal_order: int
    discarded_term_count: int
    fpe_is_exact: bool
    stochastic_interpretation: str
    moment_closure: str
    closure_assumptions: tuple[str, ...]
    state_layout_version: str = STATE_LAYOUT_VERSION
    moment_layout: str = "normal"
    layout_selection: str = "explicit"

    def manifest(self) -> dict[str, Any]:
        data = asdict(self)
        data["phase_space_variable_order"] = list(self.phase_space_variable_order)
        data["closure_assumptions"] = list(self.closure_assumptions)
        return data


def _matrix_sparsity(matrix: sp.Matrix) -> SparsityPattern:
    entries = tuple(
        (i, j)
        for i in range(matrix.rows)
        for j in range(matrix.cols)
        if sp.simplify(matrix[i, j]) != 0
    )
    return SparsityPattern((matrix.rows, matrix.cols), entries)


def _fingerprint_payload(
    *,
    model_kind: str,
    state: tuple[StateVariableSpec, ...],
    parameters: tuple[ParameterSpec, ...],
    expressions: tuple[sp.Expr, ...],
    provenance: DerivationProvenance,
) -> str:
    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_kind": model_kind,
        "state": [item.manifest() for item in state],
        "parameters": [item.manifest() for item in parameters],
        "provenance": provenance.manifest(),
        "expressions": [sp.srepr(expr) for expr in expressions],
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NumericalModelSpec:
    """Common metadata for backend-neutral numerical models."""

    schema_version: str
    model_kind: str
    name: str
    state: tuple[StateVariableSpec, ...]
    parameters: tuple[ParameterSpec, ...]
    rhs: sp.Matrix
    jacobian: sp.Matrix
    parameter_jacobian: sp.Matrix
    jacobian_sparsity: SparsityPattern
    provenance: DerivationProvenance
    fingerprint: str
    derivative_source: str = "exact_sympy"
    numeric_dtype: str = "float64"

    @property
    def state_size(self) -> int:
        return len(self.state)

    @property
    def parameter_size(self) -> int:
        return len(self.parameters)

    def state_index(self, identifier: str) -> int:
        """Return the stable flat-array index for a state ID."""
        for item in self.state:
            if item.id == identifier:
                return item.index
        raise NumericalModelError(f"Unknown state variable {identifier!r}.")

    def parameter_index(self, name: str) -> int:
        """Return the stable parameter-array index for a parameter name."""
        for item in self.parameters:
            if item.name == name:
                return item.index
        raise NumericalModelError(f"Unknown parameter {name!r}.")

    def manifest(self) -> dict[str, Any]:
        """Return metadata only; executable expressions are intentionally absent."""
        return {
            "schema_version": self.schema_version,
            "model_kind": self.model_kind,
            "name": self.name,
            "state": [item.manifest() for item in self.state],
            "parameters": [item.manifest() for item in self.parameters],
            "rhs_shape": [self.rhs.rows, self.rhs.cols],
            "jacobian_sparsity": self.jacobian_sparsity.manifest(),
            "provenance": self.provenance.manifest(),
            "fingerprint": self.fingerprint,
            "derivative_source": self.derivative_source,
            "numeric_dtype": self.numeric_dtype,
        }

    def write_manifest(self, path) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target


@dataclass(frozen=True)
class MomentDynamicsSpec(NumericalModelSpec):
    """A real moment-dynamics model with a structured state matrix."""

    state_matrix: sp.Matrix | None = None
    hamiltonian: sp.Matrix | None = None
    diffusion: sp.Matrix | None = None

    def __post_init__(self) -> None:
        if self.state_matrix is None:
            raise NumericalModelError("Moment dynamics require a state matrix.")

    @property
    def state_matrix_shape(self) -> tuple[int, int] | None:
        if self.state_matrix is None:
            return None
        return self.state_matrix.rows, self.state_matrix.cols

    @property
    def matrix_semantics(self) -> str:
        return {
            "normal": "normal_second_moment",
            "augmented": "augmented_second_moment",
            "independent": "independent_second_moment",
        }.get(self.provenance.moment_layout, "model_defined")

    @property
    def matrix_symmetry(self) -> str:
        return {
            "normal": "hermitian",
            "augmented": "hermitian",
            "independent": "complex_symmetric",
        }.get(self.provenance.moment_layout, "model_defined")

    @property
    def physical_domain_hint(self) -> str:
        return {
            "normal": "hermitian_psd",
            "augmented": "hermitian_psd",
            "independent": "model_defined",
        }.get(self.provenance.moment_layout, "model_defined")

    @property
    def capabilities(self) -> frozenset[str]:
        values = {
            "rhs",
            "jacobian",
            "parameter_jacobian",
            "jvp",
            "vjp",
            "parameter_jvp",
        }
        if self.state_matrix is not None:
            values.add("state_matrix")
        if self.hamiltonian is not None:
            values.add("hamiltonian")
        if self.diffusion is not None:
            values.add("diffusion")
        return frozenset(values)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def manifest(self) -> dict[str, Any]:
        data = super().manifest()
        normal_indices = [
            item.index
            for item in self.state
            if item.kind in {"diag", "offdiag_re", "offdiag_im"}
        ]
        anomalous_indices = [
            item.index for item in self.state if item.kind.startswith("anomalous_")
        ]
        independent_indices = [
            item.index for item in self.state if item.kind.startswith("independent_")
        ]
        data["moment_layout"] = self.provenance.moment_layout
        data["state_blocks"] = {
            "normal": normal_indices,
            "anomalous": anomalous_indices,
            "independent": independent_indices,
        }
        data["api_version"] = MOMENT_DYNAMICS_API_VERSION
        data["matrix_semantics"] = self.matrix_semantics
        data["matrix_symmetry"] = self.matrix_symmetry
        data["physical_domain_hint"] = self.physical_domain_hint
        data["capabilities"] = {
            "state_matrix": None
            if self.state_matrix is None
            else [self.state_matrix.rows, self.state_matrix.cols],
            "hamiltonian": None
            if self.hamiltonian is None
            else [self.hamiltonian.rows, self.hamiltonian.cols],
            "diffusion": None
            if self.diffusion is None
            else [self.diffusion.rows, self.diffusion.cols],
        }
        return data

    def numpy_source(self) -> str:
        """Generate a batch-aware NumPy reference implementation."""
        return _numpy_module_source(self)

    def write_numpy(self, path) -> Path:
        target = Path(path)
        target.write_text(self.numpy_source(), encoding="utf-8")
        return target

    def compile_numpy(self) -> CompiledMomentDynamics:
        """Compile the generated NumPy reference into a typed callable bundle."""
        namespace: dict[str, Any] = {}
        exec(self.numpy_source(), namespace)
        return CompiledMomentDynamics(
            model_name=self.name,
            fingerprint=self.fingerprint,
            rhs=namespace["rhs"],
            jacobian=namespace["jacobian"],
            parameter_jacobian=namespace["parameter_jacobian"],
            jvp=namespace["jvp"],
            vjp=namespace["vjp"],
            parameter_jvp=namespace["parameter_jvp"],
            state_matrix=namespace.get("state_matrix"),
        )


@dataclass(frozen=True)
class CompiledMomentDynamics:
    """Batch-aware NumPy reference callables compiled from one model spec."""

    model_name: str
    fingerprint: str
    rhs: Any
    jacobian: Any
    parameter_jacobian: Any
    jvp: Any
    vjp: Any
    parameter_jvp: Any
    state_matrix: Any | None


def make_fingerprint(
    *,
    model_kind: str,
    state: tuple[StateVariableSpec, ...],
    parameters: tuple[ParameterSpec, ...],
    expressions: tuple[sp.Expr, ...],
    provenance: DerivationProvenance,
) -> str:
    return _fingerprint_payload(
        model_kind=model_kind,
        state=state,
        parameters=parameters,
        expressions=expressions,
        provenance=provenance,
    )


def matrix_sparsity(matrix: sp.Matrix) -> SparsityPattern:
    return _matrix_sparsity(sp.Matrix(matrix))


def _numpy_print(expr: sp.Expr) -> str:
    return NumPyPrinter().doprint(expr)


def _emit_numpy_function(
    name: str,
    matrix: sp.Matrix,
    shape: tuple[int, ...],
    state: tuple[StateVariableSpec, ...],
    parameters: tuple[ParameterSpec, ...],
    *,
    dtype_name: str = "NUMERIC_DTYPE",
) -> list[str]:
    flat = list(matrix)
    replacements, reduced = sp.cse(flat, symbols=sp.numbered_symbols("t"))
    lines = [f"def {name}(x, p):", "    x, p, _batch_shape = _prepare(x, p)"]
    for item in state:
        lines.append(f"    {item.symbol} = x[..., {item.index}]")
    for item in parameters:
        lines.append(f"    {item.kernel_symbol} = p[..., {item.index}]")
    for symbol, expression in replacements:
        lines.append(f"    {symbol} = {_numpy_print(expression)}")
    if not reduced:
        lines.append(
            f"    return numpy.empty(_batch_shape + {shape!r}, dtype={dtype_name})"
        )
        return lines
    lines.append("    _values = [")
    for expression in reduced:
        lines.append(
            f"        numpy.broadcast_to({_numpy_print(expression)}, _batch_shape),"
        )
    lines.extend(
        [
            "    ]",
            "    _result = numpy.stack(_values, axis=-1).reshape(_batch_shape + "
            f"{shape!r})",
            f"    return _result.astype({dtype_name}, copy=False)",
        ]
    )
    return lines


def _numpy_module_source(spec: MomentDynamicsSpec) -> str:
    if spec.numeric_dtype != "float64":
        raise NumericalModelError(
            "The NumPy reference generator currently supports only float64."
        )
    lines = [
        '"""Generated by qpsgen; NumPy reference model."""',
        "",
        "import numpy",
        "",
        f"MODEL_NAME = {spec.name!r}",
        f"MODEL_FINGERPRINT = {spec.fingerprint!r}",
        f"STATE_SIZE = {spec.state_size}",
        f"PARAMETER_SIZE = {spec.parameter_size}",
        f"STATE_NAMES = {tuple(item.id for item in spec.state)!r}",
        f"PARAMETER_NAMES = {tuple(item.name for item in spec.parameters)!r}",
        f"API_VERSION = {MOMENT_DYNAMICS_API_VERSION!r}",
        f"SCHEMA_VERSION = {spec.schema_version!r}",
        f"STATE_LAYOUT_VERSION = {spec.provenance.state_layout_version!r}",
        f"MOMENT_LAYOUT = {spec.provenance.moment_layout!r}",
        f"MATRIX_SEMANTICS = {spec.matrix_semantics!r}",
        f"MATRIX_SYMMETRY = {spec.matrix_symmetry!r}",
        f"PHYSICAL_DOMAIN_HINT = {spec.physical_domain_hint!r}",
        "NUMERIC_DTYPE = numpy.float64",
        "COMPLEX_DTYPE = numpy.complex128",
        "",
        "def _prepare(x, p):",
        "    x = numpy.asarray(x, dtype=NUMERIC_DTYPE)",
        "    p = numpy.asarray(p, dtype=NUMERIC_DTYPE)",
        "    if x.ndim < 1 or x.shape[-1] != STATE_SIZE:",
        "        raise ValueError(f'x must end with dimension {STATE_SIZE}')",
        "    if p.ndim < 1 or p.shape[-1] != PARAMETER_SIZE:",
        "        raise ValueError(f'p must end with dimension {PARAMETER_SIZE}')",
        "    batch_shape = numpy.broadcast_shapes(x.shape[:-1], p.shape[:-1])",
        "    x = numpy.broadcast_to(x, batch_shape + (STATE_SIZE,))",
        "    p = numpy.broadcast_to(p, batch_shape + (PARAMETER_SIZE,))",
        "    return x, p, batch_shape",
        "",
    ]
    functions = (
        ("rhs", spec.rhs, (spec.state_size,)),
        ("jacobian", spec.jacobian, (spec.state_size, spec.state_size)),
        (
            "parameter_jacobian",
            spec.parameter_jacobian,
            (spec.state_size, spec.parameter_size),
        ),
    )
    for name, matrix, shape in functions:
        lines.extend(
            _emit_numpy_function(name, matrix, shape, spec.state, spec.parameters)
        )
        lines.append("")
    if spec.state_matrix is not None:
        lines.extend(
            _emit_numpy_function(
                "state_matrix",
                spec.state_matrix,
                (spec.state_matrix.rows, spec.state_matrix.cols),
                spec.state,
                spec.parameters,
                dtype_name="COMPLEX_DTYPE",
            )
        )
        lines.append("")
    lines.extend(
        [
            "def jvp(x, p, v):",
            "    v = numpy.asarray(v)",
            "    if v.ndim < 1 or v.shape[-1] != STATE_SIZE:",
            "        raise ValueError(f'v must end with dimension {STATE_SIZE}')",
            "    return numpy.einsum('...ij,...j->...i', jacobian(x, p), v)",
            "",
            "def vjp(x, p, w):",
            "    w = numpy.asarray(w)",
            "    if w.ndim < 1 or w.shape[-1] != STATE_SIZE:",
            "        raise ValueError(f'w must end with dimension {STATE_SIZE}')",
            "    return numpy.einsum('...i,...ij->...j', w, jacobian(x, p))",
            "",
            "def parameter_jvp(x, p, v):",
            "    v = numpy.asarray(v)",
            "    if v.ndim < 1 or v.shape[-1] != PARAMETER_SIZE:",
            "        raise ValueError(f'v must end with dimension {PARAMETER_SIZE}')",
            "    return numpy.einsum('...ij,...j->...i', parameter_jacobian(x, p), v)",
            "",
        ]
    )
    return "\n".join(lines)
