"""Numerical model metadata, derivatives, and structural reduction tests."""

import json

import pytest
import sympy as sp

from qpsgen import (
    MODEL_SCHEMA_VERSION,
    MOMENT_DYNAMICS_API_VERSION,
    REDUCTION_API_VERSION,
    STATE_LAYOUT_VERSION,
    CovarianceDynamics,
    LindbladChannel,
    LinearReductionPlan,
    MasterEquation,
    NumericalModelError,
    ReductionError,
    StateVariableSpec,
    boson_modes,
    derive_kramers_moyal,
)


def _vdp_dynamics():
    a, b = boson_modes("a", "b")
    parameters = sp.symbols("omega_a omega_b g gamma_a gamma_b Gamma", real=True)
    omega_a, omega_b, g, gamma_a, gamma_b, Gamma = parameters
    master = MasterEquation(
        modes=[a, b],
        hamiltonian=(
            omega_a * a.dag * a + omega_b * b.dag * b + g * (a.dag * b + a * b.dag)
        ),
        channels=[
            LindbladChannel(a.dag, gamma_a),
            LindbladChannel(a**2, Gamma),
            LindbladChannel(b, gamma_b),
        ],
    )
    langevin = derive_kramers_moyal(master, "wigner").truncate(2).to_langevin()
    return langevin.to_covariance_dynamics(parameters=parameters), parameters


def test_covariance_dynamics_separates_machine_and_display_names():
    dynamics, _ = _vdp_dynamics()
    assert dynamics.state_layout_version == STATE_LAYOUT_VERSION
    assert tuple(str(value) for value in dynamics.coordinates) == (
        "x0",
        "x1",
        "x2",
        "x3",
    )
    assert tuple(item.id for item in dynamics.state_spec) == (
        "r_diag_0",
        "r_diag_1",
        "r_re_0_1",
        "r_im_0_1",
    )
    assert tuple(item.latex_name for item in dynamics.state_spec) == (
        "R_{aa}",
        "R_{bb}",
        r"\operatorname{Re}(R_{ab})",
        r"\operatorname{Im}(R_{ab})",
    )


def test_exact_state_parameter_derivatives_and_third_directional_contraction():
    dynamics, parameters = _vdp_dynamics()
    n = len(dynamics.coordinates)
    m = len(parameters)
    v = sp.Matrix(sp.symbols(f"v0:{n}", real=True))
    u = sp.Matrix(sp.symbols(f"u0:{m}", real=True))

    assert dynamics.jvp(v) == (dynamics.jacobian() * v).applyfunc(sp.simplify)
    assert dynamics.vjp(v) == (dynamics.jacobian().T * v).applyfunc(sp.simplify)
    assert dynamics.parameter_jvp(u) == (dynamics.parameter_jacobian() * u).applyfunc(
        sp.simplify
    )

    e0 = sp.eye(n)[:, 0]
    contracted = dynamics.directional(3, state_directions=(e0, e0, e0))
    expected = dynamics.rhs.applyfunc(
        lambda expr: sp.diff(expr, dynamics.coordinates[0], 3)
    )
    assert (contracted - expected).applyfunc(sp.simplify) == sp.zeros(n, 1)

    state_parameter = dynamics.directional(
        2,
        state_directions=(e0,),
        parameter_directions=(sp.eye(m)[:, m - 1],),
    )
    expected_mixed = dynamics.rhs.applyfunc(
        lambda expr: sp.diff(expr, dynamics.coordinates[0], parameters[-1])
    )
    assert (state_parameter - expected_mixed).applyfunc(sp.simplify) == sp.zeros(n, 1)


def test_numerical_export_requires_explicit_real_parameter_contract():
    (a,) = boson_modes("a")
    omega = sp.Symbol("omega")
    master = MasterEquation(modes=[a], hamiltonian=omega * a.dag * a)
    langevin = derive_kramers_moyal(master, "wigner").truncate(2).to_langevin()

    inferred = langevin.to_covariance_dynamics()
    with pytest.raises(NumericalModelError, match="explicit parameter order"):
        inferred.to_model_spec(name="inferred")
    with pytest.raises(NumericalModelError, match="real-domain assumption"):
        langevin.to_covariance_dynamics(parameters=[omega])

    explicit = langevin.to_covariance_dynamics(
        parameters=[omega], parameter_domains={omega: "real"}
    )
    assert explicit.parameter_spec[0].domain == "real"
    assert explicit.rhs == sp.zeros(1, 1)


def test_vdp_linear_reduction_candidate_and_fraction_free_quartic():
    dynamics, _ = _vdp_dynamics()
    search = dynamics.search_linear_reductions(retained_dimension=1)
    candidates = search.candidates
    assert candidates
    assert search.coverage == "exhaustive"
    assert search.regular_branches_only
    assert search.complete
    assert any(item.retained_equations != item.retained_indices for item in candidates)
    candidate = next(
        item
        for item in candidates
        if item.retained_ids == ("r_diag_0",)
        and item.retained_equations == item.retained_indices
    )
    assert candidate.structural_rank == candidate.required_rank == 3
    assert candidate.A_sparsity.nnz == 7
    assert candidate.reduced_degree == 4
    assert candidate.denominator_operation_count is not None

    plan = dynamics.linear_reduce(candidate=candidate)
    assert plan.linearity_residual == sp.zeros(3, 1)
    assert plan.rank_condition == "rank(A) == 3"
    assert plan.singular_branch == "rank(A) < 3"

    reduced = plan.materialize()
    q = plan.retained_symbols[0]
    assert sp.Poly(reduced.numerators[0], q).degree() == 4
    assert reduced.denominators[0] != 0
    assert reduced.regularity_determinant != 0
    eliminated_rows = candidate.eliminated_equations
    assert all(sp.simplify(reduced.full_residual[row]) == 0 for row in eliminated_rows)
    conditions = reduced.multiplicity_conditions(4)
    assert len(conditions) == 4
    assert conditions[1] == reduced.derivative(1)


def test_linear_reduction_search_reports_real_work_limits():
    dynamics, _ = _vdp_dynamics()
    limited = dynamics.search_linear_reductions(
        retained_dimension=1,
        partition_limit=2,
    )
    assert limited.coverage == "budget_limited"
    assert limited.examined_equation_partitions == 2
    assert limited.truncation_reasons == ("partition_limit",)
    assert not limited.complete

    prefix = dynamics.search_linear_reductions(
        retained_dimension=1,
        return_limit=1,
    )
    assert prefix.coverage == "exhaustive"
    assert prefix.returned_prefix
    assert prefix.truncation_reasons == ("return_limit",)
    assert len(prefix.candidates) == 1
    assert prefix.manifest()["api_version"] == REDUCTION_API_VERSION
    assert not prefix.manifest()["complete"]
    assert prefix.manifest()["returned_count"] == 1

    retained = dynamics.search_linear_reductions(
        retained_dimension=1,
        retained_ids=("r_diag_0",),
    )
    assert retained.requested_retained_ids == ("r_diag_0",)
    assert {item.retained_ids for item in retained.candidates} == {("r_diag_0",)}

    with pytest.raises(ReductionError, match="Unknown retained state ID"):
        dynamics.search_linear_reductions(
            retained_dimension=1,
            retained_ids=("missing",),
        )


def _synthetic_dynamics(rhs):
    """Minimal synthetic flow over the symbols appearing in ``rhs``."""
    coordinates = tuple(sorted(rhs.free_symbols, key=lambda symbol: symbol.name))
    state_spec = tuple(
        StateVariableSpec(
            id=f"s{i}",
            index=i,
            symbol=symbol,
            latex_name=str(symbol),
            kind="synthetic",
            matrix_index=(i, i),
            mode_names=("a", "a"),
        )
        for i, symbol in enumerate(coordinates)
    )
    n = len(coordinates)
    return CovarianceDynamics(
        representation="wigner",
        modes=(),
        state_spec=state_spec,
        covariance=None,
        pair_indices=(),
        pair_order=(),
        hamiltonian=sp.zeros(n),
        diffusion=sp.zeros(n),
        rhs_matrix=rhs,
        rhs=rhs,
        langevin=None,
    )


def test_linear_reduction_candidate_chart_id_addresses_partitions():
    dynamics, _ = _vdp_dynamics()
    search = dynamics.search_linear_reductions(retained_dimension=1)
    candidate = next(
        item
        for item in search.candidates
        if item.retained_ids == ("r_diag_0",)
        and item.retained_equations == item.retained_indices
    )
    assert candidate.chart_id == "ret:r_diag_0|eq:0"
    json.dumps([item.chart_id for item in search.candidates])

    x0, x1 = sp.symbols("x0 x1", real=True)
    twin = _synthetic_dynamics(sp.Matrix([-x0 - x1, -x0 - x1]))
    twin_search = twin.search_linear_reductions(retained_dimension=1)
    chart_ids = [item.chart_id for item in twin_search.candidates]
    assert len(chart_ids) == len(set(chart_ids)) == 4
    per_state = {}
    for item in twin_search.candidates:
        per_state.setdefault(item.retained_ids, set()).add(item.chart_id)
    assert per_state == {
        ("s0",): {"ret:s0|eq:0", "ret:s0|eq:1"},
        ("s1",): {"ret:s1|eq:0", "ret:s1|eq:1"},
    }


def test_linear_reduction_search_records_rejected_partition_details():
    dynamics, _ = _vdp_dynamics()
    search = dynamics.search_linear_reductions(retained_dimension=1)
    assert len(search.rejected_partitions) == 12
    assert {entry.reason for entry in search.rejected_partitions} == {
        "non_affine_eliminated_block"
    }
    ids = tuple(item.id for item in dynamics.state_spec)
    for entry in search.rejected_partitions:
        assert entry.retained_ids == tuple(ids[i] for i in entry.retained_indices)
        assert len(entry.retained_equations) == 1
    manifest = search.manifest()
    assert manifest["rejected_reason_counts"] == {"non_affine_eliminated_block": 12}
    assert manifest["rejected_partition_count"] == len(search.rejected_partitions)
    json.dumps(manifest)


def test_linear_reduction_oversized_eliminated_skips_are_counted():
    coordinates = sp.symbols("x0:5", real=True)
    dynamics = _synthetic_dynamics(sp.Matrix([-symbol for symbol in coordinates]))
    search = dynamics.search_linear_reductions(retained_dimension=1)
    assert len(search.candidates) == 5
    assert all(
        len(candidate.eliminated_indices) == 4 for candidate in search.candidates
    )
    assert search.materialization_attempts == 0
    assert search.materialization_skipped == 0
    assert search.materialization_skipped_oversized == 5
    assert all(candidate.reduced_degree is None for candidate in search.candidates)
    assert search.rejected_reason_counts == (("structurally_rank_deficient", 20),)
    assert len(search.rejected_partitions) == 20
    manifest = search.manifest()
    assert manifest["materialization_skipped_oversized"] == 5
    assert manifest["rejected_partition_count"] == 20
    json.dumps(manifest)


def test_materialization_failure_is_diagnostic_not_rejection(monkeypatch):
    dynamics, _ = _vdp_dynamics()

    def _raise(self, *, method="fraction_free"):
        raise ReductionError("forced for test")

    monkeypatch.setattr(LinearReductionPlan, "materialize", _raise)
    search = dynamics.search_linear_reductions(retained_dimension=1)
    assert search.candidates
    assert search.materialization_attempts > 0
    assert "materialization_failed" not in dict(search.rejected_reason_counts)
    assert len(search.materialization_failures) == search.materialization_attempts
    assert all(candidate.reduced_degree is None for candidate in search.candidates)
    for failure in search.materialization_failures:
        assert failure.chart_id.startswith("ret:")
        assert "forced for test" in failure.error
    manifest = search.manifest()
    assert manifest["materialization_failure_count"] == len(
        search.materialization_failures
    )
    json.dumps(manifest)


def test_moment_model_spec_manifest_and_numpy_batch_reference(tmp_path):
    dynamics, parameters = _vdp_dynamics()
    spec = dynamics.to_model_spec(name="two_mode_vdp")
    assert spec.schema_version == MODEL_SCHEMA_VERSION
    assert spec.model_kind == "moment_dynamics"
    assert spec.fingerprint
    assert tuple(item.id for item in spec.parameters) == tuple(
        f"p{i}" for i in range(len(parameters))
    )
    assert spec.jacobian_sparsity.nnz > 0

    manifest = spec.manifest()
    assert "rhs" not in manifest
    assert manifest["capabilities"] == {
        "state_matrix": [2, 2],
        "hamiltonian": [2, 2],
        "diffusion": [2, 2],
    }
    assert manifest["moment_layout"] == "normal"
    assert manifest["api_version"] == MOMENT_DYNAMICS_API_VERSION
    assert manifest["matrix_semantics"] == "normal_second_moment"
    assert manifest["matrix_symmetry"] == "hermitian"
    assert manifest["physical_domain_hint"] == "hermitian_psd"
    assert manifest["state_blocks"] == {
        "normal": [0, 1, 2, 3],
        "anomalous": [],
        "independent": [],
    }
    target = spec.write_manifest(tmp_path / "model.json")
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["fingerprint"] == spec.fingerprint
    assert loaded["provenance"]["state_layout_version"] == STATE_LAYOUT_VERSION
    assert loaded["numeric_dtype"] == "float64"
    assert spec.state_index("r_re_0_1") == 2
    assert spec.parameter_index("Gamma") == 5
    assert spec.supports("state_matrix")

    numpy = pytest.importorskip("numpy")
    namespace = {}
    exec(spec.numpy_source(), namespace)
    x = numpy.array(
        [
            [1.0, 0.5, 0.1, -0.2],
            [1.2, 0.4, -0.1, 0.3],
        ],
        dtype=numpy.float32,
    )
    p = numpy.array([1.0, 1.4, 0.2, 0.3, 0.5, 0.1], dtype=numpy.float32)
    batched = namespace["rhs"](x, p)
    assert batched.shape == (2, 4)
    assert batched.dtype == numpy.float64
    assert namespace["jacobian"](x, p).shape == (2, 4, 4)
    assert namespace["parameter_jacobian"](x, p).shape == (2, 4, 6)
    assert numpy.allclose(batched[0], namespace["rhs"](x[0], p))
    assert namespace["jvp"](x, p, numpy.ones(4)).shape == (2, 4)
    state_matrix = namespace["state_matrix"](x, p)
    assert state_matrix.shape == (2, 2, 2)
    assert state_matrix.dtype == numpy.complex128
    assert numpy.allclose(
        state_matrix,
        state_matrix.swapaxes(-1, -2).conj(),
    )
    compiled = spec.compile_numpy()
    assert compiled.fingerprint == spec.fingerprint
    assert numpy.allclose(compiled.rhs(x, p), batched)
    assert numpy.allclose(compiled.state_matrix(x, p), state_matrix)


def test_model_fingerprint_includes_derivation_provenance():
    (a,) = boson_modes("a")
    omega = sp.Symbol("omega", real=True)
    master = MasterEquation(modes=[a], hamiltonian=omega * a.dag * a)

    specs = []
    for representation in ("wigner", "p"):
        langevin = (
            derive_kramers_moyal(master, representation).truncate(2).to_langevin()
        )
        specs.append(
            langevin.to_covariance_dynamics(parameters=[omega]).to_model_spec(
                name="same_rhs"
            )
        )

    assert specs[0].rhs == specs[1].rhs == sp.zeros(1, 1)
    assert specs[0].provenance.representation == "wigner"
    assert specs[1].provenance.representation == "p"
    assert specs[0].fingerprint != specs[1].fingerprint


def test_augmented_model_spec_exports_layout_and_numpy_reference():
    (a,) = boson_modes("a")
    omega, epsilon, kappa = sp.symbols("omega epsilon kappa", real=True)
    master = MasterEquation(
        modes=[a],
        hamiltonian=(omega * a.dag * a + epsilon / 2 * (a.dag**2 + a**2)),
        channels=[LindbladChannel(a, kappa)],
    )
    dynamics = (
        derive_kramers_moyal(master, "wigner")
        .truncate(2)
        .to_langevin()
        .to_second_moment_dynamics(parameters=(omega, epsilon, kappa), layout="auto")
    )
    spec = dynamics.to_model_spec(name="dpo_augmented")
    manifest = spec.manifest()
    assert manifest["moment_layout"] == "augmented"
    assert manifest["state_blocks"] == {
        "normal": [0],
        "anomalous": [1, 2],
        "independent": [],
    }
    assert manifest["provenance"]["moment_closure"] == "exact"
    assert manifest["provenance"]["layout_selection"] == "auto_verified"

    numpy = pytest.importorskip("numpy")
    namespace = {}
    exec(spec.numpy_source(), namespace)
    x = numpy.array([1.0, 0.2, -0.3])
    p = numpy.array([1.1, 0.4, 0.5])
    assert namespace["rhs"](x, p).shape == (3,)
    assert namespace["jacobian"](x, p).shape == (3, 3)
    state_matrix = namespace["state_matrix"](x, p)
    assert state_matrix.shape == (2, 2)
    assert state_matrix.dtype == numpy.complex128
    assert numpy.allclose(state_matrix, state_matrix.T.conj())
