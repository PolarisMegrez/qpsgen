"""Numerical model metadata, derivatives, and structural reduction tests."""

import json

import pytest
import sympy as sp

from fpgen import (
    MODEL_SCHEMA_VERSION,
    STATE_LAYOUT_VERSION,
    LindbladChannel,
    MasterEquation,
    NumericalModelError,
    boson_modes,
    derive_kramers_moyal,
)


def _vdp_dynamics():
    a, b = boson_modes("a", "b")
    parameters = sp.symbols(
        "omega_a omega_b g gamma_a gamma_b Gamma", real=True
    )
    omega_a, omega_b, g, gamma_a, gamma_b, Gamma = parameters
    master = MasterEquation(
        modes=[a, b],
        hamiltonian=(
            omega_a * a.dag * a
            + omega_b * b.dag * b
            + g * (a.dag * b + a * b.dag)
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
        "x0", "x1", "x2", "x3",
    )
    assert tuple(item.id for item in dynamics.state_spec) == (
        "r_diag_0", "r_diag_1", "r_re_0_1", "r_im_0_1",
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
    assert dynamics.parameter_jvp(u) == (
        dynamics.parameter_jacobian() * u
    ).applyfunc(sp.simplify)

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
        lambda expr: sp.diff(
            expr, dynamics.coordinates[0], parameters[-1]
        )
    )
    assert (state_parameter - expected_mixed).applyfunc(sp.simplify) == sp.zeros(n, 1)


def test_numerical_export_requires_explicit_real_parameter_contract():
    a, = boson_modes("a")
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
    candidates = dynamics.find_linear_reductions(retained_dimension=1)
    assert candidates
    assert any(
        item.retained_equations != item.retained_indices for item in candidates
    )
    candidate = next(
        item for item in candidates
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
    assert all(
        sp.simplify(reduced.full_residual[row]) == 0
        for row in eliminated_rows
    )
    conditions = reduced.multiplicity_conditions(4)
    assert len(conditions) == 4
    assert conditions[1] == reduced.derivative(1)


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
        "covariance": [2, 2],
        "hamiltonian": [2, 2],
        "diffusion": [2, 2],
    }
    target = spec.write_manifest(tmp_path / "model.json")
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["fingerprint"] == spec.fingerprint
    assert loaded["provenance"]["state_layout_version"] == STATE_LAYOUT_VERSION
    assert loaded["numeric_dtype"] == "float64"

    numpy = pytest.importorskip("numpy")
    namespace = {}
    exec(spec.numpy_source(), namespace)
    x = numpy.array([
        [1.0, 0.5, 0.1, -0.2],
        [1.2, 0.4, -0.1, 0.3],
    ], dtype=numpy.float32)
    p = numpy.array(
        [1.0, 1.4, 0.2, 0.3, 0.5, 0.1], dtype=numpy.float32
    )
    batched = namespace["rhs"](x, p)
    assert batched.shape == (2, 4)
    assert batched.dtype == numpy.float64
    assert namespace["jacobian"](x, p).shape == (2, 4, 4)
    assert namespace["parameter_jacobian"](x, p).shape == (2, 4, 6)
    assert numpy.allclose(batched[0], namespace["rhs"](x[0], p))
    assert namespace["jvp"](x, p, numpy.ones(4)).shape == (2, 4)


def test_model_fingerprint_includes_derivation_provenance():
    a, = boson_modes("a")
    omega = sp.Symbol("omega", real=True)
    master = MasterEquation(modes=[a], hamiltonian=omega * a.dag * a)

    specs = []
    for representation in ("wigner", "p"):
        langevin = (
            derive_kramers_moyal(master, representation)
            .truncate(2)
            .to_langevin()
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
