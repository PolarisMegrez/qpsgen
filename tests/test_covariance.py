"""Hermitian covariance vectorization, flow, and Jacobian tests."""

import pytest
import sympy as sp

from qpsgen import (
    AUGMENTED_SECOND_MOMENT_LAYOUT_VERSION,
    INDEPENDENT_SECOND_MOMENT_LAYOUT_VERSION,
    AugmentedSecondMomentDynamics,
    CovarianceClosureError,
    CovarianceDynamics,
    IndependentSecondMomentDynamics,
    LindbladChannel,
    MasterEquation,
    boson_modes,
    derive_kramers_moyal,
)


def test_van_der_pol_covariance_jacobian_matches_closed_equations():
    a, b = boson_modes("a", "b")
    omega_a, omega_b, g, gamma_a, gamma_b, Gamma = sp.symbols(
        "omega_a omega_b g gamma_a gamma_b Gamma", real=True
    )
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
    result = langevin.to_covariance_jacobian()
    R_aa, R_bb, re_ab, im_ab = result.coordinates
    expected = sp.Matrix(
        [
            [gamma_a - 4 * Gamma * (R_aa - 1), 0, 0, -2 * g],
            [0, -gamma_b, 0, 2 * g],
            [
                -Gamma * re_ab,
                0,
                (gamma_a - gamma_b) / 2 - Gamma * (R_aa - 1),
                omega_a - omega_b,
            ],
            [
                g - Gamma * im_ab,
                -g,
                omega_b - omega_a,
                (gamma_a - gamma_b) / 2 - Gamma * (R_aa - 1),
            ],
        ]
    )
    assert (result.jacobian - expected).applyfunc(sp.simplify) == sp.zeros(4)

    auto = langevin.to_second_moment_dynamics(
        parameters=(omega_a, omega_b, g, gamma_a, gamma_b, Gamma),
        layout="auto",
        closure="factorized_bilinear",
    )
    assert type(auto) is CovarianceDynamics
    assert auto.layout_kind == "normal"
    assert auto.layout_decision.normal_autonomous
    assert auto.layout_decision.requested_layout == "auto"
    assert auto.layout_decision.projection_residual == sp.zeros(2)


def test_vectorization_uses_declared_index_upper_triangle_order():
    c, a, b = boson_modes("c", "a", "b")
    master = MasterEquation(modes=[c, a, b])
    result = (
        derive_kramers_moyal(master, "wigner")
        .truncate(2)
        .to_langevin()
        .to_covariance_jacobian()
    )
    assert tuple(str(value) for value in result.coordinates) == (
        "R_{cc}",
        "R_{aa}",
        "R_{bb}",
        r"\operatorname{Re}(R_{ca})",
        r"\operatorname{Re}(R_{cb})",
        r"\operatorname{Re}(R_{ab})",
        r"\operatorname{Im}(R_{ca})",
        r"\operatorname{Im}(R_{cb})",
        r"\operatorname{Im}(R_{ab})",
    )
    assert result.pair_indices == ((0, 1), (0, 2), (1, 2))
    assert result.pair_order == (("c", "a"), ("c", "b"), ("a", "b"))
    assert tuple(item.id for item in result.state_spec) == (
        "r_diag_0",
        "r_diag_1",
        "r_diag_2",
        "r_re_0_1",
        "r_re_0_2",
        "r_re_1_2",
        "r_im_0_1",
        "r_im_0_2",
        "r_im_1_2",
    )
    assert result.jacobian == sp.zeros(9)


def test_anomalous_hamiltonian_requires_augmented_covariance():
    (a,) = boson_modes("a")
    omega, epsilon, kappa = sp.symbols("omega epsilon kappa", real=True)
    master = MasterEquation(
        modes=[a],
        hamiltonian=(omega * a.dag * a + epsilon / 2 * (a.dag**2 + a**2)),
        channels=[LindbladChannel(a, kappa)],
    )
    langevin = derive_kramers_moyal(master, "wigner").truncate(2).to_langevin()
    expected_h = sp.Matrix(
        [
            [omega - sp.I * kappa / 2, epsilon],
            [-epsilon, -omega - sp.I * kappa / 2],
        ]
    )
    assert (langevin.hamiltonian - expected_h).applyfunc(sp.simplify) == sp.zeros(2)
    assert langevin.anomalous_hamiltonian == sp.Matrix([[epsilon]])
    assert langevin.normal_correlation == sp.Matrix([[kappa / 2]])
    assert langevin.anomalous_correlation == sp.zeros(1)
    z = sp.Matrix(langevin.variables)
    assert (langevin.drift + sp.I * langevin.hamiltonian * z).applyfunc(
        sp.simplify
    ) == sp.zeros(2, 1)
    with pytest.raises(CovarianceClosureError, match="anomalous Hamiltonian"):
        langevin.to_covariance_jacobian()

    augmented = langevin.to_augmented_second_moment_dynamics(
        parameters=(omega, epsilon, kappa)
    )
    auto = langevin.to_second_moment_dynamics(
        parameters=(omega, epsilon, kappa), layout="auto"
    )
    assert isinstance(auto, AugmentedSecondMomentDynamics)
    assert auto.layout_decision.selected_layout == "augmented"
    assert auto.layout_decision.anomalous_dependencies == ("x2",)
    R_aa, re_C_aa, im_C_aa = augmented.coordinates
    assert augmented.state_layout_version == AUGMENTED_SECOND_MOMENT_LAYOUT_VERSION
    assert tuple(item.id for item in augmented.state_spec) == (
        "r_diag_0",
        "c_re_0_0",
        "c_im_0_0",
    )
    assert augmented.normal_moment == sp.Matrix([[R_aa]])
    assert augmented.anomalous_moment == sp.Matrix([[re_C_aa + sp.I * im_C_aa]])
    assert augmented.moment_matrix == sp.Matrix(
        [
            [R_aa, re_C_aa + sp.I * im_C_aa],
            [re_C_aa - sp.I * im_C_aa, R_aa],
        ]
    )
    blocks = langevin.to_augmented_blocks(real_parameters=(omega, epsilon, kappa))
    assert blocks.physical_hamiltonian == sp.Matrix([[omega - sp.I * kappa / 2]])
    assert blocks.pairing_hamiltonian == sp.Matrix([[epsilon]])
    assert blocks.augmented_diffusion == sp.diag(kappa / 2, kappa / 2)
    assert blocks.augmented_diffusion == (
        blocks.transpose_diffusion * blocks.exchange_matrix
    )
    H = augmented.physical_hamiltonian
    V = augmented.pairing_hamiltonian
    R = augmented.normal_moment
    C = augmented.anomalous_moment
    D = augmented.normal_diffusion
    M = augmented.anomalous_diffusion
    expected_R_flow = (
        -sp.I * (H * R + V * C.conjugate())
        + sp.I * (R * H.conjugate().T + C * V.conjugate().T)
        + D
    ).applyfunc(sp.simplify)
    expected_C_flow = (
        -sp.I * (H * C + C * H.T + V * R.conjugate() + R * V.T) + M
    ).applyfunc(sp.simplify)
    assert augmented.normal_rhs == expected_R_flow
    assert augmented.anomalous_rhs == expected_C_flow
    expected_flow = sp.Matrix(
        [
            -kappa * R_aa - 2 * epsilon * im_C_aa + kappa / 2,
            -kappa * re_C_aa + 2 * omega * im_C_aa,
            -2 * epsilon * R_aa - 2 * omega * re_C_aa - kappa * im_C_aa,
        ]
    )
    assert (augmented.rhs - expected_flow).applyfunc(sp.simplify) == sp.zeros(3, 1)
    expected_jacobian = sp.Matrix(
        [
            [-kappa, 0, -2 * epsilon],
            [0, -kappa, 2 * omega],
            [-2 * epsilon, -2 * omega, -kappa],
        ]
    )
    assert augmented.jacobian() == expected_jacobian

    reduced = augmented.linear_reduce(order_parameters=["r_diag_0"]).materialize()
    assert sp.Poly(reduced.numerators[0], R_aa).degree() == 1
    assert sp.factor(reduced.regularity_determinant) == kappa**2 + 4 * omega**2


def test_positive_p_has_no_hermitian_partner_covariance():
    (a,) = boson_modes("a")
    master = MasterEquation(modes=[a], hamiltonian=a.dag * a)
    langevin = derive_kramers_moyal(master, "positive-p").truncate(2).to_langevin()
    with pytest.raises(CovarianceClosureError, match="positive-P"):
        langevin.to_covariance_jacobian()


def test_nonlinear_anomalous_dependency_prevents_normal_fallback():
    (a,) = boson_modes("a")
    lam = sp.Symbol("lambda", real=True)
    master = MasterEquation(
        modes=[a],
        hamiltonian=lam * (a.dag**3 * a + a.dag * a**3),
    )
    langevin = derive_kramers_moyal(master, "wigner").truncate(2).to_langevin()
    with pytest.raises(CovarianceClosureError, match="state-independent"):
        langevin.to_second_moment_dynamics(
            parameters=(lam,), layout="auto", closure="exact"
        )

    dynamics = langevin.to_second_moment_dynamics(
        parameters=(lam,),
        layout="auto",
        closure="factorized_bilinear",
    )
    assert isinstance(dynamics, AugmentedSecondMomentDynamics)
    R, c_re, c_im = dynamics.coordinates
    assert dynamics.normal_rhs == sp.Matrix([2 * lam * c_im * (3 - 2 * R)])
    assert dynamics.layout_decision.anomalous_dependencies == ("x2",)
    assert dynamics.provenance.moment_closure == "factorized_bilinear"
    with pytest.raises(CovarianceClosureError, match="prevents a closed normal"):
        langevin.to_second_moment_dynamics(
            parameters=(lam,),
            layout="normal",
            closure="factorized_bilinear",
        )


def test_positive_p_uses_independent_complex_symmetric_second_moments():
    (a,) = boson_modes("a")
    omega, epsilon = sp.symbols("omega epsilon", real=True)
    master = MasterEquation(
        modes=[a],
        hamiltonian=(omega * a.dag * a + epsilon / 2 * (a.dag**2 + a**2)),
    )
    langevin = derive_kramers_moyal(master, "positive-p").truncate(2).to_langevin()
    dynamics = langevin.to_second_moment_dynamics(
        parameters=(omega, epsilon), layout="auto"
    )
    assert isinstance(dynamics, IndependentSecondMomentDynamics)
    assert dynamics.layout_kind == "independent"
    assert dynamics.state_layout_version == INDEPENDENT_SECOND_MOMENT_LAYOUT_VERSION
    assert tuple(item.id for item in dynamics.state_spec) == (
        "s_re_0_0",
        "s_re_0_1",
        "s_re_1_1",
        "s_im_0_0",
        "s_im_0_1",
        "s_im_1_1",
    )
    assert dynamics.moment_matrix == dynamics.moment_matrix.T
    assert dynamics.rhs_matrix == (
        -sp.I
        * (
            dynamics.hamiltonian * dynamics.moment_matrix
            + dynamics.moment_matrix * dynamics.hamiltonian.T
        )
        + dynamics.diffusion
    ).applyfunc(sp.simplify)
    spec = dynamics.to_model_spec(name="positive_p_parametric")
    manifest = spec.manifest()
    assert manifest["moment_layout"] == "independent"
    assert manifest["matrix_semantics"] == "independent_second_moment"
    assert manifest["matrix_symmetry"] == "complex_symmetric"
    assert manifest["physical_domain_hint"] == "model_defined"
    assert manifest["provenance"]["layout_selection"] == "auto_verified"
    assert manifest["state_blocks"] == {
        "normal": [],
        "anomalous": [],
        "independent": list(range(6)),
    }
    numpy = pytest.importorskip("numpy")
    compiled = spec.compile_numpy()
    state = numpy.asarray([1.0, 0.2, 0.4, -0.1, 0.3, -0.2])
    params = numpy.asarray([1.1, 0.4])
    matrix = compiled.state_matrix(state, params)
    assert matrix.shape == (2, 2)
    assert matrix.dtype == numpy.complex128
    assert numpy.allclose(matrix, matrix.T)
    with pytest.raises(CovarianceClosureError, match="independent"):
        langevin.to_second_moment_dynamics(
            parameters=(omega, epsilon), layout="augmented"
        )
