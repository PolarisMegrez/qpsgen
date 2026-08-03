"""Hermitian covariance vectorization, flow, and Jacobian tests."""

import sympy as sp
import pytest

from fpgen import (
    AUGMENTED_SECOND_MOMENT_LAYOUT_VERSION,
    CovarianceClosureError,
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
    result = langevin.to_covariance_jacobian()
    R_aa, R_bb, re_ab, im_ab = result.coordinates
    expected = sp.Matrix([
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
    ])
    assert (result.jacobian - expected).applyfunc(sp.simplify) == sp.zeros(4)


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
        hamiltonian=(
            omega * a.dag * a + epsilon / 2 * (a.dag**2 + a**2)
        ),
        channels=[LindbladChannel(a, kappa)],
    )
    langevin = derive_kramers_moyal(master, "wigner").truncate(2).to_langevin()
    expected_h = sp.Matrix([
        [omega - sp.I * kappa / 2, epsilon],
        [-epsilon, -omega - sp.I * kappa / 2],
    ])
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
    R_aa, re_C_aa, im_C_aa = augmented.coordinates
    assert augmented.state_layout_version == AUGMENTED_SECOND_MOMENT_LAYOUT_VERSION
    assert tuple(item.id for item in augmented.state_spec) == (
        "r_diag_0", "c_re_0_0", "c_im_0_0",
    )
    expected_flow = sp.Matrix([
        -kappa * R_aa - 2 * epsilon * im_C_aa + kappa / 2,
        -kappa * re_C_aa + 2 * omega * im_C_aa,
        -2 * epsilon * R_aa - 2 * omega * re_C_aa - kappa * im_C_aa,
    ])
    assert (augmented.rhs - expected_flow).applyfunc(sp.simplify) == sp.zeros(3, 1)
    expected_jacobian = sp.Matrix([
        [-kappa, 0, -2 * epsilon],
        [0, -kappa, 2 * omega],
        [-2 * epsilon, -2 * omega, -kappa],
    ])
    assert augmented.jacobian() == expected_jacobian

    reduced = augmented.linear_reduce(
        order_parameters=["r_diag_0"]
    ).materialize()
    assert sp.Poly(reduced.numerators[0], R_aa).degree() == 1
    assert sp.factor(reduced.regularity_determinant) == kappa**2 + 4 * omega**2


def test_positive_p_has_no_hermitian_partner_covariance():
    (a,) = boson_modes("a")
    master = MasterEquation(modes=[a], hamiltonian=a.dag * a)
    langevin = (
        derive_kramers_moyal(master, "positive-p")
        .truncate(2)
        .to_langevin()
    )
    with pytest.raises(CovarianceClosureError, match="positive-P"):
        langevin.to_covariance_jacobian()
