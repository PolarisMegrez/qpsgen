"""Hermitian covariance vectorization, flow, and Jacobian tests."""

import sympy as sp
import pytest

from fpgen import (
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


def test_vectorization_uses_mode_order_then_lexicographic_pairs():
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
        r"\operatorname{Re}(R_{ab})",
        r"\operatorname{Re}(R_{ac})",
        r"\operatorname{Re}(R_{bc})",
        r"\operatorname{Im}(R_{ab})",
        r"\operatorname{Im}(R_{ac})",
        r"\operatorname{Im}(R_{bc})",
    )
    assert result.pair_order == (("a", "b"), ("a", "c"), ("b", "c"))
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
