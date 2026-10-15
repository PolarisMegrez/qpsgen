"""Left/right multiplication rules for all four representations.

Each elementary term ``X rho`` / ``rho X`` is derived as a Kramers–Moyal
series and checked coefficient by coefficient.  Convention:

    dP/dt = sum_mu (-1)^|mu| / mu! ∂^mu [K_mu P]

so a substitution ``c_mul * z_k + c_der * ∂_{z_l}`` yields
``K_0 = c_mul * z_k`` and ``K_{e_l} = -c_der``.
"""

import pytest
import sympy as sp

from qpsgen import (
    LiouvillianTerm,
    MasterEquation,
    RepresentationError,
    boson_modes,
    derive_kramers_moyal,
)
from qpsgen.representations import normalize_representation, phase_space_variables


def single_mode_master(term):
    (a,) = boson_modes("a")
    return a, MasterEquation(modes=[a], extra_terms=[term])


def km_coeffs(rep, term):
    a, master = single_mode_master(term)
    km = derive_kramers_moyal(master, representation=rep)
    return km.variables, km.coefficients


# (side kind) -> (multiplication part, {derivative axis: coeff}) per family.
# Axes: 0 = alpha, 1 = abar/beta.
CG_RULES = {
    # s: wigner=0, p=+1, q=-1
    ("L", "a"): lambda s: (("alpha", 1), {1: sp.Rational(1 - s, 2)}),
    ("L", "adag"): lambda s: (("bar", 1), {0: -sp.Rational(1 + s, 2)}),
    ("R", "a"): lambda s: (("alpha", 1), {1: -sp.Rational(1 + s, 2)}),
    ("R", "adag"): lambda s: (("bar", 1), {0: sp.Rational(1 - s, 2)}),
}

POSP_RULES = {
    ("L", "a"): (("alpha", 1), {}),
    ("L", "adag"): (("beta", 1), {0: sp.Integer(-1)}),
    ("R", "a"): (("alpha", 1), {1: sp.Integer(-1)}),
    ("R", "adag"): (("beta", 1), {}),
}


def _op(a, side, kind):
    op = a if kind == "a" else a.dag
    return LiouvillianTerm(1, left=op) if side == "L" else LiouvillianTerm(1, right=op)


@pytest.mark.parametrize("rep,s", [("wigner", 0), ("p", 1), ("q", -1)])
@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("kind", ["a", "adag"])
def test_cahill_glauber_rules(rep, s, side, kind):
    (a,) = boson_modes("a")
    variables, coeffs = km_coeffs(rep, _op(a, side, kind))
    alpha, abar = variables
    (fam, _pow), ders = CG_RULES[(side, kind)](s)
    zero = (0, 0)
    expected = {zero: alpha if fam == "alpha" else abar}
    for axis, c in ders.items():
        if c == 0:
            continue  # vanishing derivative coefficients are dropped
        mu = [0, 0]
        mu[axis] = 1
        expected[tuple(mu)] = -c
    assert coeffs == expected


@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("kind", ["a", "adag"])
def test_positive_p_rules(side, kind):
    (a,) = boson_modes("a")
    variables, coeffs = km_coeffs("positive-p", _op(a, side, kind))
    alpha, beta = variables
    (fam, _pow), ders = POSP_RULES[(side, kind)]
    expected = {(0, 0): alpha if fam == "alpha" else beta}
    for axis, c in ders.items():
        mu = [0, 0]
        mu[axis] = 1
        expected[tuple(mu)] = -c
    assert coeffs == expected


def test_symbolic_complex_coefficients():
    (a,) = boson_modes("a")
    lam = sp.symbols("lambda")
    term = LiouvillianTerm(sp.I * lam, left=a)
    _, coeffs = km_coeffs("wigner", term)
    alpha = sp.Symbol("alpha_a", complex=True)
    assert coeffs[(0, 0)] == sp.I * lam * alpha
    assert coeffs[(0, 1)] == -sp.I * lam / 2


def test_operator_powers_and_dagger_in_term():
    (a,) = boson_modes("a")
    # a†² rho in P representation: (abar - ∂_alpha)^2
    term = LiouvillianTerm(1, left=a.dag**2)
    variables, coeffs = km_coeffs("p", term)
    alpha, abar = variables
    assert coeffs[(0, 0)] == abar**2
    assert coeffs[(1, 0)] == 2 * abar  # K = -(-2 abar)
    assert coeffs[(2, 0)] == 2  # (-1)^2 * 2! * 1


def test_multimode_global_ordering():
    a, b = boson_modes("a", "b")
    # a† rho b in Wigner: (abar_a - ½∂_{alpha_a}) (alpha_b - ½∂_{abar_b})
    master = MasterEquation(
        modes=[a, b], extra_terms=[LiouvillianTerm(1, left=a.dag, right=b)]
    )
    km = derive_kramers_moyal(master, representation="wigner")
    alpha_a, alpha_b, abar_a, abar_b = km.variables
    c = km.coefficients
    assert c[(0, 0, 0, 0)] == abar_a * alpha_b
    assert c[(0, 0, 0, 1)] == abar_a / 2  # K = -(-abar_a/2 ∂_{abar_b})
    assert c[(1, 0, 0, 0)] == alpha_b / 2  # K = -(-alpha_b/2 ∂_{alpha_a})
    assert c[(1, 0, 0, 1)] == sp.Rational(1, 4)  # (+1) * (1/2)(1/2)
    assert len(c) == 4


def test_representation_aliases_and_unknown():
    assert normalize_representation("Positive P") == "positive-p"
    assert normalize_representation("Glauber-Sudarshan") == "p"
    assert normalize_representation("Husimi Q") == "q"
    assert normalize_representation("Wigner") == "wigner"
    with pytest.raises(RepresentationError):
        normalize_representation("PositiveP+Wigner")
    with pytest.raises(RepresentationError):
        normalize_representation("bogus")


def test_phase_space_variable_orders():
    assert [v.name for v in phase_space_variables("wigner", ("a", "b"))] == [
        "alpha_a",
        "alpha_b",
        "abar_a",
        "abar_b",
    ]
    assert [v.name for v in phase_space_variables("positive-p", ("a", "b"))] == [
        "alpha_a",
        "alpha_b",
        "beta_a",
        "beta_b",
    ]
