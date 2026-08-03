"""Higher-order Kramers–Moyal terms and the exact normal -> flux conversion."""

import sympy as sp

from fpgen import (
    LiouvillianTerm,
    LindbladChannel,
    MasterEquation,
    boson_modes,
    derive_kramers_moyal,
)
from fpgen.diffop import NormalOp, diff_multi, normal_to_flux


def test_three_body_loss_keeps_third_order_and_marks_fpe_approximate():
    (a,) = boson_modes("a")
    gamma = sp.symbols("gamma", positive=True)
    master = MasterEquation(
        modes=[a], channels=[LindbladChannel(jump=a**3, rate=gamma)]
    )
    km = derive_kramers_moyal(master, representation="positive-p")
    assert km.order == 3
    third = {mu: k for mu, k in km.coefficients.items() if sum(mu) == 3}
    assert third, "third-order KM terms must be retained in the series"
    fpe = km.truncate(2)
    assert not fpe.is_exact
    assert fpe.original_max_order == 3
    assert fpe.discarded == third


def test_normal_to_flux_is_exact_operator_identity():
    # Operator equality: sum_mu f_mu ∂^mu f  ==  sum_mu ∂^mu [g_mu f]
    # for an arbitrary test function f.
    al, be = sp.symbols("alpha_a beta_a", complex=True)
    variables = (al, be)
    f = sp.exp(al * be) + al**2 * sp.sin(be)
    terms = {
        (1, 2): be,  # variable coefficient at order 3
        (0, 2): al * be,
        (2, 1): al**2,
        (1, 0): be**2,
    }
    op = NormalOp(variables, terms)
    flux = normal_to_flux(op)
    lhs = sum(c * diff_multi(f, variables, mu) for mu, c in terms.items())
    rhs = sum(diff_multi(g * f, variables, mu) for mu, g in flux.items())
    assert sp.simplify(lhs - rhs) == 0


def test_truncation_after_flux_conversion_keeps_leibniz_lower_orders():
    # left = a†, right = a² in positive-P gives normal form
    #   D = β α² - 2βα ∂_β + β ∂_β² - α² ∂_α - 2α + 2α ∂_α∂_β + 2∂_β - ∂_α∂_β².
    # The order-3 term β ∂_β² contributes -2 to the flux coefficient
    # g_{e_beta}; truncating in normal form first would drop it.
    (a,) = boson_modes("a")
    master = MasterEquation(
        modes=[a], extra_terms=[LiouvillianTerm(1, left=a.dag, right=a**2)]
    )
    km = derive_kramers_moyal(master, representation="positive-p")
    assert km.order == 3
    alpha, beta = km.variables
    fpe = km.truncate(2)
    assert not fpe.is_exact
    # Hand-derived: g_{e_beta} = -2 αβ - 2  =>  K_{e_beta} = 2αβ + 2.
    assert sp.simplify(fpe.coefficients[(0, 1)] - (2 * alpha * beta + 2)) == 0
    # The wrong path (drop order-3 normal terms before conversion) would
    # give 2αβ instead.
    assert sp.simplify(fpe.coefficients[(0, 1)] - (2 * alpha * beta)) != 0
    # Discarded record: only ∂_α∂_β² remains above order 2, with
    # K = (-1)^3 (1! 2!) g_{(1,2)}, g_{(1,2)} = -1.
    assert fpe.discarded == {(1, 2): sp.Integer(2)}
