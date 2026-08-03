"""Regression tests for notebook LaTeX rendering."""

import sympy as sp

from fpgen.results import KramersMoyalSeries


def test_kramers_moyal_latex_groups_nested_variable_subscripts():
    alpha_a, beta_a = sp.symbols("alpha_a beta_a")
    km = KramersMoyalSeries(
        representation="positive-p",
        variables=(alpha_a, beta_a),
        coefficients={
            (1, 0): alpha_a,
            (0, 1): beta_a,
            (1, 2): alpha_a * beta_a,
        },
    )

    latex = km.to_latex()

    assert r"K_{\partial_{\alpha_{a}}}" in latex
    assert r"K_{\partial_{\beta_{a}}}" in latex
    assert (
        r"K_{\partial_{\alpha_{a}} \partial_{\beta_{a}}^{2}}" in latex
    )
    assert "∂_" not in latex


def test_plain_repr_keeps_readable_derivative_labels():
    (alpha_a,) = sp.symbols("alpha_a", seq=True)
    km = KramersMoyalSeries(
        representation="positive-p",
        variables=(alpha_a,),
        coefficients={(2,): alpha_a},
    )

    assert "K_{∂_alpha_a^2}" in repr(km)
