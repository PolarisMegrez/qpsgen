"""Canonical noise-factor construction and noise correlation definitions."""

import sympy as sp

from fpgen.results import _canonical_noise_factor


def check_factor(G):
    B = _canonical_noise_factor(G)
    assert sp.simplify(B * B.T - G) == sp.zeros(G.rows)
    return B


def test_factor_diagonal():
    x, y = sp.symbols("x y")
    G = sp.Matrix([[x, 0], [0, y]])
    B = check_factor(G)
    assert B.cols == 2  # one noise column per diagonal entry


def test_factor_offdiagonal():
    g = sp.symbols("g")
    G = sp.Matrix([[0, g], [g, 0]])
    B = check_factor(G)
    assert B.cols == 2  # two columns with relative phase


def test_factor_general_symbolic_symmetric():
    a, b, c, d, e, f = sp.symbols("a b c d e f")
    G = sp.Matrix(
        [
            [a, d, e],
            [d, b, f],
            [e, f, c],
        ]
    )
    check_factor(G)


def test_factor_with_zero_entries():
    g, h = sp.symbols("g h")
    G = sp.Matrix(
        [
            [g, 0, h],
            [0, 0, 0],
            [h, 0, 0],
        ]
    )
    check_factor(G)


def test_factor_zero_matrix():
    G = sp.zeros(2)
    B = _canonical_noise_factor(G)
    assert B.cols == 0
    assert B * B.T == G


def test_correlation_definitions():
    # M = <xi xi^T> = G and D = <xi xi†> = B B†, verified symbolically
    # against the eta-level definitions <eta_m eta_n> = delta_mn.
    x, g = sp.symbols("x g")
    G = sp.Matrix([[x, g], [g, 0]])
    B = _canonical_noise_factor(G)
    m = B.cols
    # M_ij = sum_k B_ik B_jk <eta_k^2> = (B B^T)_ij; likewise D with |B_ik|^2.
    M = sum((B[:, k] * B[:, k].T for k in range(m)), sp.zeros(2))
    D = sum(
        (B[:, k] * B[:, k].conjugate().T for k in range(m)), sp.zeros(2)
    )
    assert sp.simplify(M - G) == sp.zeros(2)
    assert sp.simplify(D - B * B.conjugate().T) == sp.zeros(2)
