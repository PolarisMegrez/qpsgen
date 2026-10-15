"""Tests for the non-commutative operator algebra."""

import pytest
import sympy as sp

from qpsgen import MasterEquationError, boson_modes


def test_boson_modes_basic():
    a, b = boson_modes("a", "b")
    assert a.terms == ((sp.Integer(1), (("a", "a"),)),)
    assert b.terms == ((sp.Integer(1), (("b", "a"),)),)


def test_invalid_mode_names():
    with pytest.raises(MasterEquationError):
        boson_modes()
    with pytest.raises(MasterEquationError):
        boson_modes("a", "a")
    with pytest.raises(MasterEquationError):
        boson_modes("1x")


def test_dag_reverses_order_and_conjugates():
    (a,) = boson_modes("a")
    chi = sp.symbols("chi")
    op = chi * a.dag * a
    dagged = op.dag
    assert dagged.terms == ((sp.conjugate(chi), (("a", "adag"), ("a", "a"))),)


def test_dag_swaps_annihilation_creation():
    (a,) = boson_modes("a")
    assert a.dag.terms == ((sp.Integer(1), (("a", "adag"),)),)
    assert a.dag.dag == a


def test_powers_expand():
    (a,) = boson_modes("a")
    assert (a**2).terms == ((sp.Integer(1), (("a", "a"), ("a", "a"))),)
    assert (a**0).terms == ((sp.Integer(1), ()),)
    with pytest.raises(MasterEquationError):
        _ = a ** (-1)


def test_noncommutative_same_mode():
    (a,) = boson_modes("a")
    comm = a * a.dag - a.dag * a
    # Not simplified away: the algebra is non-commutative.
    assert len(comm.terms) == 2


def test_different_modes_commute_canonically():
    a, b = boson_modes("a", "b")
    assert a * b.dag == b.dag * a


def test_scalar_coefficients():
    (a,) = boson_modes("a")
    om = sp.symbols("omega")
    op = om * a + sp.I * a.dag
    kinds = {f[0][1] for _, f in op.terms if f}
    assert kinds == {"a", "adag"}
    coeffs = {f[0][1]: c for c, f in op.terms}
    assert coeffs["a"] == om
    assert coeffs["adag"] == sp.I


def test_addition_combines_like_terms():
    (a,) = boson_modes("a")
    op = a + a
    assert op.terms == ((sp.Integer(2), (("a", "a"),)),)
    assert (a - a).is_zero()
