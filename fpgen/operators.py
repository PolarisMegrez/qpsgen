"""Non-commutative bosonic operator algebra.

``boson_modes("a", "b")`` creates annihilation operators.  Expressions support
``+``, ``-``, ``*``, non-negative integer powers, scalar SymPy coefficients and
the adjoint ``.dag`` (which reverses operator order and complex-conjugates
coefficients).  Operators of different modes commute; each monomial is stored
canonically with factors stable-sorted by mode name so that equal expressions
compare equal.
"""

from __future__ import annotations

import sympy as sp

from fpgen.errors import MasterEquationError

__all__ = ["Op", "boson_modes"]

# A factor is (mode_name, kind) with kind in ("a", "adag").
_Factor = tuple[str, str]


class Op:
    """A polynomial of non-commuting bosonic mode operators.

    Internally a tuple of ``(coefficient, factors)`` terms, where
    ``coefficient`` is a :class:`sympy.Expr` and ``factors`` is a tuple of
    ``(mode_name, "a"|"adag")`` applied left-to-right.  Factors of different
    modes commute and are stored stable-sorted by mode name.
    """

    __slots__ = ("terms",)

    def __init__(self, terms=()):
        cleaned = []
        for coeff, factors in terms:
            coeff = sp.sympify(coeff)
            if coeff == 0:
                continue
            cleaned.append((coeff, tuple(factors)))
        self.terms: tuple[tuple[sp.Expr, tuple[_Factor, ...]], ...] = tuple(cleaned)

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @staticmethod
    def zero() -> Op:
        return Op(())

    @staticmethod
    def scalar(value) -> Op:
        return Op(((sp.sympify(value), ()),))

    @staticmethod
    def annihilation(mode: str) -> Op:
        return Op(((sp.Integer(1), ((mode, "a"),)),))

    # ------------------------------------------------------------------
    # predicates / accessors
    # ------------------------------------------------------------------
    def is_zero(self) -> bool:
        return not self.terms

    def mode_names(self) -> frozenset[str]:
        return frozenset(mode for _, factors in self.terms for mode, _ in factors)

    # ------------------------------------------------------------------
    # arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other) -> Op:
        other = _as_op(other)
        acc: dict[tuple[_Factor, ...], sp.Expr] = {}
        for coeff, factors in self.terms + other.terms:
            acc[factors] = acc.get(factors, sp.Integer(0)) + coeff
        return Op(((c, f) for f, c in acc.items()))

    __radd__ = __add__

    def __neg__(self) -> Op:
        return Op(((-c, f) for c, f in self.terms))

    def __sub__(self, other) -> Op:
        return self + (-_as_op(other))

    def __rsub__(self, other) -> Op:
        return _as_op(other) + (-self)

    def __mul__(self, other) -> Op:
        other = _as_op(other)
        out = []
        for c1, f1 in self.terms:
            for c2, f2 in other.terms:
                factors = _canonical_factors(f1 + f2)
                out.append((c1 * c2, factors))
        return Op(out)

    def __rmul__(self, other) -> Op:
        # Scalars commute with operators.
        return self * _as_op(other)

    def __pow__(self, n: int) -> Op:
        if not isinstance(n, int) or n < 0:
            raise MasterEquationError(
                "Only non-negative integer powers of operators are supported."
            )
        result = Op.scalar(1)
        for _ in range(n):
            result = result * self
        return result

    @property
    def dag(self) -> Op:
        """Adjoint: reverse factor order, swap a <-> a†, conjugate coefficients."""
        out = []
        for coeff, factors in self.terms:
            swapped = tuple(
                (mode, "adag" if kind == "a" else "a")
                for mode, kind in reversed(factors)
            )
            out.append((sp.conjugate(coeff), _canonical_factors(swapped)))
        return Op(out)

    # ------------------------------------------------------------------
    # dunder conveniences
    # ------------------------------------------------------------------
    def __eq__(self, other) -> bool:
        if not isinstance(other, Op):
            return NotImplemented
        return set(self.terms) == set(other.terms)

    def __hash__(self) -> int:
        return hash(frozenset(self.terms))

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for coeff, factors in self.terms:
            ops = " ".join(
                mode if kind == "a" else f"{mode}†" for mode, kind in factors
            )
            if coeff == 1 and ops:
                parts.append(ops)
            elif ops:
                parts.append(f"({coeff})*{ops}")
            else:
                parts.append(f"({coeff})")
        return " + ".join(parts)


def _canonical_factors(factors: tuple[_Factor, ...]) -> tuple[_Factor, ...]:
    """Stable-sort factors by mode name (operators of different modes commute)."""
    return tuple(sorted(factors, key=lambda f: f[0]))


def _as_op(value) -> Op:
    if isinstance(value, Op):
        return value
    try:
        return Op.scalar(value)
    except Exception as exc:  # noqa: BLE001 - re-raise as domain error
        raise MasterEquationError(
            f"Cannot interpret {value!r} as an operator or scalar coefficient."
        ) from exc


def boson_modes(*names: str) -> tuple[Op, ...]:
    """Create annihilation operators for the named modes.

    >>> a, b = boson_modes("a", "b")
    >>> (a * a.dag - a.dag * a)  # not simplified; non-commutative algebra
    """
    if not names:
        raise MasterEquationError("boson_modes requires at least one mode name.")
    seen: set[str] = set()
    ops = []
    for name in names:
        if not isinstance(name, str) or not name.isidentifier():
            raise MasterEquationError(f"Invalid mode name: {name!r}")
        if name in seen:
            raise MasterEquationError(f"Duplicate mode name: {name!r}")
        seen.add(name)
        ops.append(Op.annihilation(name))
    return tuple(ops)
