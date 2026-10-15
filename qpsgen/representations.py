"""Phase-space representations and their left/right multiplication rules.

Supported representations (strict enumeration with canonical aliases; unknown
names raise :class:`RepresentationError` — there is no silent fallback):

- ``"wigner"``, ``"p"``, ``"q"``: Cahill–Glauber s-ordered representations
  (s = 0, +1, -1) on the augmented variable order
  ``z = (alpha_1, ..., alpha_N, abar_1, ..., abar_N)``, where ``abar`` is
  treated as an independent complex variable in all symbolics.
- ``"positive-p"``: the positive-P representation on
  ``z = (alpha_1, ..., alpha_N, beta_1, ..., beta_N)``, where ``beta`` is
  explicitly *not* the complex conjugate of ``alpha``.

Substitution rules (each maps ``X rho`` or ``rho X`` to a differential
operator acting on the quasi-probability distribution ``P``):

- Cahill–Glauber (parameter s)::

    a ρ  -> (α + (1-s)/2 ∂_ᾱ) P      a† ρ -> (ᾱ - (1+s)/2 ∂_α) P
    ρ a  -> (α - (1+s)/2 ∂_ᾱ) P      ρ a† -> (ᾱ + (1-s)/2 ∂_α) P

- positive-P::

    a ρ  -> α P                      a† ρ -> (β - ∂_α) P
    ρ a  -> (α - ∂_β) P              ρ a† -> β P
"""

from __future__ import annotations

import sympy as sp

from qpsgen.diffop import NormalOp
from qpsgen.errors import RepresentationError

__all__ = [
    "REPRESENTATIONS",
    "normalize_representation",
    "phase_space_variables",
    "substitution",
]

REPRESENTATIONS = ("wigner", "p", "q", "positive-p")

_ALIASES = {
    "wigner": "wigner",
    "w": "wigner",
    "weyl": "wigner",
    "wigner-weyl": "wigner",
    "weyl-wigner": "wigner",
    "p": "p",
    "glauber": "p",
    "glauber p": "p",
    "glauber-sudarshan": "p",
    "glauber sudarshan": "p",
    "q": "q",
    "husimi": "q",
    "husimi q": "q",
    "positive-p": "positive-p",
    "positive p": "positive-p",
    "positivep": "positive-p",
    "p+": "positive-p",
    "+p": "positive-p",
}

_S_VALUES = {"wigner": sp.Integer(0), "p": sp.Integer(1), "q": sp.Integer(-1)}


def normalize_representation(name: str) -> str:
    """Resolve ``name`` to a canonical representation key or raise."""
    if not isinstance(name, str):
        raise RepresentationError(
            f"Representation must be a string, got {type(name).__name__}."
        )
    key = _ALIASES.get(name.strip().lower())
    if key is None:
        raise RepresentationError(
            f"Unknown representation {name!r}. Supported: {', '.join(REPRESENTATIONS)}."
        )
    return key


def phase_space_variables(rep: str, modes: tuple[str, ...]) -> tuple[sp.Symbol, ...]:
    """Augmented phase-space variable order for the representation.

    Cahill–Glauber: ``(alpha_<m>..., abar_<m>...)``; positive-P:
    ``(alpha_<m>..., beta_<m>...)``, following the global mode order.
    """
    rep = normalize_representation(rep)
    second = "beta" if rep == "positive-p" else "abar"
    return tuple(
        sp.Symbol(f"{family}_{mode}", complex=True)
        for family in ("alpha", second)
        for mode in modes
    )


def conjugate_variable_map(
    variables: tuple[sp.Symbol, ...],
) -> dict[sp.Symbol, sp.Symbol]:
    """Map each augmented variable to its formal conjugate partner.

    For Cahill–Glauber ``alpha_i <-> abar_i``; for positive-P
    ``alpha_i <-> beta_i``.  Used only for display/conjugation purposes —
    symbolics always treat the partners as independent variables.
    """
    n = len(variables) // 2
    out: dict[sp.Symbol, sp.Symbol] = {}
    for i in range(n):
        out[variables[i]] = variables[n + i]
        out[variables[n + i]] = variables[i]
    return out


def substitution(
    rep: str,
    side: str,
    kind: str,
    mode_index: int,
    variables: tuple[sp.Symbol, ...],
) -> NormalOp:
    """Differential operator for ``a ρ``/``a† ρ``/``ρ a``/``ρ a†``.

    ``side`` is ``"L"`` (``X rho``) or ``"R"`` (``rho X``); ``kind`` is
    ``"a"`` or ``"adag"``; ``mode_index`` follows the global mode order.
    """
    rep = normalize_representation(rep)
    n = len(variables) // 2
    i = mode_index  # alpha_i lives at axis i
    j = n + mode_index  # abar_i / beta_i lives at axis n + i
    alpha, partner = variables[i], variables[j]

    if rep == "positive-p":
        if side == "L":
            if kind == "a":
                return NormalOp.multiply_by(variables, alpha)
            return NormalOp.multiply_by(variables, partner) + NormalOp.partial(
                variables, i, sp.Integer(-1)
            )
        if kind == "a":
            return NormalOp.multiply_by(variables, alpha) + NormalOp.partial(
                variables, j, sp.Integer(-1)
            )
        return NormalOp.multiply_by(variables, partner)

    s = _S_VALUES[rep]
    c_minus = sp.Rational(1, 2) * (1 - s)  # (1-s)/2
    c_plus = sp.Rational(1, 2) * (1 + s)  # (1+s)/2
    if side == "L":
        if kind == "a":
            return NormalOp.multiply_by(variables, alpha) + NormalOp.partial(
                variables, j, c_minus
            )
        return NormalOp.multiply_by(variables, partner) + NormalOp.partial(
            variables, i, -c_plus
        )
    if kind == "a":
        return NormalOp.multiply_by(variables, alpha) + NormalOp.partial(
            variables, j, -c_plus
        )
    return NormalOp.multiply_by(variables, partner) + NormalOp.partial(
        variables, i, c_minus
    )
