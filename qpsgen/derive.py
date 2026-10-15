"""Derivation of the Kramers–Moyal series from a structured master equation.

The single sign convention for the whole package is

    dP/dt = sum_mu (-1)^|mu| / mu! * ∂^mu [ K_mu(z) P ],

where ``mu`` ranges over multi-indices on the augmented phase-space variable
order and ``mu!`` is the componentwise factorial product.

Derivation pipeline:

1. Expand the master equation into explicit ``c * left * rho * right`` terms.
2. Replace every ``a``/``a†`` left/right multiplication by the representation's
   differential operator and compose them (left factors left-to-right, then
   right factors right-to-left — the factor closest to ``rho`` acts first).
   This yields an exact normal form ``sum_mu f_mu(z) ∂^mu P``.
3. Convert normal -> flux *exactly* (multivariate Leibniz), keeping every
   lower-order term the conversion generates.  Nothing is truncated here.
4. Read off ``K_mu = (-1)^|mu| mu! g_mu`` from the flux coefficients ``g_mu``.
"""

from __future__ import annotations

import sympy as sp

from qpsgen.diffop import (
    NormalOp,
    multi_index_factorial,
    multi_index_order,
    normal_to_flux,
)
from qpsgen.master import MasterEquation
from qpsgen.representations import (
    normalize_representation,
    phase_space_variables,
    substitution,
)
from qpsgen.results import KramersMoyalSeries

__all__ = ["derive_kramers_moyal"]


def derive_kramers_moyal(
    master: MasterEquation, representation: str = "wigner"
) -> KramersMoyalSeries:
    """Derive the full Kramers–Moyal series of ``master`` in ``representation``.

    The result keeps *all* orders; use
    :meth:`KramersMoyalSeries.truncate` to obtain a Fokker–Planck equation.
    """
    if not isinstance(master, MasterEquation):
        raise TypeError("derive_kramers_moyal expects a MasterEquation.")
    rep = normalize_representation(representation)
    variables = phase_space_variables(rep, master.modes)
    mode_index = {name: i for i, name in enumerate(master.modes)}

    total = NormalOp.zero(variables)
    for term in master.terms:
        for c_left, f_left in term.left.terms:
            for c_right, f_right in term.right.terms:
                # Sequence of substitution operators: left factors
                # left-to-right, then right factors right-to-left (the factor
                # closest to rho is the innermost one).
                seq = [("L", kind, mode) for mode, kind in f_left]
                seq += [("R", kind, mode) for mode, kind in reversed(f_right)]
                acc = NormalOp.identity(variables)
                for side, kind, mode in seq:
                    sub = substitution(rep, side, kind, mode_index[mode], variables)
                    acc = acc * sub
                total = total + acc.scale(term.coefficient * c_left * c_right)

    flux = normal_to_flux(total)
    coefficients = {}
    for mu, g in flux.items():
        sign = -1 if multi_index_order(mu) % 2 else 1
        k = sp.simplify(sign * multi_index_factorial(mu) * g)
        if k != 0:
            coefficients[mu] = k
    return KramersMoyalSeries(
        representation=rep,
        variables=variables,
        coefficients=coefficients,
        master=master,
    )
