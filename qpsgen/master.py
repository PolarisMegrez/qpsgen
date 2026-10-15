"""Structured master-equation data model.

A :class:`MasterEquation` explicitly declares the global mode order and holds
the Liouvillian as

- ``hamiltonian=H``: contributes ``-I [H, rho]``;
- ``channels=[LindbladChannel(jump=L, rate=gamma)]``: each contributes
  ``gamma * (L rho L† - {L†L, rho}/2)``;
- ``extra_terms=[LiouvillianTerm(coefficient, left, right)]``: arbitrary
  ``coefficient * left * rho * right`` terms.

All content is expanded into a flat list of :class:`LiouvillianTerm` objects
(``coefficient * left * rho * right`` with operator polynomials ``left`` and
``right``), which is what the derivation engine consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from qpsgen.errors import MasterEquationError, ModeError
from qpsgen.operators import Op

__all__ = ["LindbladChannel", "LiouvillianTerm", "MasterEquation"]


@dataclass(frozen=True)
class LindbladChannel:
    """A Lindblad decay channel ``rate * (jump rho jump† - {jump† jump, rho}/2)``."""

    jump: Op
    rate: sp.Expr = sp.Integer(1)

    def __post_init__(self):
        if not isinstance(self.jump, Op):
            raise MasterEquationError("LindbladChannel jump must be an Op.")
        object.__setattr__(self, "rate", sp.sympify(self.rate))


@dataclass(frozen=True)
class LiouvillianTerm:
    """An explicit term ``coefficient * left * rho * right``."""

    coefficient: sp.Expr
    left: Op = field(default_factory=lambda: Op.scalar(1))
    right: Op = field(default_factory=lambda: Op.scalar(1))

    def __post_init__(self):
        object.__setattr__(self, "coefficient", sp.sympify(self.coefficient))
        if not isinstance(self.left, Op) or not isinstance(self.right, Op):
            raise MasterEquationError("LiouvillianTerm left/right must be Op.")


class MasterEquation:
    """A quantum master equation with an explicit global mode order.

    Parameters
    ----------
    modes:
        The annihilation operators created by :func:`qpsgen.boson_modes`, in
        the desired global order.  This order defines the phase-space
        variable ordering everywhere downstream.
    hamiltonian, channels, extra_terms:
        Liouvillian content; see module docstring for conventions.
    """

    def __init__(
        self,
        modes,
        *,
        hamiltonian: Op | None = None,
        channels=(),
        extra_terms=(),
    ):
        if not modes:
            raise MasterEquationError("MasterEquation requires an explicit mode order.")
        mode_names: list[str] = []
        for m in modes:
            if (
                not isinstance(m, Op)
                or len(m.terms) != 1
                or m.terms[0][1] is None
                or len(m.terms[0][1]) != 1
                or m.terms[0][1][0][1] != "a"
                or m.terms[0][0] != 1
            ):
                raise MasterEquationError(
                    "modes must be annihilation operators from boson_modes()."
                )
            name = m.terms[0][1][0][0]
            if name in mode_names:
                raise MasterEquationError(f"Duplicate mode in mode order: {name!r}")
            mode_names.append(name)
        self.modes: tuple[str, ...] = tuple(mode_names)
        self.hamiltonian = hamiltonian
        self.channels = tuple(channels)
        self.extra_terms = tuple(extra_terms)
        self._terms: tuple[LiouvillianTerm, ...] | None = None
        self._validate_modes()

    # ------------------------------------------------------------------
    def _validate_modes(self) -> None:
        declared = set(self.modes)

        def check(op: Op, where: str) -> None:
            unknown = op.mode_names() - declared
            if unknown:
                raise ModeError(
                    f"{where} references mode(s) {sorted(unknown)} not present in "
                    f"the declared mode order {list(self.modes)}."
                )

        if self.hamiltonian is not None:
            check(self.hamiltonian, "hamiltonian")
        for ch in self.channels:
            check(ch.jump, "LindbladChannel jump")
        for t in self.extra_terms:
            check(t.left, "extra_terms left")
            check(t.right, "extra_terms right")

    # ------------------------------------------------------------------
    @property
    def terms(self) -> tuple[LiouvillianTerm, ...]:
        """The Liouvillian expanded as explicit ``c * left * rho * right`` terms."""
        if self._terms is None:
            self._terms = tuple(self._expand())
        return self._terms

    def _expand(self):
        if self.hamiltonian is not None:
            # -I [H, rho] = -I H rho + I rho H
            for coeff, factors in self.hamiltonian.terms:
                yield LiouvillianTerm(-sp.I * coeff, Op(((sp.Integer(1), factors),)))
            for coeff, factors in self.hamiltonian.terms:
                yield LiouvillianTerm(
                    sp.I * coeff, right=Op(((sp.Integer(1), factors),))
                )
        for ch in self.channels:
            L, gamma = ch.jump, ch.rate
            # gamma * L rho L†
            for c1, f1 in L.terms:
                for c2, f2 in L.dag.terms:
                    yield LiouvillianTerm(
                        gamma * c1 * c2,
                        Op(((sp.Integer(1), f1),)),
                        Op(((sp.Integer(1), f2),)),
                    )
            # -gamma/2 * (L†L rho + rho L†L)
            m = L.dag * L
            for coeff, factors in m.terms:
                yield LiouvillianTerm(
                    -gamma * coeff / 2, Op(((sp.Integer(1), factors),))
                )
                yield LiouvillianTerm(
                    -gamma * coeff / 2, right=Op(((sp.Integer(1), factors),))
                )
        yield from self.extra_terms

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        parts = [f"modes={self.modes!r}"]
        if self.hamiltonian is not None:
            parts.append(f"hamiltonian={self.hamiltonian!r}")
        if self.channels:
            parts.append(f"channels={list(self.channels)!r}")
        if self.extra_terms:
            parts.append(f"extra_terms={list(self.extra_terms)!r}")
        return f"MasterEquation({', '.join(parts)})"
