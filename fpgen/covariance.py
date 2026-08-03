"""Hermitian covariance dynamics and Jacobian construction.

For a c-number Langevin equation

    dot(alpha) = -I H(alpha) alpha + xi,

with normal diffusion ``D = <xi xi^dagger>``, the normal covariance
``R_ij = <alpha_i alpha_j^*>`` obeys the closed second-moment equation

    dot(R) = -I (H(R) R - R H(R)^dagger) + D(R)

when every phase-space dependence in ``H`` and ``D`` is constant or a single
bilinear ``alpha_j abar_i``.  No Gaussian/Wick closure is applied implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from fpgen.errors import CovarianceClosureError

__all__ = ["HermitianCovarianceJacobian"]


def _simplified_matrix(matrix) -> sp.Matrix:
    return sp.Matrix(matrix).applyfunc(sp.simplify)


def _coordinate_name(kind: str, mode_i: str, mode_j: str) -> str:
    base = f"R_{{{mode_i}{mode_j}}}"
    if kind == "diag":
        return base
    return rf"\operatorname{{{kind}}}({base})"


def _hermitian_state(mode_names: tuple[str, ...]):
    """Build ``R`` and its real coordinates in the public vector order."""
    n = len(mode_names)
    mode_index = {mode: i for i, mode in enumerate(mode_names)}
    R = sp.zeros(n)

    diagonal = tuple(
        sp.Symbol(_coordinate_name("diag", mode, mode), real=True)
        for mode in mode_names
    )
    for i, value in enumerate(diagonal):
        R[i, i] = value

    pair_order = tuple(combinations(sorted(mode_names), 2))
    real_parts = tuple(
        sp.Symbol(_coordinate_name("Re", left, right), real=True)
        for left, right in pair_order
    )
    imag_parts = tuple(
        sp.Symbol(_coordinate_name("Im", left, right), real=True)
        for left, right in pair_order
    )
    for (left, right), re_part, im_part in zip(
        pair_order, real_parts, imag_parts, strict=True
    ):
        i, j = mode_index[left], mode_index[right]
        R[i, j] = re_part + sp.I * im_part
        R[j, i] = re_part - sp.I * im_part

    coordinates = diagonal + real_parts + imag_parts
    return R, coordinates, pair_order


def _close_bilinears(
    expr: sp.Expr,
    phase_variables: tuple[sp.Symbol, ...],
    covariance: sp.Matrix,
) -> sp.Expr:
    """Map ``alpha_j abar_i`` to ``R_ji`` without higher-moment closure."""
    expr = sp.expand(expr)
    try:
        polynomial = sp.Poly(expr, *phase_variables)
    except sp.PolynomialError as exc:
        raise CovarianceClosureError(
            f"Expression {expr} is not polynomial in the phase-space variables."
        ) from exc

    n = len(phase_variables) // 2
    closed = sp.Integer(0)
    for powers, coefficient in polynomial.terms():
        degree = sum(powers)
        if degree == 0:
            closed += coefficient
            continue

        alpha_axes = [i for i, power in enumerate(powers[:n]) if power]
        partner_axes = [i for i, power in enumerate(powers[n:]) if power]
        is_single_bilinear = (
            degree == 2
            and len(alpha_axes) == 1
            and len(partner_axes) == 1
            and powers[alpha_axes[0]] == 1
            and powers[n + partner_axes[0]] == 1
        )
        if not is_single_bilinear:
            raise CovarianceClosureError(
                "Second-moment closure requires constants or one bilinear "
                f"alpha_j*abar_i; found monomial powers {powers} in {expr}."
            )
        j, i = alpha_axes[0], partner_axes[0]
        closed += coefficient * covariance[j, i]
    return sp.simplify(closed)


def _close_matrix(matrix, phase_variables, covariance) -> sp.Matrix:
    return sp.Matrix(matrix).applyfunc(
        lambda entry: _close_bilinears(entry, phase_variables, covariance)
    )


def _real_part(expr: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.expand_complex(sp.re(expr)))


def _imag_part(expr: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.expand_complex(sp.im(expr)))


@dataclass(frozen=True)
class HermitianCovarianceJacobian:
    """Closed Hermitian covariance flow and its real Jacobian.

    ``coordinates`` follow the fixed order:

    1. diagonal entries in the declared mode order;
    2. real parts of off-diagonal entries in lexicographic mode-pair order;
    3. imaginary parts in the same lexicographic order.
    """

    modes: tuple[str, ...]
    covariance: sp.Matrix
    coordinates: tuple[sp.Symbol, ...]
    pair_order: tuple[tuple[str, str], ...]
    hamiltonian: sp.Matrix
    diffusion: sp.Matrix
    flow_matrix: sp.Matrix
    flow_vector: sp.Matrix
    jacobian: sp.Matrix

    @classmethod
    def from_langevin(cls, langevin) -> HermitianCovarianceJacobian:
        if langevin.representation == "positive-p":
            raise CovarianceClosureError(
                "A Hermitian covariance Jacobian requires conjugate-augmented "
                "Wigner, P, or Q variables; positive-P partners are independent."
            )
        if langevin.physical_hamiltonian is None:
            raise CovarianceClosureError(
                "The drift has no closed physical Hamiltonian block."
            )
        anomalous_h = _simplified_matrix(langevin.anomalous_hamiltonian)
        if anomalous_h != sp.zeros(*anomalous_h.shape):
            raise CovarianceClosureError(
                "The anomalous Hamiltonian block is nonzero, so the Hermitian "
                "normal covariance R is not closed without anomalous moments."
            )
        master = getattr(langevin.fokker_planck, "master", None)
        if master is None:
            raise CovarianceClosureError(
                "Mode names are unavailable because the originating master "
                "equation was not retained."
            )

        modes = tuple(master.modes)
        covariance, coordinates, pair_order = _hermitian_state(modes)
        phase_variables = tuple(langevin.variables)
        H = _close_matrix(
            langevin.physical_hamiltonian, phase_variables, covariance
        )
        D = _close_matrix(langevin.normal_correlation, phase_variables, covariance)
        flow_matrix = _simplified_matrix(
            -sp.I * (H * covariance - covariance * H.conjugate().T) + D
        )
        hermitian_residual = _simplified_matrix(
            flow_matrix - flow_matrix.conjugate().T
        )
        if hermitian_residual != sp.zeros(*flow_matrix.shape):
            raise CovarianceClosureError(
                "The derived covariance flow is not Hermitian; residual: "
                f"{hermitian_residual}."
            )

        mode_index = {mode: i for i, mode in enumerate(modes)}
        diagonal_flow = tuple(_real_part(flow_matrix[i, i]) for i in range(len(modes)))
        real_flow = tuple(
            _real_part(flow_matrix[mode_index[left], mode_index[right]])
            for left, right in pair_order
        )
        imag_flow = tuple(
            _imag_part(flow_matrix[mode_index[left], mode_index[right]])
            for left, right in pair_order
        )
        flow_vector = sp.Matrix(diagonal_flow + real_flow + imag_flow)
        jacobian = flow_vector.jacobian(sp.Matrix(coordinates)).applyfunc(sp.simplify)
        return cls(
            modes=modes,
            covariance=covariance,
            coordinates=coordinates,
            pair_order=pair_order,
            hamiltonian=H,
            diffusion=D,
            flow_matrix=flow_matrix,
            flow_vector=flow_vector,
            jacobian=jacobian,
        )

    def __repr__(self) -> str:
        return (
            f"HermitianCovarianceJacobian(modes={self.modes!r}, "
            f"coordinates={tuple(str(v) for v in self.coordinates)!r})\n"
            f"R = {self.covariance}\n"
            f"dot(r) = {self.flow_vector}\n"
            f"J = {self.jacobian}"
        )

    def to_latex(self) -> str:
        coordinate_vector = sp.Matrix(self.coordinates)
        return r" \\ ".join([
            r"\mathbf r = " + sp.latex(coordinate_vector),
            r"\dot{\mathbf r} = " + sp.latex(self.flow_vector),
            r"\mathbf J(\mathbf R) = " + sp.latex(self.jacobian),
        ])

    def _repr_latex_(self) -> str:
        return f"${self.to_latex()}$"
