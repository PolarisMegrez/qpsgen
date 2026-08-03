"""Structured results: Kramers–Moyal series, Fokker–Planck, Langevin.

Sign convention (unique to the whole package)::

    dP/dt = sum_mu (-1)^|mu| / mu! * ∂^mu [ K_mu(z) P ]

- :class:`KramersMoyalSeries` keeps every order ``mu`` with its SymPy
  coefficient ``K_mu``; nothing is truncated at derivation time.
- :meth:`KramersMoyalSeries.truncate` drops flux terms with ``|mu| > order``
  and records them explicitly (:class:`FokkerPlanckEquation`).
- :meth:`FokkerPlanckEquation.to_langevin` builds an Itô c-number Langevin
  equation on the augmented state ``z`` (:class:`LangevinEquation`).
"""

from __future__ import annotations

import sympy as sp

from fpgen.diffop import multi_index_order
from fpgen.errors import ProbabilityConservationError

__all__ = [
    "KramersMoyalSeries",
    "FokkerPlanckEquation",
    "LangevinEquation",
]

MultiIndex = tuple[int, ...]


# ----------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------
def _latex_symbol(sym: sp.Symbol) -> str:
    name = sym.name
    for prefix, tex in (("abar_", r"\bar{\alpha}_"), ("alpha_", r"\alpha_"),
                        ("beta_", r"\beta_")):
        if name.startswith(prefix):
            return tex + "{" + name[len(prefix):] + "}"
    return name


def _mu_str(mu: MultiIndex, variables, latex: bool = False) -> str:
    parts = []
    for m, var in zip(mu, variables, strict=True):
        if not m:
            continue
        if latex:
            base = _latex_symbol(var)
            derivative = rf"\partial_{{{base}}}"
            if m > 1:
                derivative += rf"^{{{m}}}"
        else:
            derivative = f"∂_{var.name}"
            if m > 1:
                derivative += f"^{m}"
        parts.append(derivative)
    return " ".join(parts) if parts else ("1" if not latex else "1")


def _format_coefficients(
    coefficients: dict[MultiIndex, sp.Expr], variables, latex: bool = False
) -> list[str]:
    lines = []
    for mu in sorted(coefficients, key=lambda m: (multi_index_order(m), m)):
        k = coefficients[mu]
        k_str = sp.latex(k) if latex else str(k)
        lines.append(f"K_{{{_mu_str(mu, variables, latex)}}} = {k_str}")
    return lines or ["(vanishing series)"]


# ----------------------------------------------------------------------
class KramersMoyalSeries:
    """Full Kramers–Moyal series ``sum_mu (-1)^|mu|/mu! ∂^mu[K_mu P]``.

    Attributes
    ----------
    representation:
        Canonical phase-space representation name.  It determines whether
        the augmented state has a conjugate pairing (Wigner/P/Q) or two
        independent halves (positive-P).
    representation:
        Canonical representation name (``wigner``/``p``/``q``/``positive-p``).
    variables:
        Augmented phase-space variable order ``z``.
    coefficients:
        Mapping multi-index ``mu -> K_mu`` (SymPy expressions); all orders
        are kept exactly as derived.
    master:
        The originating :class:`fpgen.master.MasterEquation`.
    """

    def __init__(self, representation, variables, coefficients, master=None):
        self.representation = representation
        self.variables = tuple(variables)
        self.coefficients: dict[MultiIndex, sp.Expr] = dict(coefficients)
        self.master = master

    @property
    def order(self) -> int:
        """Highest Kramers–Moyal order present."""
        return max((multi_index_order(mu) for mu in self.coefficients), default=0)

    @property
    def zero_order(self) -> sp.Expr:
        """The ``|mu| = 0`` coefficient (must vanish for probability
        conservation)."""
        return self.coefficients.get(tuple([0] * len(self.variables)), sp.Integer(0))

    def truncate(self, order: int = 2) -> FokkerPlanckEquation:
        """Truncate to a Fokker–Planck equation, keeping ``|mu| <= order``.

        Truncation happens here — in flux form, after the exact normal ->
        flux rearrangement — never in the normal form.  Dropped terms are
        recorded on the result.
        """
        kept, discarded = {}, {}
        for mu, k in self.coefficients.items():
            (kept if multi_index_order(mu) <= order else discarded)[mu] = k
        return FokkerPlanckEquation(
            representation=self.representation,
            variables=self.variables,
            coefficients=kept,
            discarded=discarded,
            original_max_order=self.order,
            truncation_order=order,
            master=self.master,
        )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        lines = [
            f"KramersMoyalSeries(representation={self.representation!r}, "
            f"order={self.order})",
            "  z = (" + ", ".join(v.name for v in self.variables) + ")",
        ]
        lines += ["  " + line for line in _format_coefficients(
            self.coefficients, self.variables)]
        return "\n".join(lines)

    def to_latex(self) -> str:
        head = (
            r"\dot P = \sum_{\mu} \frac{(-1)^{|\mu|}}{\mu!}\,"
            r"\partial^{\mu}\left[K_{\mu}(z)\,P\right]"
        )
        terms = r" \\ ".join(_format_coefficients(self.coefficients,
                                                   self.variables, latex=True))
        return head + r"\\[4pt]" + terms

    def _repr_latex_(self) -> str:
        return f"${self.to_latex()}$"


# ----------------------------------------------------------------------
class FokkerPlanckEquation:
    """Kramers–Moyal series truncated at ``|mu| <= truncation_order``.

    ``discarded`` holds every dropped coefficient, ``original_max_order`` the
    highest order of the untruncated series, and ``is_exact`` is ``True``
    only when nothing was dropped.  The truncation is always displayed in
    ``repr``/LaTeX output.
    """

    def __init__(
        self,
        representation,
        variables,
        coefficients,
        discarded,
        original_max_order,
        truncation_order=2,
        master=None,
    ):
        self.representation = representation
        self.variables = tuple(variables)
        self.coefficients: dict[MultiIndex, sp.Expr] = dict(coefficients)
        self.discarded: dict[MultiIndex, sp.Expr] = dict(discarded)
        self.original_max_order = original_max_order
        self.truncation_order = truncation_order
        self.master = master

    @property
    def is_exact(self) -> bool:
        """``True`` when the truncation discarded nothing."""
        return not self.discarded

    @property
    def zero_order(self) -> sp.Expr:
        return self.coefficients.get(tuple([0] * len(self.variables)), sp.Integer(0))

    def _unit(self, i: int) -> MultiIndex:
        mu = [0] * len(self.variables)
        mu[i] = 1
        return tuple(mu)

    @property
    def drift(self) -> list[sp.Expr]:
        """Drift vector ``A_i = K_{e_i}``."""
        return [
            self.coefficients.get(self._unit(i), sp.Integer(0))
            for i in range(len(self.variables))
        ]

    @property
    def diffusion(self) -> sp.Matrix:
        """Complex-symmetric FPE diffusion matrix ``G_ij = K_{e_i + e_j}``."""
        d = len(self.variables)

        def entry(i, j):
            mu = list(self._unit(i))
            mu[j] += 1
            return self.coefficients.get(tuple(mu), sp.Integer(0))

        return sp.Matrix(d, d, entry)

    # ------------------------------------------------------------------
    def to_langevin(self) -> LangevinEquation:
        """Convert to an Itô c-number Langevin equation on the augmented
        state ``z``.

        Raises :class:`ProbabilityConservationError` if the zero-order
        Kramers–Moyal coefficient is nonzero: the underlying master equation
        does not conserve probability and no stochastic process exists.
        """
        residual = sp.simplify(self.zero_order)
        if residual != 0:
            raise ProbabilityConservationError(
                "Cannot convert to a Langevin equation: the zero-order "
                f"Kramers–Moyal coefficient is {residual}, so the master "
                "equation does not conserve probability."
            )
        d = len(self.variables)
        A = sp.Matrix(self.drift)
        G = self.diffusion
        B = _canonical_noise_factor(G)
        H_eff, diagnostics = _extract_hamiltonian(A, self.variables)
        return LangevinEquation(
            representation=self.representation,
            variables=self.variables,
            drift=A,
            diffusion=G,
            noise_factor=B,
            hamiltonian=H_eff,
            diagnostics=diagnostics,
            fokker_planck=self,
        )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        status = "exact" if self.is_exact else "approximate"
        lines = [
            f"FokkerPlanckEquation(representation={self.representation!r}, "
            f"truncation_order={self.truncation_order}, "
            f"original_max_order={self.original_max_order}, {status})",
            "  z = (" + ", ".join(v.name for v in self.variables) + ")",
            "  kept:",
        ]
        lines += ["    " + line for line in _format_coefficients(
            self.coefficients, self.variables)]
        if self.discarded:
            lines.append("  discarded by truncation:")
            lines += ["    " + line for line in _format_coefficients(
                self.discarded, self.variables)]
        return "\n".join(lines)

    def to_latex(self) -> str:
        head = (
            r"\dot P = -\sum_i \partial_i\left[A_i(z)\,P\right]"
            r" + \frac{1}{2}\sum_{i,j}\partial_i\partial_j"
            r"\left[G_{ij}(z)\,P\right]"
        )
        if not self.is_exact:
            head += (
                r"\quad\text{(truncated at }|\mu|="
                + str(self.truncation_order)
                + r"\text{; original order }"
                + str(self.original_max_order)
                + r"\text{)}"
            )
        terms = r" \\ ".join(_format_coefficients(self.coefficients,
                                                   self.variables, latex=True))
        return head + r"\\[4pt]" + terms

    def _repr_latex_(self) -> str:
        return f"${self.to_latex()}$"


# ----------------------------------------------------------------------
class LangevinEquation:
    """Itô c-number Langevin equation on the augmented state ``z``::

        dz/dt = A(z) + xi,   xi = B eta,

    with ``eta`` independent real Gaussian white noises,
    ``<eta_m(t) eta_n(t')> = delta_mn delta(t-t')``.

    Attributes
    ----------
    drift:
        ``A(z) = -i H_eff(z) z`` whenever ``hamiltonian`` is available.
    diffusion:
        Full complex-symmetric augmented FPE matrix ``G_aug``.  For
        Wigner/P/Q its upper-left and upper-right blocks directly give the
        physical anomalous and normal correlations ``M`` and ``D``.
    noise_factor:
        Canonical factor ``B`` with ``B B^T = G`` (after SymPy
        simplification).  Not guaranteed to have the minimal number of noise
        columns.
    hamiltonian:
        Augmented effective Hamiltonian ``H_eff(z)``, or ``None`` when the
        drift cannot be decomposed (see ``diagnostics``).  ``H_eff``
        includes dissipative drift, may be non-Hermitian, and is *not* the
        physical Hamiltonian of the input master equation.
    physical_hamiltonian, anomalous_hamiltonian:
        Upper-left and upper-right ``N x N`` blocks for Wigner/P/Q.  They are
        ``None`` for positive-P because its second half is independent.
    """

    def __init__(
        self,
        representation,
        variables,
        drift,
        diffusion,
        noise_factor,
        hamiltonian,
        diagnostics,
        fokker_planck=None,
    ):
        self.representation = representation
        self.variables = tuple(variables)
        self.drift = drift
        self.diffusion = diffusion
        self.noise_factor = noise_factor
        self.hamiltonian = hamiltonian
        self.diagnostics = tuple(diagnostics)
        self.fokker_planck = fokker_planck

    # ------------------------------------------------------------------
    @property
    def eta(self) -> sp.Matrix:
        """Independent real Gaussian white-noise symbols."""
        return sp.Matrix(
            [sp.Symbol(f"eta_{m}", real=True) for m in range(self.noise_factor.cols)]
        )

    @property
    def xi(self) -> sp.Matrix:
        """Complex noise vector ``xi = B eta``."""
        return self.noise_factor * self.eta

    @property
    def anomalous_correlation(self) -> sp.Matrix:
        """Anomalous correlation determined directly by the FPE.

        For Cahill--Glauber representations this is the upper-left block
        ``M = <xi xi^T>`` of the augmented diffusion matrix.  Positive-P has
        no conjugate pairing between the two halves, so its full augmented
        diffusion matrix is the transpose correlation.
        """
        if self.representation == "positive-p":
            return self.diffusion.applyfunc(sp.simplify)
        n = len(self.variables) // 2
        return self.diffusion[:n, :n].applyfunc(sp.simplify)

    @property
    def normal_correlation(self) -> sp.Matrix:
        """Normal correlation ``D = <xi xi†>``.

        For Cahill--Glauber representations this is read directly from the
        upper-right block of the augmented FPE diffusion matrix, avoiding a
        factor-dependent and needlessly complicated round trip through
        ``B``.  In positive-P the partner variables are independent, so the
        FPE does not determine this Hermitian correlation; there it describes
        the canonical ``B`` realization.
        """
        if self.representation != "positive-p":
            n = len(self.variables) // 2
            return self.diffusion[:n, n:].applyfunc(sp.simplify)
        B = self.noise_factor
        return (B * B.conjugate().T).applyfunc(sp.simplify)

    @property
    def physical_hamiltonian(self) -> sp.Matrix | None:
        """Hamiltonian block acting on the unbarred Cahill--Glauber state.

        Returns ``None`` for positive-P, whose second-half variables are not
        conjugates, or when the augmented drift has no canonical factorization.
        """
        if self.representation == "positive-p" or self.hamiltonian is None:
            return None
        n = len(self.variables) // 2
        return self.hamiltonian[:n, :n].applyfunc(sp.simplify)

    @property
    def anomalous_hamiltonian(self) -> sp.Matrix | None:
        """Upper-right augmented Hamiltonian block for Cahill--Glauber states."""
        if self.representation == "positive-p" or self.hamiltonian is None:
            return None
        n = len(self.variables) // 2
        return self.hamiltonian[:n, n:].applyfunc(sp.simplify)

    def to_covariance_jacobian(self):
        """Build the closed Hermitian covariance flow and real Jacobian.

        The coordinate ordering is diagonal entries in declared mode order,
        followed by upper-triangle real parts in declared index-pair order and
        then imaginary parts.  Raises ``CovarianceClosureError`` when the
        normal covariance is not closed at second order.
        """
        from fpgen.covariance import HermitianCovarianceJacobian

        return HermitianCovarianceJacobian.from_langevin(self)

    def to_covariance_dynamics(self, *, parameters=None, parameter_domains=None):
        """Build a machine-safe closed Hermitian covariance dynamics object.

        ``parameters`` fixes the flat numerical parameter order.  It is
        optional for symbolic inspection, but required before exporting a
        numerical model specification.  Parameters without SymPy real-domain
        assumptions require an explicit ``parameter_domains`` override.
        """
        from fpgen.covariance import CovarianceDynamics

        return CovarianceDynamics.from_langevin(
            self,
            parameters=parameters,
            parameter_domains=parameter_domains,
        )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        d = len(self.variables)
        lines = [
            "LangevinEquation(Itô, augmented state)",
            "  z = (" + ", ".join(v.name for v in self.variables) + ")",
            f"  dz/dt = A(z) + xi,  xi = B eta  ({self.noise_factor.cols} noise(s))",
            "  A(z) =",
        ]
        for i in range(d):
            lines.append(f"    [{i}] {self.drift[i]}")
        lines.append("  G_aug = <xi_aug xi_aug^T> =")
        lines += ["    " + row for row in str(self.diffusion).splitlines()]
        lines.append("  B =")
        lines += ["    " + row for row in str(self.noise_factor).splitlines()]
        if self.representation == "positive-p":
            lines.append("  D = <xi xi†> = B B† (canonical realization) =")
        else:
            lines.append("  M = <xi xi^T> (direct FPE block) =")
            lines += ["    " + row for row in str(self.anomalous_correlation).splitlines()]
            lines.append("  D = <xi xi†> (direct FPE block) =")
        lines += ["    " + row for row in str(self.normal_correlation).splitlines()]
        if self.hamiltonian is not None:
            lines.append("  H_eff(z)  (includes dissipation; may be non-Hermitian) =")
            lines += ["    " + row for row in str(self.hamiltonian).splitlines()]
        else:
            lines.append("  H_eff = None (drift not decomposable as -i H_eff z)")
            for note in self.diagnostics:
                lines.append(f"    - {note}")
        return "\n".join(lines)

    def to_latex(self) -> str:
        parts = [
            r"\dot z = A(z) + \xi,\quad \xi = B\,\eta",
            r"A(z) = " + sp.latex(self.drift),
            r"G_{\mathrm{aug}} = \langle\xi_{\mathrm{aug}}\,"
            r"\xi_{\mathrm{aug}}^{T}\rangle = " + sp.latex(self.diffusion),
            r"B = " + sp.latex(self.noise_factor),
        ]
        if self.representation == "positive-p":
            parts.append(
                r"D = \langle\xi\,\xi^{\dagger}\rangle_{B} = B B^{\dagger} = "
                + sp.latex(self.normal_correlation)
            )
        else:
            parts.extend([
                r"M = \langle\xi\,\xi^{T}\rangle = "
                + sp.latex(self.anomalous_correlation),
                r"D = \langle\xi\,\xi^{\dagger}\rangle = "
                + sp.latex(self.normal_correlation),
            ])
        if self.hamiltonian is not None:
            parts.append(r"A = -i\,H_{\mathrm{eff}}(z)\,z,\quad H_{\mathrm{eff}} = "
                         + sp.latex(self.hamiltonian))
        else:
            parts.append(
                r"H_{\mathrm{eff}} = \varnothing\ \text{(drift not decomposable)}"
            )
        return r" \\ ".join(parts)

    def _repr_latex_(self) -> str:
        return f"${self.to_latex()}$"


# ----------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------
def _canonical_noise_factor(G: sp.Matrix) -> sp.Matrix:
    """Sparse canonical factor ``B`` with ``B B^T = G`` for complex-symmetric
    ``G``.

    Each nonzero diagonal entry contributes one column ``sqrt(G_ii) e_i``;
    each nonzero off-diagonal pair ``(i, j)`` contributes two columns with a
    relative phase ``(1, G_ij/2)`` and ``(I, -I G_ij/2)`` on rows ``(i, j)``,
    whose excess diagonal contributions cancel.  Works for any symbolic
    complex-symmetric ``G``; the number of noise columns is not minimal.
    """
    d = G.rows
    cols: list[sp.Matrix] = []
    for i in range(d):
        g = sp.simplify(G[i, i])
        if g != 0:
            col = sp.zeros(d, 1)
            col[i] = sp.sqrt(g)
            cols.append(col)
    for i in range(d):
        for j in range(i + 1, d):
            g = sp.simplify(G[i, j])
            if g != 0:
                u = sp.zeros(d, 1)
                u[i] = 1
                u[j] = g / 2
                v = sp.zeros(d, 1)
                v[i] = sp.I
                v[j] = -sp.I * g / 2
                cols.extend([u, v])
    if not cols:
        return sp.zeros(d, 0)
    return sp.Matrix.hstack(*cols)


def _extract_hamiltonian(
    A: sp.Matrix, variables
) -> tuple[sp.Matrix | None, tuple[str, ...]]:
    """Decompose drift ``A`` as ``A = -i H_eff(z) z`` if canonically possible.

    Per row ``i``: monomials containing ``z_i`` are assigned to the diagonal
    entry ``H_ii``.  Otherwise a unique factor from the same augmented half is
    preferred, which gives the physical non-diagonal factorization used by
    nonlinear mode-coupling models.  A unique factor in the opposite half is
    accepted when no same-half factor exists.  Constant terms or ambiguous
    choices make the decomposition impossible.
    """
    d = len(variables)
    H = sp.zeros(d)
    diagnostics: list[str] = []
    ok = True
    for i in range(d):
        expanded = sp.expand(A[i])
        try:
            poly = sp.Poly(expanded, *variables)
        except sp.PolynomialError:
            ok = False
            diagnostics.append(
                f"A[{i}] = {expanded} is not a polynomial in z; cannot "
                "extract H_eff."
            )
            continue
        for exp, coeff in poly.terms():
            if sp.simplify(coeff) == 0:
                continue
            degree = sum(exp)
            if degree == 0:
                ok = False
                diagnostics.append(
                    f"A[{i}] contains constant drive {coeff}; H_eff undefined."
                )
            elif exp[i] >= 1:
                factor = sp.Integer(1)
                for j, (var, p) in enumerate(zip(variables, exp, strict=True)):
                    p_eff = p - 1 if j == i else p
                    if p_eff:
                        factor *= var**p_eff
                H[i, i] += sp.I * coeff * factor
            else:
                n = d // 2
                same_half = range(0, n) if i < n else range(n, d)
                candidates = [j for j in same_half if exp[j] >= 1]
                if not candidates:
                    candidates = [j for j, p in enumerate(exp) if p >= 1]
                if len(candidates) != 1:
                    ok = False
                    diagnostics.append(
                        f"A[{i}] contains term {coeff}*z^{exp} with "
                        f"ambiguous Hamiltonian columns {candidates}."
                    )
                    continue
                j = candidates[0]
                factor = sp.Integer(1)
                for k, (var, p) in enumerate(zip(variables, exp, strict=True)):
                    p_eff = p - 1 if k == j else p
                    if p_eff:
                        factor *= var**p_eff
                H[i, j] += sp.I * coeff * factor
    if not ok:
        return None, tuple(diagnostics)
    return H, ()
