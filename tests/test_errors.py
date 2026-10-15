"""Error paths: conservation, decomposability, representations, modes."""

import pytest
import sympy as sp

from qpsgen import (
    LiouvillianTerm,
    MasterEquation,
    MasterEquationError,
    ModeError,
    ProbabilityConservationError,
    RepresentationError,
    boson_modes,
    derive_kramers_moyal,
)
from qpsgen.results import _extract_hamiltonian


def test_non_conserving_liouvillian_refuses_langevin():
    (a,) = boson_modes("a")
    # A bare a rho term is not trace-preserving: K_0 = alpha != 0.
    master = MasterEquation(modes=[a], extra_terms=[LiouvillianTerm(1, left=a)])
    km = derive_kramers_moyal(master, representation="positive-p")
    fpe = km.truncate(2)
    assert fpe.zero_order != 0
    # The FPE itself remains displayable.
    assert "FokkerPlanckEquation" in repr(fpe)
    with pytest.raises(ProbabilityConservationError):
        fpe.to_langevin()


def test_constant_drive_blocks_hamiltonian_extraction():
    (a,) = boson_modes("a")
    eps = sp.symbols("epsilon")
    # Linear drive H = eps (a + a†): drift has a constant inhomogeneous term.
    master = MasterEquation(modes=[a], hamiltonian=eps * (a + a.dag))
    lan = (
        derive_kramers_moyal(master, representation="positive-p")
        .truncate(2)
        .to_langevin()
    )
    assert lan.hamiltonian is None
    assert any("constant drive" in note for note in lan.diagnostics)


def test_unique_nonlinear_cross_term_is_factored_into_hamiltonian():
    al, be = sp.symbols("alpha_a beta_a", complex=True)
    H, diag = _extract_hamiltonian(sp.Matrix([be**2, al]), (al, be))
    assert diag == ()
    assert H == sp.Matrix([[0, sp.I * be], [sp.I, 0]])
    assert sp.simplify(
        sp.Matrix([be**2, al]) + sp.I * H * sp.Matrix([al, be])
    ) == sp.zeros(2, 1)


def test_ambiguous_nonlinear_cross_term_blocks_hamiltonian_extraction():
    z = sp.symbols("z0:6", complex=True)
    H, diag = _extract_hamiltonian(sp.Matrix([z[1] * z[2], 0, 0, 0, 0, 0]), z)
    assert H is None
    assert any("ambiguous Hamiltonian columns" in note for note in diag)


def test_unknown_representation_raises():
    (a,) = boson_modes("a")
    master = MasterEquation(modes=[a], hamiltonian=a.dag * a)
    with pytest.raises(RepresentationError):
        derive_kramers_moyal(master, representation="PositiveP-Wigner")


def test_undeclared_mode_reference_raises():
    a, b = boson_modes("a", "b")
    with pytest.raises(ModeError):
        MasterEquation(modes=[a], hamiltonian=a.dag * b)
    with pytest.raises(ModeError):
        MasterEquation(modes=[a], extra_terms=[LiouvillianTerm(1, right=b.dag)])


def test_master_equation_requires_modes_and_mode_ops():
    (a,) = boson_modes("a")
    with pytest.raises(MasterEquationError):
        MasterEquation(modes=[])
    with pytest.raises(MasterEquationError):
        MasterEquation(modes=[a * a])  # not a bare annihilation operator
    with pytest.raises(MasterEquationError):
        MasterEquation(modes=[a, a])
