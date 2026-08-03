"""Wigner-representation models: damping diffusion blocks, truncation records."""

import sympy as sp

from fpgen import (
    LindbladChannel,
    MasterEquation,
    boson_modes,
    derive_kramers_moyal,
)


def test_wigner_damping_diffusion_block():
    (a,) = boson_modes("a")
    kappa = sp.symbols("kappa", positive=True)
    master = MasterEquation(
        modes=[a], channels=[LindbladChannel(jump=a, rate=kappa)]
    )
    km = derive_kramers_moyal(master, representation="wigner")
    fpe = km.truncate(2)
    assert fpe.is_exact
    alpha, abar = fpe.variables
    A = fpe.drift
    assert sp.simplify(A[0] + kappa / 2 * alpha) == 0
    assert sp.simplify(A[1] + kappa / 2 * abar) == 0
    G = fpe.diffusion
    assert G[0, 0] == 0 and G[1, 1] == 0
    # Anomalous block: G_{alpha, abar} = kappa/2 (and symmetric).
    assert sp.simplify(G[0, 1] - kappa / 2) == 0
    assert sp.simplify(G[1, 0] - kappa / 2) == 0

    lan = fpe.to_langevin()
    B = lan.noise_factor
    assert sp.simplify(B * B.T - G) == sp.zeros(2)
    # For conjugate-augmented Cahill--Glauber variables, M and D are read
    # directly from the FPE blocks G[:n, :n] and G[:n, n:].
    assert lan.anomalous_correlation == sp.zeros(1)
    D = lan.normal_correlation
    assert D.shape == (1, 1)
    assert sp.simplify(D[0, 0] - kappa / 2) == 0


def test_wigner_kerr_has_third_order_and_truncation_is_recorded():
    (a,) = boson_modes("a")
    chi = sp.symbols("chi")
    master = MasterEquation(modes=[a], hamiltonian=chi / 2 * a.dag**2 * a**2)
    km = derive_kramers_moyal(master, representation="wigner")
    assert km.order == 3
    fpe = km.truncate(2)
    assert not fpe.is_exact
    assert fpe.original_max_order == 3
    assert fpe.discarded, "third-order terms must be recorded"
    assert all(sum(mu) == 3 for mu in fpe.discarded)
    text = repr(fpe)
    assert "approximate" in text
    assert "discarded" in text
