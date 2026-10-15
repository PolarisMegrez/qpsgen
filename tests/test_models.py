"""Physical model tests: harmonic oscillator + loss, Kerr, parametric drive,
multimode linear coupling."""

import sympy as sp

from qpsgen import (
    LindbladChannel,
    MasterEquation,
    boson_modes,
    derive_kramers_moyal,
)


def test_harmonic_oscillator_with_loss_positive_p():
    (a,) = boson_modes("a")
    omega, kappa = sp.symbols("omega kappa", positive=True)
    master = MasterEquation(
        modes=[a],
        hamiltonian=omega * a.dag * a,
        channels=[LindbladChannel(jump=a, rate=kappa)],
    )
    km = derive_kramers_moyal(master, representation="positive-p")
    fpe = km.truncate(2)
    assert fpe.is_exact  # nothing above order 1 here
    alpha, beta = fpe.variables
    A = fpe.drift
    assert sp.simplify(A[0] - (-(sp.I * omega + kappa / 2) * alpha)) == 0
    assert sp.simplify(A[1] - ((sp.I * omega - kappa / 2) * beta)) == 0
    lan = fpe.to_langevin()
    assert lan.diffusion == sp.zeros(2)
    # Augmented effective Hamiltonian is diagonal.
    H = lan.hamiltonian
    assert H is not None
    assert sp.simplify(H[0, 1]) == 0 and sp.simplify(H[1, 0]) == 0
    # Round-trip: A = -i H_eff z.
    z = sp.Matrix(lan.variables)
    assert sp.simplify(lan.drift + sp.I * H * z) == sp.zeros(2, 1)
    # Diagonal entries carry both rotation and damping.
    assert sp.simplify(H[0, 0] - (omega - sp.I * kappa / 2)) == 0
    assert sp.simplify(H[1, 1] - (-omega - sp.I * kappa / 2)) == 0


def test_kerr_positive_p():
    (a,) = boson_modes("a")
    chi = sp.symbols("chi")
    master = MasterEquation(modes=[a], hamiltonian=chi / 2 * a.dag**2 * a**2)
    km = derive_kramers_moyal(master, representation="positive-p")
    assert km.order == 2
    fpe = km.truncate(2)
    assert fpe.is_exact
    alpha, beta = fpe.variables
    A = fpe.drift
    assert sp.simplify(A[0] - (-sp.I * chi * beta * alpha**2)) == 0
    assert sp.simplify(A[1] - (sp.I * chi * alpha * beta**2)) == 0
    G = fpe.diffusion
    assert sp.simplify(G[0, 0] - (-sp.I * chi * alpha**2)) == 0
    assert sp.simplify(G[1, 1] - (sp.I * chi * beta**2)) == 0
    assert G[0, 1] == 0 and G[1, 0] == 0
    lan = fpe.to_langevin()
    B = lan.noise_factor
    assert sp.simplify(B * B.T - G) == sp.zeros(2)
    assert lan.anomalous_correlation == G
    assert (
        lan.normal_correlation - (B * B.conjugate().T).applyfunc(sp.simplify)
    ).applyfunc(sp.simplify) == sp.zeros(2)
    assert lan.physical_hamiltonian is None


def test_quadratic_parametric_drive_cross_hamiltonian():
    (a,) = boson_modes("a")
    eps = sp.symbols("epsilon")
    H = eps / 2 * a.dag**2 + sp.conjugate(eps) / 2 * a**2
    master = MasterEquation(modes=[a], hamiltonian=H)
    lan = (
        derive_kramers_moyal(master, representation="positive-p")
        .truncate(2)
        .to_langevin()
    )
    H_eff = lan.hamiltonian
    assert H_eff is not None
    # Drift: A_alpha = -i eps beta, A_beta = i conj(eps) alpha.
    alpha, beta = lan.variables
    A = lan.drift
    assert sp.simplify(A[0] - (-sp.I * eps * beta)) == 0
    assert sp.simplify(A[1] - (sp.I * sp.conjugate(eps) * alpha)) == 0
    # Cross (anomalous) elements of the augmented Hamiltonian.
    assert sp.simplify(H_eff[0, 1] - eps) == 0
    assert sp.simplify(H_eff[1, 0] + sp.conjugate(eps)) == 0
    assert H_eff[0, 0] == 0 and H_eff[1, 1] == 0
    z = sp.Matrix(lan.variables)
    assert sp.simplify(A + sp.I * H_eff * z) == sp.zeros(2, 1)


def test_two_mode_linear_coupling_cross_hamiltonian():
    a, b = boson_modes("a", "b")
    g = sp.symbols("g")
    H = g * a.dag * b + sp.conjugate(g) * a * b.dag
    master = MasterEquation(modes=[a, b], hamiltonian=H)
    lan = (
        derive_kramers_moyal(master, representation="positive-p")
        .truncate(2)
        .to_langevin()
    )
    alpha_a, alpha_b, beta_a, beta_b = lan.variables
    A = lan.drift
    assert sp.simplify(A[0] - (-sp.I * g * alpha_b)) == 0
    assert sp.simplify(A[1] - (-sp.I * sp.conjugate(g) * alpha_a)) == 0
    assert sp.simplify(A[2] - (sp.I * sp.conjugate(g) * beta_b)) == 0
    assert sp.simplify(A[3] - (sp.I * g * beta_a)) == 0
    H_eff = lan.hamiltonian
    assert H_eff is not None
    assert sp.simplify(H_eff[0, 1] - g) == 0
    assert sp.simplify(H_eff[1, 0] - sp.conjugate(g)) == 0
    assert sp.simplify(H_eff[2, 3] + sp.conjugate(g)) == 0
    assert sp.simplify(H_eff[3, 2] + g) == 0
    z = sp.Matrix(lan.variables)
    assert sp.simplify(A + sp.I * H_eff * z) == sp.zeros(4, 1)


def test_wigner_van_der_pol_physical_hamiltonian_and_diffusion_blocks():
    a, b = boson_modes("a", "b")
    omega_a, omega_b, g, gamma_a, gamma_b, Gamma = sp.symbols(
        "omega_a omega_b g gamma_a gamma_b Gamma", real=True
    )
    master = MasterEquation(
        modes=[a, b],
        hamiltonian=(
            omega_a * a.dag * a + omega_b * b.dag * b + g * (a.dag * b + a * b.dag)
        ),
        channels=[
            LindbladChannel(jump=a.dag, rate=gamma_a),
            LindbladChannel(jump=a**2, rate=Gamma),
            LindbladChannel(jump=b, rate=gamma_b),
        ],
    )
    lan = derive_kramers_moyal(master, "wigner").truncate(2).to_langevin()
    alpha_a, _, abar_a, _ = lan.variables
    expected_h = sp.Matrix(
        [
            [
                omega_a + sp.I * (gamma_a / 2 + Gamma * (1 - abar_a * alpha_a)),
                g,
            ],
            [g, omega_b - sp.I * gamma_b / 2],
        ]
    )
    expected_d = sp.diag(gamma_a / 2 + Gamma * (2 * abar_a * alpha_a - 1), gamma_b / 2)
    assert (lan.physical_hamiltonian - expected_h).applyfunc(sp.simplify) == sp.zeros(2)
    assert lan.anomalous_hamiltonian == sp.zeros(2)
    assert (lan.normal_correlation - expected_d).applyfunc(sp.simplify) == sp.zeros(2)
    assert lan.anomalous_correlation == sp.zeros(2)


def test_wigner_pair_hopping_non_diagonal_factorization():
    a, b = boson_modes("a", "b")
    omega_a, omega_b, g, k, gamma_a, gamma_b = sp.symbols(
        "omega_a omega_b g k gamma_a gamma_b", real=True
    )
    master = MasterEquation(
        modes=[a, b],
        hamiltonian=(
            omega_a * a.dag * a
            + omega_b * b.dag * b
            + g * (a.dag * b + a * b.dag)
            + k * (a.dag**2 * b**2 + b.dag**2 * a**2)
        ),
        channels=[
            LindbladChannel(jump=a.dag, rate=gamma_a),
            LindbladChannel(jump=b, rate=gamma_b),
        ],
    )
    lan = derive_kramers_moyal(master, "wigner").truncate(2).to_langevin()
    alpha_a, alpha_b, abar_a, abar_b = lan.variables
    expected_h = sp.Matrix(
        [
            [omega_a + sp.I * gamma_a / 2, g + 2 * k * abar_a * alpha_b],
            [g + 2 * k * abar_b * alpha_a, omega_b - sp.I * gamma_b / 2],
        ]
    )
    assert (lan.physical_hamiltonian - expected_h).applyfunc(sp.simplify) == sp.zeros(2)
    assert lan.normal_correlation == sp.diag(gamma_a / 2, gamma_b / 2)
    assert lan.anomalous_correlation == sp.zeros(2)
