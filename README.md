# fpgen — Kramers–Moyal / Fokker–Planck / Langevin toolkit

fpgen derives the **Kramers–Moyal series** of a quantum master equation in a
phase-space representation (**Wigner**, **P**, **Q**, or **positive-P**),
truncates it to a **Fokker–Planck equation**, and converts that to an Itô
**c-number Langevin equation** on the augmented complex state — all with
exact SymPy symbolics.

**Authors: Yu Xue-Hao and Qiao Cong-Feng (University of Chinese Academy of Sciences, UCAS)**

```python
from fpgen import boson_modes, MasterEquation, LindbladChannel, derive_kramers_moyal

a, = boson_modes("a")
master = MasterEquation(modes=[a], hamiltonian=..., channels=[...])
km = derive_kramers_moyal(master, representation="positive-p")
fpe = km.truncate(2)
langevin = fpe.to_langevin()
```

> **Breaking change (0.2.0).** The legacy string-parsing API (`opr2fp`,
> `to_fp_normal_form`, exponent-tuple output) has been removed. See
> [Migration](#migration-from-01x) below.

## Installation

- Requirements: Python >= 3.10, SymPy (>=1.12, <2).

```powershell
git clone https://github.com/PolarisMegrez/fpgen.git
cd fpgen
pip install -e .
```

For development (tests):

```powershell
pip install -e .[dev]
pytest tests/
```

## Building a master equation

Modes are created explicitly and their order is declared on the
`MasterEquation` — it defines the phase-space variable order everywhere
downstream:

```python
import sympy as sp
from fpgen import boson_modes, MasterEquation, LindbladChannel, LiouvillianTerm

a, b = boson_modes("a", "b")
omega, kappa, g = sp.symbols("omega kappa g", positive=True)

master = MasterEquation(
    modes=[a, b],                                    # explicit global order
    hamiltonian=omega * a.dag * a + g * (a.dag * b + b.dag * a),
    channels=[LindbladChannel(jump=a, rate=kappa)],  # kappa*(a ρ a† - {a†a, ρ}/2)
    extra_terms=[LiouvillianTerm(1, left=a.dag, right=b)],  # a† ρ b, if needed
)
```

- Operators are non-commutative polynomials: `+`, `-`, `*`, integer powers,
  SymPy scalar coefficients (including `sp.I` and symbols), and `.dag`
  (reverses operator order and conjugates coefficients).
- `hamiltonian=H` contributes `-I [H, ρ]`.
- `LindbladChannel(jump=L, rate=γ)` contributes `γ (L ρ L† - {L†L, ρ}/2)`.
- `LiouvillianTerm(c, left, right)` is an explicit `c * left * ρ * right`.

## The three-step pipeline

### 1. `derive_kramers_moyal(master, representation=...) -> KramersMoyalSeries`

The single sign convention of the package is

```
dP/dt = sum_mu (-1)^|mu| / mu! * ∂^mu [ K_mu(z) P ]
```

over multi-indices `mu` on the augmented variable order `z`:

- Cahill–Glauber (`"wigner"`, `"p"`, `"q"`): `z = (α_1..α_N, ᾱ_1..ᾱ_N)`, with
  `ᾱ` treated as an independent complex variable.
- `"positive-p"`: `z = (α_1..α_N, β_1..β_N)`, where `β` is **not** the
  conjugate of `α`.

All orders are kept exactly — nothing is truncated during derivation.
Representation names are a strict enumeration with canonical aliases
(`"Positive P"`, `"Glauber-Sudarshan"`, `"Husimi Q"`, ...); unknown names
raise `RepresentationError`.

### 2. `km.truncate(2) -> FokkerPlanckEquation`

Truncation happens in flux form, after the exact normal → flux Leibniz
rearrangement (never in the normal form, which would lose lower-order
terms). Dropped coefficients are recorded:

```python
fpe.is_exact            # False if anything was dropped
fpe.discarded           # {mu: K_mu} of dropped terms
fpe.original_max_order  # highest order of the untruncated series
fpe.drift               # A_i = K_{e_i}
fpe.diffusion           # G_ij = K_{e_i + e_j}  (complex symmetric)
```

### 3. `fpe.to_langevin() -> LangevinEquation`

An Itô equation `dz/dt = A(z) + ξ` with `ξ = B η` (η independent real
Gaussian white noises):

```python
lan.drift                  # A(z)
lan.diffusion              # G_aug — full augmented FPE diffusion matrix
lan.noise_factor           # canonical B with B Bᵀ = G (not minimal)
lan.anomalous_correlation  # Cahill–Glauber: M = G_aug[:N, :N]
lan.normal_correlation     # Cahill–Glauber: D = G_aug[:N, N:]
lan.physical_hamiltonian   # Cahill–Glauber: upper-left N×N block of H_eff
lan.anomalous_hamiltonian  # Cahill–Glauber: upper-right N×N block of H_eff
lan.hamiltonian            # H_eff(z) with A = -i H_eff z, or None
lan.diagnostics            # why H_eff is None, when it is
```

- If the zero-order KM coefficient is nonzero (master equation does not
  conserve probability), `to_langevin()` raises
  `ProbabilityConservationError`.
- For Wigner/P/Q, the conjugate-augmented FPE directly determines
  `M = G_aug[:N, :N]` and `D = G_aug[:N, N:]`; these matrices are not
  reconstructed through `B`. In positive-P the second half is independent,
  so `normal_correlation` remains the realization-dependent `B B†`.
- `H_eff` is extracted only when the drift decomposes canonically: terms of
  `A_i` containing `z_i` go to the diagonal; otherwise a unique variable in
  the same augmented half is preferred as the matrix column. Constant drives
  or ambiguous factorizations yield `H_eff = None` with diagnostics. **Note:** `H_eff` includes
  dissipative drift, may be non-Hermitian, and is not the physical
  Hamiltonian of the input master equation.

All result objects offer structured attributes, readable `repr`,
`to_latex()`, and notebook `_repr_latex_`.

## Example: Kerr oscillator in positive-P

```python
import sympy as sp
from fpgen import boson_modes, MasterEquation, derive_kramers_moyal

a, = boson_modes("a")
chi = sp.symbols("chi")
master = MasterEquation(modes=[a], hamiltonian=chi / 2 * a.dag**2 * a**2)

km = derive_kramers_moyal(master, representation="positive-p")
fpe = km.truncate(2)          # exact: the series stops at order 2
lan = fpe.to_langevin()

lan.drift      # [-I*α²βχ, I*αβ²χ]
lan.diffusion  # diag(-I χ α², I χ β²)
```

## Migration from 0.1.x

| 0.1.x (removed) | 0.2.0 |
| --- | --- |
| `opr2fp("a^{\\dagger} \\rho a", representation='P')` | build a `MasterEquation` with `LiouvillianTerm(1, left=a.dag, right=a)` and call `derive_kramers_moyal(..., representation="p")` |
| `to_fp_normal_form(...)` tuple output | `KramersMoyalSeries.coefficients` (`{mu: K_mu}` SymPy mapping) |
| string parsing, float exponent tuples | structured `Op` algebra + SymPy coefficients |
| silent fallback of unknown representations to Wigner | `RepresentationError` |

## Notes

- SDEs use the Itô interpretation; no Stratonovich conversion is provided.
- The canonical noise factor `B` uses one column per nonzero diagonal entry
  of `G` and two phase-related columns per nonzero off-diagonal pair; it is
  valid for any complex-symmetric symbolic `G` but not minimal.

## Acknowledgements and License

This project was developed with the assistance of Copilot. Licensed under the MIT License. See LICENSE for details.
