# Reverse Recovery in Silicon Power Rectifiers — simulation

From-scratch reproduction of the analytical/transport core of

> H. Benda and E. Spenke, **"Reverse recovery processes in silicon power
> rectifiers,"** *Proceedings of the IEEE*, **55**(8), 1331–1354, Aug. 1967.

Source PDF: `Reverse_recovery_processes_in_silicon_power_rectifiers  IEEE.pdf`
(this folder). Everything — code, figures, report — lives in this one folder.

## The physics in one paragraph

A p-s-n / p-i-n power rectifier under forward load is flooded by a high-level
electron–hole plasma (`n = p >>` doping). That plasma obeys the **ambipolar
diffusion equation** `∂n/∂t = D ∂²n/∂x² − n/τ` with high-injection current
boundary conditions in which the left (p-side) gradient is `μn/μp = 3×` steeper
than the right. When the current is reversed, carriers are swept out from both
edges; the left border empties first. The instants the border concentrations
hit zero — the **storage times** `t0l`, `t0r` — mark where space-charge zones
form and the reverse voltage begins to rise. Those storage times, and the
carrier profiles that lead to them, are what this reproduction computes, two
independent ways.

## What's reproduced

### Stage 1 — quasi-neutral ambipolar plasma (Sections II–III)

| paper | this repo | result |
|---|---|---|
| **Fig. 4** | `fig4_forward_profile.py` | stationary forward profile (asymmetric catenary), Eq. 21 |
| **Fig. 7** | `fig7_recombination_decay.py` | decay by recombination, `iR = 0` |
| **Fig. 8** | `fig8_sweepout.py` | sweeping-out by strong reverse current `iR = 10 iF` |
| **Fig. 9** | `fig9_storage_time.py` | storage times `t0l, t0r` vs `iR/iF`, `d=L` and `d=L/2` |

Each is produced by **three independent solvers** — the paper's closed forms
(Eq. 21 forward, Eq. 36 transient), a numerical ambipolar PDE, and a
**Laplace-transform (operational) closed form inverted by the Talbot method**
(`laplace_solver.py`) — which agree throughout (the primary correctness check).
`fig8_three_methods.py` overlays all three: Laplace ≡ Eq. 36 to **2.5×10⁻⁸**
(same closed form, independent inversion), PDE to **3.5×10⁻⁴** (discretization).
The Laplace solution's poles `s = −1/τ − D(mπ/d)²` are exactly Eq. 36's decay
rates — an analytic cross-check.

### Stage 2 — space charge & voltage recovery (Sections IV–VII)

| paper | this repo | result |
|---|---|---|
| **Figs. 11–13** | `fig_scz_growth.py` | sweeping-out boundaries advance inward; widths grow linearly, meet off-centre |
| **Figs. 14–16** | `fig14_sc_field.py` | √-field & falling carrier (p-i-n); doping → linear/Schottky field |
| **Sec. V–VI** | `fig_voltage_recovery.py` | reverse-recovery voltage `U_rect(t) ∝ (t−t0)^{3/2}` (p-i-n) / `^2` (Schottky) |

Core module: `space_charge_model.py` (Eqs. 46–62). Key checks reproduced from
the paper's own text: boundary-speed ratio `= μn/μp` exactly; **left voltage
share = 90.0 %** for silicon (`b=3`); fitted voltage exponents **1.53** (p-i-n,
theory 3/2) and **2.02** (Schottky, theory 2); peak recovery voltage of several
kV over a fraction of a lifetime.

### Extension — wide-bandgap materials (`fig_materials_compare.py`)

The model is re-run for **4H-SiC** and **GaN** with cited literature parameters
(`materials.py`). Going Si → SiC → GaN, the mobility asymmetry
`B = (μn−μp)/(μn+μp)` rises **0.50 → 0.78 → 0.94**, so:

- the forward plasma and the whole recovery grow increasingly **one-sided**
  (storage-time ratio `t0r/t0l` and the left-zone voltage share both climb —
  90.0 % → 98.4 % → 99.9 %);
- the recovery space charge can hold **~70–130× more voltage** before dynamic
  avalanche — the Baliga figure of merit `ε·μ·Ec³` emerging from the
  reverse-recovery physics.

GaN's hole mobility (hence `B` and its hole-zone avalanche margin) is the most
uncertain input and is flagged with its sensitivity range in `materials.py`.

## Verification highlights (`python verify.py`)

- `B = (μn−μp)/(μn+μp) = 0.500` (paper's `B = 0.5`).
- forward-profile average `= n̄` exactly (Eq. 24): `<n>/n̄ = 1.000000`.
- recombination decay `iR=0`: spatial average `= e^{−t/τ}` to 4 places
  (0.368 / 0.135 / 0.050 at `t/τ = 1/2/3`).
- closed-form Eq. 36 vs numerical PDE storage times agree to **< 0.5 %**
  at every `(d/L, iR/iF)`; and `t0r > t0l` always (n-side empties later).

| d/L | iR/iF | t0l/τ | t0r/τ |
|---|---|---|---|
| 1.0 | 2 | 0.120 | 0.354 |
| 1.0 | 10 | 0.0083 | 0.0229 |
| 1.0 | 50 | 0.0004 | 0.0011 |
| 0.5 | 10 | 0.0167 | 0.0935 |

## Files

```
benda_spenke_model.py        Stage-1 core: Device, forward_profile (Eq.21),
                             n_analytic (Eq.36), solve_transient (PDE),
                             storage_times (Eq.37/38)
laplace_solver.py            3rd solver: Laplace-domain N(x,s) + Talbot inverse
fig8_three_methods.py        PDE / Eq.36 / Laplace overlay + agreement plot
space_charge_model.py        Stage-2 core: boundary kinematics (Eq.46-49),
                             sqrt-field / voltage (Eq.54-61), doped/Schottky
                             solver (Eq.62), voltage_recovery
fig4_forward_profile.py      Fig. 4       fig7_recombination_decay.py  Fig. 7
fig8_sweepout.py             Fig. 8       fig9_storage_time.py         Fig. 9
fig_scz_growth.py            Sec. IV      fig14_sc_field.py            Sec. V/VI
fig_voltage_recovery.py      Sec. V/VI reverse-recovery voltage
materials.py                 cited Si / 4H-SiC / GaN parameter sets
fig_materials_compare.py     4-panel Si/SiC/GaN comparison
fig_materials_extra.py       waveform, Fig.9-style storage times, avalanche margin
fig_materials_fixedgeom.py   fixed 100um-base comparison (comparable peaks)
fig_conformal_edge.py        conformal map: 2-D space-charge edge field crowding
methods_comparison.py        accuracy/iterations/convergence of the 3 solvers
verify.py                    model-vs-paper / self-consistency table
compare.py                   side-by-side vs paper + accuracy table
build_report.py              assembled PDF (renders paper crops on the fly)
benda_spenke_full.tex        FULL LaTeX report (theory->derivation->plots, 12pp)
derivation.md                full derivation (Eq. 19-62, section by section)
README.md                    this file
fig*.png / compare_*.png     generated figures
benda_spenke_report.pdf      assembled report
```

## Running

Plain Python 3 with numpy + scipy + matplotlib + pymupdf (same interpreter as
the sibling reproductions:
`C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe`).

```bash
python benda_spenke_model.py   # core self-checks
python verify.py               # full comparison table
python fig4_forward_profile.py
python fig7_recombination_decay.py
python fig8_sweepout.py
python fig9_storage_time.py
python build_report.py         # -> benda_spenke_report.pdf (matplotlib PdfPages)
python methods_comparison.py   # solver accuracy/convergence benchmark
python fig_conformal_edge.py   # conformal 2-D edge-field extension
```

Full LaTeX report (needs a TeX toolchain; this environment uses WSL) -- run
pdflatex three times for the ToC/refs:

```bash
wsl bash -c "cd /mnt/e/MATERIAL/PIN && pdflatex benda_spenke_full.tex && \
             pdflatex benda_spenke_full.tex && pdflatex benda_spenke_full.tex"
# -> benda_spenke_full.pdf  (theory -> derivation -> methods -> simulation -> plots)
```

## Notes

- **Units:** CGS-practical (cm, s, V, A/cm², cm⁻³). Results for Figs. 7–9 are
  normalised (`n/n̄`, `x/d`, `t/τ`) and so depend only on `d/L`, `B`, `iR/iF` —
  exactly the paper's parameterisation.
- **Two methods on purpose:** `n_analytic` (Eq. 36) and `solve_transient` (an
  implicit finite-volume PDE with the derived Neumann BCs) never share code;
  their agreement validates both. The numerical steady state also matches
  Eq. 21 to RMS 2×10⁻⁶ of `n̄`.
- **Scope:** the full analytical arc of the paper (Sections II–VII) is now
  simulated — the quasi-neutral ambipolar plasma (II–III) *and* the
  space-charge growth + voltage recovery (IV–VII, `space_charge_model.py`,
  `derivation.md` §7). Only Sections VIII–X (oscilloscope traces, lifetime
  measurements, device technology) are left out as measured data.
```
