# Derivation — Reverse Recovery of Silicon Power Rectifiers

Reproduces the physics of

> H. Benda and E. Spenke, "Reverse recovery processes in silicon power
> rectifiers," *Proceedings of the IEEE*, **55**(8), 1331–1354, Aug. 1967.

Focus: the analytically tractable, first-principles core — the high-injection
plasma in the middle region during the **quasi-neutral** phase of switching
(the paper's Sections II–III), which fixes the forward carrier profile, the
carrier removal by recombination and by sweeping-out, and the **storage
(delay) times** `t0l`, `t0r` at which the space-charge zones begin to form.
The later space-charge / voltage-recovery analysis (Sections IV–VII) is
summarised but not simulated — see *Scope* at the end.

Units: CGS-practical (cm, s, V, A/cm², cm⁻³).

---

## 0. Device and regime

A p-s-n (or p-i-n) power rectifier has a wide, lightly doped **middle region**
of half-width `d`, from `x = -d` (p-side) to `x = +d` (n-side). Under forward
load the region is "swamped" by high-level injection,

```
n(x,t) = p(x,t)  >>  doping        (Eq. 1, 2)
```

so the excess electron–hole plasma moves together (ambipolar transport) and the
doping drops out of the transport problem. This is why one model serves the
p-i-n, p-s-n and p-s_n-n structures alike.

At `t = 0` the external circuit reverses the current. The paper studies the
tractable limiting case

```
iR = const  >>  iF                 (Eq. 41)
```

(large EMF, large series resistance → constant reverse current), which is where
the sweeping-out physics is cleanest.

---

## 1. The ambipolar diffusion equation (Eq. 19)

Insert the drift–diffusion currents

```
Jp = q mu_p p E - q Dp dp/dx ,   Jn = q mu_n n E + q Dn dn/dx        (Eq. 7,8)
```

into the continuity equations with recombination `R = n/τ` (high injection),
impose `n = p`, and eliminate the field `E`. The result is the ambipolar
diffusion equation

```
dn/dt = D d²n/dx² - n/τ ,                                            (Eq. 19)
```

with the **ambipolar diffusion constant** and **length**

```
D = (n+p)/(n/Dp + p/Dn) |_{n=p} = 2 Dn Dp/(Dn+Dp) = 2 mu_n mu_p/(mu_n+mu_p)·VT
L = sqrt(D τ).
```

For silicon `mu_n ≈ 3 mu_p`, giving the paper's mobility-asymmetry parameter

```
B = (mu_n - mu_p)/(mu_n + mu_p) = 0.5.
```

Numerically (`mu_n=1350`, `mu_p=450`, `τ=10 µs`): `D = 17.45 cm²/s`,
`L = 132 µm` — a realistic power-rectifier base.

---

## 2. High-injection boundary conditions (Eqs. 3–12, 18)

At the borders the current is carried by **one** carrier type:

```
Jp(-d)=i, Jn(-d)=0     (p-contact, left)
Jp(+d)=0, Jn(+d)=i     (n-contact, right)                            (Eq. 3-6)
```

Writing the electron current with the field eliminated (using `n=p`, Einstein,
and `i = q(mu_n+mu_p)nE + q(Dn-Dp)dn/dx`):

```
Jn(x) = mu_n/(mu_n+mu_p) · i + q D dn/dx.                            (★, derived)
```

Applying `Jn(-d)=0` and `Jn(+d)=i` gives the **Neumann gradients** (current `i`
positive in the forward direction):

```
dn/dx|_{-d} = - i · mu_n /((mu_n+mu_p) q D)
dn/dx|_{+d} = + i · mu_p /((mu_n+mu_p) q D).
```

The left gradient is `mu_n/mu_p = 3×` steeper than the right — the origin of
**every** asymmetry in the paper (Figs. 4, 5, 8, 9). At switching, `i` flips
from `+iF` to `-iR`, so both border gradients flip sign: the plasma is pulled
out from both edges. This `(★)` identity and these BCs are the whole basis of
the simulation and are what make the closed forms below reduce correctly (§3–4).

---

## 3. Stationary forward profile (Eq. 21)

Steady state drops `dn/dt`, leaving `L² n'' = n`, so
`n(x) = C1 cosh(x/L) + C2 sinh(x/L)`. Fixing `C1, C2` with the two forward
gradients (`i = iF`) gives the closed form

```
n(x) = iF L/(2 q D) [ cosh(x/L)/sinh(d/L) - B sinh(x/L)/cosh(d/L) ].   (Eq. 21)
```

**Two exact self-checks** (both verified in `verify.py`):

* the symmetric `cosh` term integrates to the average concentration `n̄`, and
  `iF = 2 d q n̄ / τ` — the stored charge `2 d q n̄` is destroyed once per
  lifetime (Eq. 24). The reproduction gives `<n>/n̄ = 1.000000`.
* the profile is asymmetric (higher on the p-side) exactly by the factor `B`,
  matching Fig. 4.

The middle-region voltage `Um` (Eq. 33, Fig. 6) follows by integrating the
field `E(x)` — an ohmic term (∝ 75 % of the drop for `B=1/2`) plus a "built-in"
term; the small-`d/L` approximation is `Um ≈ 3V·(d/L)²·(1/α)` (Eq. 34–35). This
is documented but the voltage build-up is outside the simulated core (see
*Scope*).

---

## 4. Transient during the quasi-neutral phase (Eq. 36)

Solving Eq. (19) with initial condition = the forward profile (Eq. 21) and the
**reverse** border gradients (`i = -iR`) gives the closed-form series

```
(iF/(iF+iR))·n(x,t)/n̄
  = exp(-t/τ)
  + 2 Σ_{ν≥1} (-1)^ν/(1+(νπL/d)²) · cos(νπx/d) · exp{-[1+(νπL/d)²] t/τ}
  - (iR/(iF+iR))·(d/L)/sinh(d/L)·cosh(x/L)
  - B { 2 Σ_{ν≥0} (-1)^ν/(1+((ν+½)πL/d)²)·sin((ν+½)πx/d)·exp{-[1+((ν+½)πL/d)²]t/τ}
        - (iR/(iF+iR))·(d/L)/cosh(d/L)·sinh(x/L) }.                   (Eq. 36)
```

Structure: an exponentially decaying **symmetric** (cosine) part carrying the
recombination decay + the symmetric sweep-out, and a `B`-weighted
**antisymmetric** (sine) part carrying the mobility asymmetry.

**Consistency checks (proved analytically, confirmed numerically):**

* `t = 0` → the transient exponentials are 1 and the Fourier series sum to the
  `cosh`/`sinh` terms, so Eq. (36) collapses exactly to the forward profile
  Eq. (21). (`verify.py`: RMS = 3×10⁻⁴ of `n̄`, limited only by series
  truncation at 200 terms.)
* `iR = 0` → the steady terms vanish and the fundamental mode decays as
  `exp(-t/τ)`; the reproduction's spatial average hits `0.368/0.135/0.050` at
  `t/τ = 1/2/3` — exactly `e^{-1}, e^{-2}, e^{-3}` (Fig. 7).
* `t → ∞` with `iR>0` → the profile goes **negative** — the unphysical signal
  that quasi-neutrality has failed and a space-charge zone must form.

---

## 5. Numerical ambipolar PDE (independent method)

`solve_transient` integrates Eq. (19) directly by an implicit (backward-Euler)
finite-volume scheme on `[-d,+d]`, with the §2 Neumann gradients folded in via
second-order ghost nodes. Backward Euler is chosen for robustness across the
**discontinuous gradient flip** at `t = 0`. This is a genuinely independent
solver (no use of Eq. 21/36), and it reproduces:

* the forward steady state to **RMS 2×10⁻⁶ of `n̄`** vs Eq. (21);
* the transient profiles of Figs. 7 and 8 on top of the Eq. (36) dots;
* the storage times of Fig. 9 to **< 0.5 %** vs the analytic roots.

The agreement of two independent methods (closed form vs PDE) is the primary
validation, in lieu of a test suite.

---

## 6. Storage (delay) times (Eqs. 37, 38 — Fig. 9)

Because the border concentrations fall fast, the instant a space-charge zone
forms is well defined by the border concentration reaching zero:

```
n(-d, t0l) = 0     (left,  p-side)          (Eq. 37)
n(+d, t0r) = 0     (right, n-side)          (Eq. 38)
```

`storage_times` finds these two roots (from Eq. 36, or from the PDE
zero-crossing). Because the left gradient is `3×` steeper, **the left border
always empties first**: `t0l < t0r` throughout — the two unequal delay times
`t0l`, `t0r` the paper emphasises (Fig. 5). Fig. 9 plots `t0l/τ`, `t0r/τ` vs
`iR/iF` (log-log) for `d/L = 1` and `1/2`; at large ratios the sweep-out
dominates and `t0 ∝ (iR/iF)^{-1}`. Representative reproduced values:

| d/L | iR/iF | t0l/τ | t0r/τ |
|---|---|---|---|
| 1.0 | 2  | 0.120 | 0.354 |
| 1.0 | 10 | 0.0083 | 0.0229 |
| 1.0 | 50 | 0.0004 | 0.0011 |
| 0.5 | 10 | 0.0167 | 0.0935 |

---

## 7. Space-charge growth and voltage recovery (Sections IV–VII)

Once a border concentration reaches zero (at `t0l`, `t0r`) the quasi-neutral
model of §1–6 hands off to the **space-charge stage**, implemented in
`space_charge_model.py`.

### 7.1 Sweeping-out boundaries (Section IV, Eqs. 46–49)

Two boundaries advance inward from `x = ∓d`. In the swamped zone the hole and
electron currents split by the mobility ratio (`jp:jn = μp:μn`); in each
space-charge zone a single carrier carries all of `iR`. The current **jump** at
a boundary equals the charge removal rate `q·n̄·(dw/dt)`, giving the constant
boundary speeds

```
dw_l/dt = μn/(μn+μp) · iR/(q n̄)      (left,  holes)     Eq. (46)
dw_r/dt = μp/(μn+μp) · iR/(q n̄)      (right, electrons) Eq. (47)
```

so `w_l(t) = (dw_l/dt)(t−t0l)`, `w_r(t) = (dw_r/dt)(t−t0r)` (Eqs. 48–49). The
left zone grows `μn/μp = 3×` faster **and** starts earlier (`t0l < t0r`), so the
boundaries meet **right of centre** (reproduced: they meet at `x ≈ +72 µm`,
`t/τ ≈ 0.112` for `iR=10iF`) — the asymmetry of Fig. 5. Verified: the modelled
speed ratio is exactly `μn/μp = 3.00`.

### 7.2 p-i-n voltage (Section V, Eqs. 50–61)

In a swept-out zone the single carrier drifts in the field of its own diffuse
space charge. Poisson `dE/dx = q p/ε` with the field-only current
`iR = q μ p |E|` (diffusion and displacement current negligible — Appendices
I–II) gives, with `E = 0` at the inner boundary (Eq. 53),

```
F(ξ) = sqrt( 2 iR ξ /(ε μ) )                      (square-root field, Eq. 54)
p(ξ) = iR/(q μ F) ∝ ξ^{-1/2}                        (carrier falls outward, Eq. 55)
```

Integrating the field across a zone of width `w`,

```
U_sc = ∫₀^w F dξ = (2/3) sqrt(2 iR/(ε μ)) · w^{3/2}.      Eq. (58,59)
```

Left uses `μ = μp` (holes), right `μ = μn` (electrons). Since `w ∝ (t−t0)`,

```
U_rect(t) = U_scl + U_scr  ∝  (t − t0)^{3/2}
```

— the paper's central, experimentally-confirmed voltage law. **Two exact
cross-checks against the paper's own text:**

* the left zone carries `sqrt(μn/μp)·(μn/μp)^{3/2} = √3·3^{3/2} = 9` times the
  right, i.e. **90.0 %** of the voltage for `b=3` — the paper's stated figure
  (still higher in practice because `t0l < t0r`; the reproduction gives ~92–94 %);
* the fitted exponent of the modelled `U_rect(t−t0l)` is **1.53** (theory 3/2).

The modelled peak reaches several kV over a fraction of a lifetime — the reverse
recovery overshoot (the idealisation holds `iR` constant; in a real circuit the
voltage saturates at the source EMF).

### 7.3 Doped and diffused junctions (Sections VI–VII)

With finite doping, Poisson gains a fixed-charge term
`dF/dξ = (q/ε)(iR/(q μ F) + N)` (Eq. 62). `field_doped` integrates this for any
`N` (scalar for the uniform p-s-n of Section VI, or an exponential `N(ξ)` for
the diffused junction of Section VII). When the mobile carrier falls below the
doping, the field becomes **linear** — a Schottky (constant-charge) zone — and

```
U_sc ∝ w²  ∝  (t − t0)²        (Schottky, Section VI)
```

The reproduction recovers the `N→0` `^{3/2}` law and, for `N = 2×10¹⁵`, a fitted
`w²` exponent of **2.02** (theory 2), reproducing the paper's statement that a
Schottky zone (`U ∝ (d−a)²`) forms and soon dominates the rectifier voltage.

### 7.4 Still out of scope

Only Section VIII–X (experimental oscilloscope traces, lifetime measurements,
and device-technology discussion) are not reproduced — they are measured data,
not analytical results. The full analytical arc of the paper (Sections II–VII)
is now simulated.
