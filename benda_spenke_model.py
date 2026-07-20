"""
benda_spenke_model.py -- core physics for the reverse-recovery reproduction.

Reproduces the analytical/transport model of

    H. Benda and E. Spenke, "Reverse recovery processes in silicon power
    rectifiers," Proc. IEEE, vol. 55, no. 8, pp. 1331-1354, Aug. 1967.

The device is a p-s-n / p-i-n power rectifier: a wide, lightly doped middle
region of half-width `d` (so the base runs from x = -d to x = +d), swamped in
the forward state by high-level injection so that n(x,t) = p(x,t) >> doping.
Under that condition the excess plasma obeys the AMBIPOLAR diffusion equation

        dn/dt = D d^2n/dx^2 - n/tau                                    (Eq. 19)

with the ambipolar diffusion constant and length

        D = 2 mu_n mu_p /(mu_n + mu_p) * VT ,     L = sqrt(D tau).

High-injection current boundary conditions (derived from Eqs. 3-12,18 of the
paper -- at x=-d the p-contact carries only hole current, at x=+d the
n-contact carries only electron current).  Writing the electron current as
Jn(x) = mu_n/(mu_n+mu_p) * i + q D dn/dx, the requirements Jn(-d)=0 and
Jn(+d)=i give the Neumann gradients (i = +iF forward, i = -iR in reverse):

        dn/dx|_{-d} = - i * mu_n /((mu_n+mu_p) q D)
        dn/dx|_{+d} = + i * mu_p /((mu_n+mu_p) q D)

so the left gradient is (mu_n/mu_p)x steeper than the right -> the forward
profile (and the recovery) is asymmetric.  With mu_n = 3 mu_p (silicon) the
mobility-asymmetry parameter is

        B = (mu_n - mu_p)/(mu_n + mu_p) = 0.5     (the paper's "B = 0.5").

This module provides:
  * `Device`                -- material/geometry parameters, derived D, L, nbar
  * `forward_profile`       -- closed-form steady forward n(x)          (Eq. 21)
  * `n_analytic`            -- closed-form transient n(x,t)             (Eq. 36)
  * `solve_transient`       -- numerical ambipolar PDE solver (implicit)
  * `storage_times`         -- t0l, t0r from n(-d,t)=0, n(+d,t)=0    (Eq. 37,38)

Units: CGS-practical (cm, s, V, A/cm^2, cm^-3).  q in coulomb.
Normalised outputs (x/d, t/tau, n/nbar) are used for the figures, exactly as
in the paper, so most results depend only on d/L, B and iR/iF.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

Q = 1.602176634e-19       # C
VT = 0.025852             # kT/q at 300 K (V)  -- paper uses ~25 mV
EPS0 = 8.8541878128e-14   # vacuum permittivity (F/cm)


@dataclass
class Device:
    """A p-s-n / p-i-n power rectifier in high-level injection.

    Defaults are silicon (mu_n = 3 mu_p -> B = 0.5, eps_r = 11.9), matching the
    Benda & Spenke paper.  Other semiconductors are obtained by overriding the
    mobilities / permittivity / lifetime (see materials.py)."""
    d_over_L: float = 1.0          # half-width / ambipolar diffusion length
    mu_n: float = 1350.0           # electron mobility (cm^2/V.s)
    mu_p: float = 450.0            # hole mobility  (mu_n = 3 mu_p -> B = 0.5)
    tau: float = 1.0e-5            # high-injection lifetime (s)
    iF: float = 100.0             # forward current density (A/cm^2)
    eps_r: float = 11.9            # relative permittivity (Si)

    # derived
    D: float = field(init=False)   # ambipolar diffusion constant (cm^2/s)
    L: float = field(init=False)   # ambipolar diffusion length (cm)
    d: float = field(init=False)   # half-width (cm)
    B: float = field(init=False)   # mobility asymmetry (mu_n-mu_p)/(mu_n+mu_p)
    nbar: float = field(init=False)  # average forward concentration (cm^-3)
    eps: float = field(init=False)   # absolute permittivity (F/cm)

    def __post_init__(self):
        self.D = 2.0 * self.mu_n * self.mu_p / (self.mu_n + self.mu_p) * VT
        self.L = np.sqrt(self.D * self.tau)
        self.d = self.d_over_L * self.L
        self.B = (self.mu_n - self.mu_p) / (self.mu_n + self.mu_p)
        self.eps = self.eps_r * EPS0
        # iF = 2 d q nbar / tau  (stored charge destroyed once per lifetime, Eq.24)
        self.nbar = self.iF * self.tau / (2.0 * Q * self.d)

    # ---- boundary gradients dn/dx at -d and +d for a signed current i -----
    def gradients(self, i_current: float):
        """(gL, gR) = dn/dx at x=-d, +d for total current density i (A/cm^2,
        forward positive)."""
        s = self.mu_n + self.mu_p
        gL = -i_current * self.mu_n / (s * Q * self.D)
        gR = +i_current * self.mu_p / (s * Q * self.D)
        return gL, gR


# ---------------------------------------------------------------------------
# closed-form steady forward profile -- Eq. (21)
# ---------------------------------------------------------------------------
def forward_profile(dev: Device, x):
    """Steady-state high-injection forward concentration n(x) (cm^-3), Eq. (21):

        n(x) = iF L/(2 q D) [ cosh(x/L)/sinh(d/L) - B sinh(x/L)/cosh(d/L) ].

    Its average over [-d,+d] equals nbar (Eq. 24) -- an exact self-check."""
    x = np.asarray(x, float)
    L, d = dev.L, dev.d
    amp = dev.iF * L / (2.0 * Q * dev.D)
    return amp * (np.cosh(x / L) / np.sinh(d / L)
                  - dev.B * np.sinh(x / L) / np.cosh(d / L))


# ---------------------------------------------------------------------------
# closed-form transient during the quasi-neutral recovery phase -- Eq. (36)
# ---------------------------------------------------------------------------
def n_analytic(dev: Device, x, t, iR, nmax: int = 200):
    """Transient concentration n(x,t) during 0 < t < t0 (quasi-neutral phase),
    the closed-form Eq. (36).  x, t may be arrays (broadcast).  iR = reverse
    current density (>0).  Returns n in cm^-3.

    Valid until n reaches zero at a border (space charge then forms); beyond
    that the formula gives the unphysical negative values the paper uses to
    *define* the storage times t0l, t0r (Eq. 37, 38)."""
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    L, d, B = dev.L, dev.d, dev.B
    tau = dev.tau
    iF = dev.iF
    LD = L / d                                  # (L/d)
    fr = iR / (iF + iR)

    # symmetric (cosine) series + steady cosh term
    rhs = np.exp(-t / tau)
    for v in range(1, nmax + 1):
        lam = 1.0 + (v * np.pi * LD) ** 2
        rhs = rhs + 2.0 * ((-1) ** v) / lam * np.cos(v * np.pi * x / d) \
            * np.exp(-lam * t / tau)
    rhs = rhs - fr * (d / L) / np.sinh(d / L) * np.cosh(x / L)

    # antisymmetric (sine) series + steady sinh term, weighted by B
    anti = np.zeros(np.broadcast(x, t).shape, float)
    for v in range(0, nmax):
        w = v + 0.5
        lam = 1.0 + (w * np.pi * LD) ** 2
        anti = anti + 2.0 * ((-1) ** v) / lam * np.sin(w * np.pi * x / d) \
            * np.exp(-lam * t / tau)
    anti = anti - fr * (d / L) / np.cosh(d / L) * np.sinh(x / L)
    rhs = rhs - B * anti

    return dev.nbar * (iF + iR) / iF * rhs


# ---------------------------------------------------------------------------
# numerical ambipolar diffusion solver (implicit, Neumann BCs)
# ---------------------------------------------------------------------------
def solve_transient(dev: Device, iR, t_end, nx=401, nt=4000,
                    recombination=True, i_current=None, n0=None):
    """Integrate  dn/dt = D n_xx - n/tau  on [-d,+d] with high-injection
    Neumann BCs for reverse current iR (or an explicit signed `i_current`).

    Backward-Euler (unconditionally stable -- robust across the gradient sign
    flip at t=0).  Returns (x, t, N) with N[k] = n(x, t_k) in cm^-3.
    Initial condition defaults to the steady forward profile (Eq. 21)."""
    d, D, tau = dev.d, dev.D, dev.tau
    x = np.linspace(-d, d, nx)
    dx = x[1] - x[0]
    dt = t_end / nt

    n = forward_profile(dev, x).copy() if n0 is None else np.asarray(n0, float).copy()

    i_cur = -iR if i_current is None else i_current
    gL, gR = dev.gradients(i_cur)
    inv_tau = (1.0 / tau) if recombination else 0.0

    # operator A: A n = D n_xx - n/tau, with Neumann ghost nodes folded in.
    r = D / dx ** 2
    main = np.full(nx, -2.0 * r - inv_tau)
    lower = np.full(nx - 1, r)
    upper = np.full(nx - 1, r)
    # left boundary (ghost n_{-1} = n_1 - 2 dx gL):  couples 0<->1 with 2r
    upper[0] = 2.0 * r
    # right boundary (ghost n_{N+1} = n_{N-1} + 2 dx gR)
    lower[-1] = 2.0 * r
    A = sp.diags([lower, main, upper], [-1, 0, 1], format="csc")

    # constant flux vector from the Neumann ghosts
    c = np.zeros(nx)
    c[0] = -2.0 * r * dx * gL          # = -2 D gL / dx
    c[-1] = +2.0 * r * dx * gR         # = +2 D gR / dx

    # backward Euler:  (I - dt A) n^{k+1} = n^k + dt c
    M = (sp.identity(nx, format="csc") - dt * A).tocsc()
    lu = spla.splu(M)

    times = np.linspace(0, t_end, nt + 1)
    N = np.empty((nt + 1, nx))
    N[0] = n
    for k in range(nt):
        n = lu.solve(n + dt * c)
        N[k + 1] = n
    return x, times, N


# ---------------------------------------------------------------------------
# storage times t0l, t0r  -- Eq. (37), (38)
# ---------------------------------------------------------------------------
def storage_times(dev: Device, iR, method="analytic", t_end=None,
                  nmax=200, **kw):
    """First times the border concentrations reach zero:
        n(-d, t0l) = 0    (left  border, Eq. 37)
        n(+d, t0r) = 0    (right border, Eq. 38)
    Returns (t0l, t0r) in seconds.  method = 'analytic' (Eq. 36 root find) or
    'numeric' (PDE zero crossing)."""
    tau = dev.tau
    d = dev.d

    if method == "analytic":
        from scipy.optimize import brentq

        def nborder(t, xb):
            return float(n_analytic(dev, xb, t, iR, nmax=nmax))

        # bracket: search outward in t/tau
        def find(xb):
            # wide bracket: very asymmetric materials (GaN) empty the fast
            # border in << 1e-4 tau at large iR/iF, so start the search low.
            ts = np.geomspace(1e-6, 5.0, 700) * tau
            vals = np.array([nborder(t, xb) for t in ts])
            sgn = np.where(vals <= 0)[0]
            if len(sgn) == 0:
                return np.nan
            j = sgn[0]
            if j == 0:
                return ts[0]
            return brentq(lambda t: nborder(t, xb), ts[j - 1], ts[j], xtol=tau * 1e-6)

        return find(-d), find(+d)

    # numeric
    if t_end is None:
        t_end = 2.0 * tau
    x, t, N = solve_transient(dev, iR, t_end, **kw)
    nl, nr = N[:, 0], N[:, -1]

    def cross(sig):
        idx = np.where(sig <= 0)[0]
        if len(idx) == 0:
            return np.nan
        j = idx[0]
        if j == 0:
            return t[0]
        # linear interpolation of the zero crossing
        t0, t1 = t[j - 1], t[j]
        y0, y1 = sig[j - 1], sig[j]
        return t0 - y0 * (t1 - t0) / (y1 - y0)

    return cross(nl), cross(nr)


if __name__ == "__main__":
    print("=== Benda-Spenke 1967 -- model self-checks ===")
    dev = Device(d_over_L=1.0)
    print(f"mu_n/mu_p            = {dev.mu_n/dev.mu_p:.3f}   -> B = {dev.B:.3f}  "
          f"(paper B = 0.5)")
    print(f"D (ambipolar)        = {dev.D:.3f} cm^2/s")
    print(f"L (ambipolar length) = {dev.L*1e4:.2f} um")
    print(f"d (half-width)       = {dev.d*1e4:.2f} um   (d/L = {dev.d_over_L})")
    print(f"nbar (avg fwd conc.) = {dev.nbar:.3e} cm^-3")

    x = np.linspace(-dev.d, dev.d, 501)
    nfwd = forward_profile(dev, x)
    avg = np.trapz(nfwd, x) / (2 * dev.d)
    print(f"\nforward-profile average / nbar = {avg/dev.nbar:.6f}  (should be 1.0)")

    # analytic vs numeric transient at a few times, iR = 10 iF
    iR = 10 * dev.iF
    print("\nanalytic (Eq.36) vs numeric PDE, iR=10iF, border values n/nbar:")
    for tf in (0.0, 0.01, 0.03):
        t = tf * dev.tau
        nlA = n_analytic(dev, -dev.d, t, iR) / dev.nbar
        nrA = n_analytic(dev, +dev.d, t, iR) / dev.nbar
        print(f"  t/tau={tf:4.2f} : analytic  L={nlA:+.4f}  R={nrA:+.4f}")

    print("\nstorage times (analytic), iR=10iF:")
    t0l, t0r = storage_times(dev, iR, method="analytic")
    print(f"  t0l/tau = {t0l/dev.tau:.4f}   t0r/tau = {t0r/dev.tau:.4f}  "
          f"(t0r > t0l expected)")
    t0lN, t0rN = storage_times(dev, iR, method="numeric", nt=8000)
    print(f"  numeric : t0l/tau = {t0lN/dev.tau:.4f}   t0r/tau = {t0rN/dev.tau:.4f}")
