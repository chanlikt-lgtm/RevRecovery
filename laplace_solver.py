"""
laplace_solver.py -- THIRD independent solver for the reverse-recovery
transient: the Laplace-transform (operational) closed form + numerical (Talbot)
inversion.

Same problem, same conditions as benda_spenke_model.py:
    dn/dt = D d^2n/dx^2 - n/tau  on [-d,+d],
    initial condition  n(x,0) = forward profile (Eq. 21),
    step-change Neumann gradients at t=0 (forward current iF -> reverse -iR).

Laplace transform in time,  N(x,s) = L{n(x,t)} :

    D N'' - (s + 1/tau) N = -n0(x)      =>   N'' - k^2 N = -n0(x)/D,
    k(s) = sqrt((s + 1/tau)/D).

Because k^2 - 1/L^2 = s/D exactly (L^2 = D tau), the particular solution is
simply n0(x)/s, and the step-change Neumann BCs (gradient jump from the forward
to the reverse value, transform -> (g_rev - g_fwd)/s) give the closed form

    N(x,s) = n0(x)/s
             - (iF+iR)/(2 q D s k) [ cosh(kx)/sinh(kd) - B sinh(kx)/cosh(kd) ].

Its poles sinh(kd)=0 / cosh(kd)=0 sit at s = -1/tau - D(m pi/d)^2 and
s = -1/tau - D((m+1/2) pi/d)^2 -- exactly the decay rates of the Fourier series
Eq. (36).  So this is the same solution expressed through operational calculus;
inverting numerically (fixed Talbot method, Abate & Valko 2004) is a genuinely
independent numerical route -- it shares no code with the eigenfunction series
(n_analytic) or the finite-difference PDE (solve_transient).

Units: CGS-practical (cm, s, V, A/cm^2, cm^-3).
"""
from __future__ import annotations
import numpy as np
from benda_spenke_model import Device, forward_profile, Q


# ---------------------------------------------------------------------------
# Laplace-domain solution N(x,s)
# ---------------------------------------------------------------------------
def _k(dev: Device, s):
    return np.sqrt((s + 1.0 / dev.tau) / dev.D)


def _cosh_over_sinh(k, x, d):
    """cosh(kx)/sinh(kd), overflow-safe for Re(k) >= 0 and |x| <= d."""
    e1 = np.exp(k * (x - d))
    e2 = np.exp(-k * (x + d))
    e3 = np.exp(-2.0 * k * d)
    return (e1 + e2) / (1.0 - e3)


def _sinh_over_cosh(k, x, d):
    """sinh(kx)/cosh(kd), overflow-safe for Re(k) >= 0 and |x| <= d."""
    e1 = np.exp(k * (x - d))
    e2 = np.exp(-k * (x + d))
    e3 = np.exp(-2.0 * k * d)
    return (e1 - e2) / (1.0 + e3)


def N_hat(dev: Device, x, s, iR):
    """Laplace transform N(x,s) of n(x,t) at position x (scalar), for complex
    s (scalar or array).  iR = reverse current density (>0)."""
    k = _k(dev, s)
    d = dev.d
    n0 = float(forward_profile(dev, np.array([x]))[0])
    pref = (dev.iF + iR) / (2.0 * Q * dev.D)
    return n0 / s - pref / (s * k) * (
        _cosh_over_sinh(k, x, d) - dev.B * _sinh_over_cosh(k, x, d))


# ---------------------------------------------------------------------------
# numerical inverse Laplace transform -- fixed Talbot (Abate & Valko 2004)
# ---------------------------------------------------------------------------
def talbot_inverse(F, t, M=48):
    """Invert F(s) at time t>0 by the fixed Talbot contour with M nodes.
    F must accept a complex scalar and a complex numpy array."""
    r = 2.0 * M / (5.0 * t)
    total = 0.5 * np.real(np.exp(r * t) * F(r))          # theta = 0 node
    kk = np.arange(1, M)
    th = kk * np.pi / M
    cot = 1.0 / np.tan(th)
    sigma = th + (th * cot - 1.0) * cot
    s = r * th * (cot + 1j)
    total += np.sum(np.real(np.exp(s * t) * (1.0 + 1j * sigma) * F(s)))
    return (r / M) * total


def n_laplace(dev: Device, x, t, iR, M=48):
    """n(x,t) from the Laplace solution (Talbot inversion).  x, t scalars."""
    return talbot_inverse(lambda s: N_hat(dev, x, s, iR), t, M=M)


def profile_laplace(dev: Device, xs, t, iR, M=48):
    """n(x,t) over an array of positions xs at a single time t."""
    return np.array([n_laplace(dev, float(x), t, iR, M=M) for x in xs])


def storage_times_laplace(dev: Device, iR, M=48):
    """(t0l, t0r) from n(-d,t)=0, n(+d,t)=0 via the Laplace solution."""
    from scipy.optimize import brentq

    def find(xb):
        ts = np.geomspace(1e-5, 3.0, 200) * dev.tau
        vals = np.array([n_laplace(dev, xb, t, iR, M=M) for t in ts])
        idx = np.where(vals <= 0)[0]
        if len(idx) == 0:
            return np.nan
        j = idx[0]
        if j == 0:
            return ts[0]
        return brentq(lambda t: n_laplace(dev, xb, t, iR, M=M),
                      ts[j - 1], ts[j], xtol=dev.tau * 1e-6)

    return find(-dev.d), find(+dev.d)


if __name__ == "__main__":
    # 1) validate the Talbot inverter on a known transform: 1/(s+a) -> e^{-at}
    print("=== Talbot inverter self-test:  L^-1{1/(s+a)} = e^{-at} ===")
    a = 3.0
    for t in (0.1, 0.5, 1.0, 2.0):
        approx = talbot_inverse(lambda s: 1.0 / (s + a), t)
        print(f"  t={t:.1f}:  talbot={approx:.6f}   exact={np.exp(-a*t):.6f}   "
              f"err={abs(approx-np.exp(-a*t)):.1e}")

    # 2) cross-check vs the Fourier series (Eq.36) and the FD PDE
    from benda_spenke_model import n_analytic, solve_transient
    dev = Device(d_over_L=1.0)
    iR = 10 * dev.iF
    print("\n=== three-method cross-check, iR=10iF (n/nbar) ===")
    print("  t/tau   x/d    Laplace     Eq.36     PDE")
    x, tg, N = solve_transient(dev, iR, 0.25 * dev.tau, nt=8000)
    for tf in (0.01, 0.03, 0.10, 0.20):
        for xd in (-1.0, 0.0, 1.0):
            xx = xd * dev.d
            nl = n_laplace(dev, xx, tf * dev.tau, iR) / dev.nbar
            na = n_analytic(dev, xx, tf * dev.tau, iR) / dev.nbar
            k = int(round(tf * dev.tau / (tg[1] - tg[0])))
            ip = int(round((xx + dev.d) / (x[1] - x[0])))
            npde = N[k, ip] / dev.nbar
            print(f"  {tf:4.2f}  {xd:+4.1f}   {nl:+8.4f}  {na:+8.4f}  {npde:+8.4f}")

    # 3) storage times, three methods
    from benda_spenke_model import storage_times
    print("\n=== storage times t0/tau, three methods ===")
    tlL, trL = storage_times_laplace(dev, iR)
    tlA, trA = storage_times(dev, iR, method="analytic")
    tlP, trP = storage_times(dev, iR, method="numeric", nt=12000,
                             t_end=1.5 * dev.tau)
    print(f"  t0l:  Laplace={tlL/dev.tau:.4f}  Eq.36={tlA/dev.tau:.4f}  "
          f"PDE={tlP/dev.tau:.4f}")
    print(f"  t0r:  Laplace={trL/dev.tau:.4f}  Eq.36={trA/dev.tau:.4f}  "
          f"PDE={trP/dev.tau:.4f}")
