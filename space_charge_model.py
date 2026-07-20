"""
space_charge_model.py -- Sections IV-VII of Benda & Spenke 1967.

The later reverse-recovery stages, after the quasi-neutral phase ends at the
storage times t0l, t0r (computed in benda_spenke_model.py):

  Section IV  -- two "sweeping-out boundaries" advance inward from x=-d and
                 x=+d, leaving space-charge zones behind.  Their widths grow
                 linearly in time (Eqs. 46-49):

                   w_l(t) = mu_n/(mu_n+mu_p) * iR/(q nbar) * (t - t0l)
                   w_r(t) = mu_p/(mu_n+mu_p) * iR/(q nbar) * (t - t0r)

                 The left zone grows faster (mu_n > mu_p) and starts earlier
                 (t0l < t0r), so the left side dominates.

  Section V   -- p-i-n rectifier voltage.  In each swept-out zone a single
                 carrier drifts in the field of its own (diffuse) space charge.
                 Poisson + the field-only current (Eq. 51,52) give a square-root
                 field  F(xi) = sqrt(2 iR xi /(eps mu))  (Eq. 54) and therefore
                 a space-charge voltage (Eq. 58,59)

                   U_sc = (2/3) sqrt(2 iR/(eps mu)) * w^{3/2}

                 so U_sc ~ (t - t0)^{3/2}.  Left uses mu_p (holes), right mu_n.

  Section VI  -- uniformly doped p-s-n rectifier.  A net doping term enters
                 Poisson (Eq. 62); when the mobile carrier drops below the
                 doping the zone turns into a Schottky (constant-charge) zone
                 with a *linear* field and U_sc ~ w^2 (Eq. quadratic).  Solved
                 here by integrating the field ODE for arbitrary doping N,
                 which recovers the 3/2 law for N->0 and the square law for
                 heavy N.

  Section VII -- diffused (exponentially graded) junction: an exponential
                 doping N(x) ~ exp(-x/Lambda); the zone is neutral where the
                 hole concentration exceeds the local doping and becomes a
                 space charge once it falls below n1 (Eq. 72,73).  Included as
                 the general doped solver with an exponential N(xi).

Units: CGS-practical (cm, s, V, A/cm^2, cm^-3, F/cm).
"""
from __future__ import annotations
import numpy as np
from benda_spenke_model import Device, storage_times, Q, VT   # noqa: F401

# NB: the space-charge field/voltage now uses the per-material permittivity
# dev.eps (F/cm), so 4H-SiC and GaN devices are handled correctly.  The module
# constant below is kept only as the silicon reference value.
EPS = 11.9 * 8.8541878128e-14        # absolute permittivity of Si (F/cm)


# ---------------------------------------------------------------------------
# Section IV -- sweeping-out boundary kinematics
# ---------------------------------------------------------------------------
def boundary_rates(dev: Device, iR):
    """(dw_l/dt, dw_r/dt) in cm/s -- Eqs. (46),(47)."""
    s = dev.mu_n + dev.mu_p
    base = iR / (Q * dev.nbar)
    return dev.mu_n / s * base, dev.mu_p / s * base


def sc_widths(dev: Device, iR, t, t0l, t0r):
    """Space-charge zone widths w_l(t), w_r(t) (cm), zero before t0l/t0r.
    Eqs. (48),(49).  Clamped at the half-width d (zone cannot exceed the base)."""
    # A zone can grow past the mid-plane (the faster left zone usually does),
    # up to the moment the two boundaries meet (w_l + w_r = 2d).  Clamp only at
    # 2d for safety; callers restrict t <= meet_time(), which enforces
    # w_l + w_r <= 2d automatically.
    Rl, Rr = boundary_rates(dev, iR)
    wl = np.clip(Rl * (np.asarray(t) - t0l), 0.0, 2 * dev.d)
    wr = np.clip(Rr * (np.asarray(t) - t0r), 0.0, 2 * dev.d)
    return wl, wr


def meet_time(dev: Device, iR, t0l, t0r):
    """Time when the two boundaries meet (w_l + w_r = 2d); the swamped zone is
    then fully consumed and this quasi-static picture ends."""
    Rl, Rr = boundary_rates(dev, iR)
    # Rl (t-t0l) + Rr (t-t0r) = 2 d
    return (2 * dev.d + Rl * t0l + Rr * t0r) / (Rl + Rr)


# ---------------------------------------------------------------------------
# Section V -- p-i-n space-charge field and voltage (closed form)
# ---------------------------------------------------------------------------
def field_intrinsic(dev: Device, iR, w, mu, npts=200):
    """Field magnitude F(xi) through a swept-out zone of width w, single carrier
    of mobility mu, intrinsic (no doping).  F = sqrt(2 iR xi/(eps mu)), Eq.(54).
    Returns (xi, F) with xi = 0 at the inner (swamped-side) boundary."""
    xi = np.linspace(0.0, w, npts)
    F = np.sqrt(2.0 * iR * xi / (dev.eps * mu))
    return xi, F


def U_intrinsic(dev: Device, iR, w, mu):
    """Space-charge voltage across an intrinsic zone of width w (Eq. 58/59):
        U = (2/3) sqrt(2 iR/(eps mu)) w^{3/2}."""
    w = np.asarray(w, float)
    return (2.0 / 3.0) * np.sqrt(2.0 * iR / (dev.eps * mu)) * np.power(np.maximum(w, 0), 1.5)


# ---------------------------------------------------------------------------
# Section VI/VII -- doped zone: numerical field integration
# ---------------------------------------------------------------------------
def field_doped(dev: Device, iR, w, mu, N, npts=600):
    """Integrate the space-charge field ODE for a swept-out zone with a net
    fixed doping that *adds* to the mobile-carrier charge (the Schottky-forming
    side, Eq. 62):

        dF/dxi = (q/eps) ( iR/(q mu F) + N(xi) )

    with F(0)=0.  N may be a scalar (uniform, Section VI) or a callable
    N(xi) (e.g. exponential, Section VII).  Returns (xi, F, U) with U the
    voltage across the zone = integral F dxi.

    N=0 reproduces the intrinsic sqrt-field / 3/2-law; large N gives a nearly
    linear field and U ~ w^2 (Schottky)."""
    xi = np.linspace(0.0, w, npts)
    dx = xi[1] - xi[0] if npts > 1 else 0.0
    F = np.zeros(npts)
    # start just off zero to avoid the 1/F singularity at the boundary
    F[0] = np.sqrt(2.0 * iR * dx / (dev.eps * mu)) * 1e-3
    Nfun = N if callable(N) else (lambda s: N)
    for k in range(npts - 1):
        Fk = max(F[k], 1e-6)
        dFdx = (Q / dev.eps) * (iR / (Q * mu * Fk) + Nfun(xi[k]))
        # midpoint (Heun) step for accuracy near the boundary
        Fmid = Fk + 0.5 * dx * dFdx
        Fmid = max(Fmid, 1e-6)
        dFdx2 = (Q / dev.eps) * (iR / (Q * mu * Fmid) + Nfun(xi[k] + 0.5 * dx))
        F[k + 1] = Fk + dx * dFdx2
    U = np.trapz(F, xi)
    return xi, F, U


def U_doped(dev: Device, iR, w, mu, N, npts=600):
    """Voltage only, from field_doped."""
    return field_doped(dev, iR, w, mu, N, npts=npts)[2]


# ---------------------------------------------------------------------------
# assembled reverse-recovery voltage transient  U_rect(t)  (p-i-n, Section V)
# ---------------------------------------------------------------------------
def voltage_recovery(dev: Device, iR, tgrid, t0l, t0r):
    """U_scl(t), U_scr(t) and U_rect = U_scl+U_scr over tgrid (p-i-n)."""
    wl, wr = sc_widths(dev, iR, tgrid, t0l, t0r)
    U_scl = U_intrinsic(dev, iR, wl, dev.mu_p)   # left = holes
    U_scr = U_intrinsic(dev, iR, wr, dev.mu_n)   # right = electrons
    return U_scl, U_scr, U_scl + U_scr


def left_voltage_share(dev: Device, iR, t0l, t0r, t):
    """Fraction of the total space-charge voltage carried by the left zone at
    time t.  The paper: ~90% for silicon (b=3) even if t0l=t0r; higher because
    t0l < t0r."""
    U_scl, U_scr, U = voltage_recovery(dev, iR, np.array([t]), t0l, t0r)
    return float(U_scl[0] / U[0])


if __name__ == "__main__":
    dev = Device(d_over_L=1.0)
    iR = 10 * dev.iF
    t0l, t0r = storage_times(dev, iR, method="analytic")
    tm = meet_time(dev, iR, t0l, t0r)
    Rl, Rr = boundary_rates(dev, iR)
    print("=== Benda-Spenke Sections IV-VII -- space charge & voltage ===")
    print(f"iR = 10 iF = {iR:.0f} A/cm^2,  nbar = {dev.nbar:.3e} cm^-3")
    print(f"t0l/tau = {t0l/dev.tau:.4f}   t0r/tau = {t0r/dev.tau:.4f}")
    print(f"left/right boundary speed = {Rl:.1f} / {Rr:.1f} cm/s  "
          f"(ratio {Rl/Rr:.2f} = mu_n/mu_p)")
    print(f"boundaries meet at t/tau = {tm/dev.tau:.4f}")

    # 90% left-share check (paper, Section V)
    # evaluate at equal elapsed time in each zone (isolate the mobility effect)
    t_eq = t0r + 0.3 * (tm - t0r)
    U_scl, U_scr, U = voltage_recovery(dev, iR, np.array([t_eq]), t0l, t0r)
    print(f"\nat t/tau={t_eq/dev.tau:.3f}:  U_scl={U_scl[0]:.1f} V  "
          f"U_scr={U_scr[0]:.1f} V  ->  left share = {U_scl[0]/U[0]*100:.1f}%")
    # pure-mobility share with equal widths (t0l=t0r hypothetical): sqrt(mu_n/mu_p)*(mu_n/mu_p)^1.5
    b = dev.mu_n / dev.mu_p
    share_eq = np.sqrt(b) * b**1.5 / (np.sqrt(b) * b**1.5 + 1.0)
    print(f"pure-mobility left share (equal t0) = {share_eq*100:.1f}%  "
          f"(paper: ~90% for b=3)")

    # 3/2 power-law check
    ts = np.linspace(t0l + 1e-3 * dev.tau, tm, 50)
    _, _, Ut = voltage_recovery(dev, iR, ts, t0l, t0r)
    slope = np.polyfit(np.log(ts - t0l), np.log(np.maximum(Ut, 1e-9)), 1)[0]
    print(f"\nU_rect ~ (t-t0)^p,  fitted p = {slope:.3f}  (theory 3/2 = 1.5)")

    # doped (Schottky) vs intrinsic voltage law
    w = 0.5 * dev.d
    Ui = U_intrinsic(dev, iR, w, dev.mu_n)
    Ud = U_doped(dev, iR, w, dev.mu_n, N=1e15)
    print(f"\nzone width w=d/2:  U_intrinsic={Ui:.1f} V   U_doped(N=1e15)={Ud:.1f} V")
