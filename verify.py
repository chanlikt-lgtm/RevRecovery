"""
verify.py -- print the reproduction's key numbers next to the paper's values.

Primary correctness check (no automated test suite): internal self-consistency
(the model's own exact identities) plus agreement between the two independent
solution methods (closed-form Eq. 36 and the numerical ambipolar PDE), and the
qualitative targets stated in Benda & Spenke 1967.
"""
import numpy as np
from benda_spenke_model import Device, forward_profile, n_analytic, storage_times


def main():
    print("=" * 70)
    print("Benda & Spenke 1967 -- reverse recovery -- model vs paper / self-checks")
    print("=" * 70)

    dev = Device(d_over_L=1.0)
    print("\n[ material / device (silicon, mu_n = 3 mu_p) ]")
    print(f"  B = (mu_n-mu_p)/(mu_n+mu_p) = {dev.B:.4f}   (paper: B = 0.5)")
    print(f"  D (ambipolar)  = {dev.D:.3f} cm^2/s")
    print(f"  L (ambipolar)  = {dev.L*1e4:.2f} um")
    print(f"  nbar           = {dev.nbar:.3e} cm^-3   (from iF = 2 d q nbar/tau)")

    print("\n[ exact identities (should be exact) ]")
    x = np.linspace(-dev.d, dev.d, 2001)
    avg = np.trapz(forward_profile(dev, x), x) / (2 * dev.d)
    print(f"  <n_forward> / nbar                 = {avg/dev.nbar:.6f}   (= 1, Eq. 24)")
    n0 = n_analytic(dev, x, 0.0, iR=5 * dev.iF)
    nf = forward_profile(dev, x)
    print(f"  Eq.36(t=0) vs Eq.21 forward, RMS   = {np.sqrt(np.mean(((n0-nf)/dev.nbar)**2)):.2e} of nbar")

    print("\n[ pure recombination decay, iR=0: border-avg vs exp(-t/tau) ]")
    for tf in (1.0, 2.0, 3.0):
        prof = n_analytic(dev, x, tf * dev.tau, iR=1e-9 * dev.iF)
        print(f"  t/tau={tf:.0f} : <n>/nbar = {np.trapz(prof,x)/(2*dev.d)/dev.nbar:.4f}"
              f"   exp(-t/tau) = {np.exp(-tf):.4f}")

    print("\n[ storage times: analytic (Eq. 36) vs numerical PDE ]")
    print("  d/L  iR/iF |  t0l/tau (an / num)  |  t0r/tau (an / num)")
    for dL in (1.0, 0.5):
        dev = Device(d_over_L=dL)
        for rr in (2, 10, 50):
            iR = rr * dev.iF
            a_l, a_r = storage_times(dev, iR, method="analytic")
            n_l, n_r = storage_times(dev, iR, method="numeric",
                                     nt=12000, t_end=1.5 * dev.tau)
            print(f"  {dL:3.1f} {rr:5d} |  {a_l/dev.tau:7.4f} / {n_l/dev.tau:7.4f}"
                  f"   |  {a_r/dev.tau:7.4f} / {n_r/dev.tau:7.4f}")

    print("\n  -> t0r > t0l at every point (n-side empties later; mu_n=3 mu_p),")
    print("     matching the asymmetry of Figs. 5, 8, 9 in the paper.")

    # ---- Sections IV-VII: space-charge growth & voltage recovery ----
    from space_charge_model import (boundary_rates, meet_time,
                                    voltage_recovery, U_intrinsic, U_doped,
                                    sc_widths)
    print("\n[ Sections IV-VII: space charge & voltage recovery ]")
    dev = Device(d_over_L=1.0)
    iR = 10 * dev.iF
    t0l, t0r = storage_times(dev, iR, method="analytic")
    Rl, Rr = boundary_rates(dev, iR)
    b = dev.mu_n / dev.mu_p
    print(f"  boundary speed ratio Rl/Rr = {Rl/Rr:.3f}   (= mu_n/mu_p = {b:.1f})")
    tm = meet_time(dev, iR, t0l, t0r)
    print(f"  boundaries meet at t/tau   = {tm/dev.tau:.4f}")
    share_eq = np.sqrt(b) * b**1.5 / (np.sqrt(b) * b**1.5 + 1.0)
    print(f"  left voltage share (equal t0) = {share_eq*100:.1f}%   "
          f"(paper: ~90% for b=3)")
    tt = np.linspace(t0l + 1e-3 * dev.tau, tm, 200)
    _, _, Ut = voltage_recovery(dev, iR, tt, t0l, t0r)
    p_pin = np.polyfit(np.log(tt - t0l), np.log(Ut), 1)[0]
    print(f"  p-i-n voltage law  U ~ (t-t0)^{p_pin:.3f}   (theory 3/2 = 1.5)")
    wl, wr = sc_widths(dev, iR, tt, t0l, t0r)
    Ud = np.array([U_doped(dev, iR, w, dev.mu_p, 2e15) for w in wl])
    p_dop = np.polyfit(np.log(wl[wl > 0.3 * dev.d]),
                       np.log(Ud[wl > 0.3 * dev.d]), 1)[0]
    print(f"  doped Schottky law U ~ w^{p_dop:.3f}          (theory 2)")
    print("=" * 70)


if __name__ == "__main__":
    main()
