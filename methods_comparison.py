"""
methods_comparison.py -- quantitative comparison of the three reverse-recovery
solvers: finite-difference PDE, Fourier series (Eq. 36), and Laplace-Talbot.

Measures, at a representative operating point (d/L=1, iR=10 iF, t/tau=0.03),
for each method:
  * accuracy at its default resolution (vs an essentially-exact reference),
  * the resolution parameter needed to reach 1e-4 and 1e-6 RMS error,
  * the empirical convergence rate,
  * wall-clock cost for one full x-profile.

Reference "exact": Eq. 36 with 4000 terms (fully converged for t>0, where the
series' exp(-[1+(v pi L/d)^2] t/tau) factor kills high harmonics).  It is used
only as the yardstick; the PDE and Laplace routes are independent of it.

Writes fig_methods_convergence.png and prints the comparison table.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benda_spenke_model import Device, n_analytic, solve_transient
from laplace_solver import n_laplace

HERE = os.path.dirname(__file__)


def main():
    dev = Device(d_over_L=1.0)
    iR = 10 * dev.iF
    tf = 0.03
    t = tf * dev.tau
    x = np.linspace(-dev.d, dev.d, 81)
    ref = n_analytic(dev, x, t, iR, nmax=4000) / dev.nbar

    def rms(a):
        return float(np.sqrt(np.mean((a - ref) ** 2)))

    # ---------------- Fourier series (Eq. 36): vary number of terms ----------
    fourier_n = [2, 3, 5, 8, 12, 20, 40, 80, 160]
    fourier_err = []
    for nmax in fourier_n:
        a = n_analytic(dev, x, t, iR, nmax=nmax) / dev.nbar
        fourier_err.append(rms(a))

    # ---------------- Laplace-Talbot: vary number of contour nodes M ---------
    # (fixed Talbot converges spectrally to an optimal M ~ 16, beyond which
    #  round-off in the contour sum degrades it -- captured by the last points)
    talbot_M = [4, 6, 8, 10, 12, 14, 16, 20, 32, 48]
    talbot_err = []
    for M in talbot_M:
        a = np.array([n_laplace(dev, float(xi), t, iR, M=M) for xi in x]) / dev.nbar
        talbot_err.append(rms(a))

    # ---------------- PDE (backward Euler): vary # time steps ----------------
    # fixed fine grid nx=801; march from 0 to t with nt steps (t_end = t).
    pde_nt = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    pde_err = []
    for nt in pde_nt:
        xg, tg, N = solve_transient(dev, iR, t, nx=801, nt=nt)
        a = np.interp(x, xg, N[-1]) / dev.nbar
        pde_err.append(rms(a))

    # ---------------- empirical convergence rates ---------------------------
    def rate(params, errs, loglog=True):
        # fit only the descending branch: from the start up to argmin(error),
        # so neither the machine-precision floor nor any round-off up-tick
        # (fixed Talbot beyond its optimal M) flattens the measured rate.
        p = np.array(params, float)
        e = np.array(errs, float)
        jmin = int(np.argmin(e)) + 1
        p, e = p[:jmin], e[:jmin]
        m = e > 1e-13
        if m.sum() < 2:
            m = np.ones_like(e, bool)
        if loglog:
            return np.polyfit(np.log(p[m]), np.log(e[m]), 1)[0]
        return np.polyfit(p[m], np.log(e[m]), 1)[0]   # semilog slope

    r_pde = rate(pde_nt, pde_err, loglog=True)          # ~ -1 (O(dt))
    r_four = rate(fourier_n, fourier_err, loglog=False)  # semilog (exponential)
    r_talb = rate(talbot_M, talbot_err, loglog=False)    # semilog (spectral)

    def to_reach(params, errs, tol):
        for p, e in zip(params, errs):
            if e <= tol:
                return p
        return None

    # ---------------- timing one full profile at default resolution ---------
    def time_call(fn, n=3):
        best = 1e9
        for _ in range(n):
            t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
        return best

    t_four = time_call(lambda: n_analytic(dev, x, t, iR, nmax=200))
    t_talb = time_call(lambda: [n_laplace(dev, float(xi), t, iR, M=48) for xi in x])
    t_pde = time_call(lambda: solve_transient(dev, iR, t, nx=401, nt=200))

    # accuracy at the resolution actually used in the figures
    acc_four = rms(n_analytic(dev, x, t, iR, nmax=200) / dev.nbar)
    acc_talb = rms(np.array([n_laplace(dev, float(xi), t, iR, M=48)
                             for xi in x]) / dev.nbar)
    xg, tg, N = solve_transient(dev, iR, t, nx=401, nt=8000)
    acc_pde = rms(np.interp(x, xg, N[-1]) / dev.nbar)

    # ---------------- table --------------------------------------------------
    print("=" * 92)
    print(f"THREE-METHOD COMPARISON  (d/L=1, iR=10 iF, t/tau={tf}, "
          f"RMS error over x vs Eq.36[4000])")
    print("=" * 92)
    hdr = (f"{'method':<22}{'type':<16}{'res. param':<12}"
           f"{'->1e-4':>8}{'->1e-6':>8}{'convergence':>16}")
    print(hdr); print("-" * 92)
    print(f"{'Fourier series Eq.36':<22}{'analytic sum':<16}{'N terms':<12}"
          f"{str(to_reach(fourier_n, fourier_err, 1e-4)):>8}"
          f"{str(to_reach(fourier_n, fourier_err, 1e-6)):>8}"
          f"{'exp: e^' + format(r_four, '.2f') + 'N':>16}")
    print(f"{'Laplace-Talbot':<22}{'analytic+invert':<16}{'M nodes':<12}"
          f"{str(to_reach(talbot_M, talbot_err, 1e-4)):>8}"
          f"{str(to_reach(talbot_M, talbot_err, 1e-6)):>8}"
          f"{'spectral: e^' + format(r_talb, '.2f') + 'M':>16}")
    print(f"{'finite-diff PDE':<22}{'time-march':<16}{'nt steps':<12}"
          f"{str(to_reach(pde_nt, pde_err, 1e-4)):>8}"
          f"{str(to_reach(pde_nt, pde_err, 1e-6)):>8}"
          f"{'O(dt): n^' + format(r_pde, '.2f'):>16}")
    print("-" * 92)
    def fmt(e):
        return "<1e-15" if e < 1e-15 else f"{e:.2e}"
    print(f"{'DEFAULT (as in figs)':<22}{'accuracy':<16}{'':<12}")
    print(f"  Fourier  nmax=200 : RMS={fmt(acc_four):>8}   "
          f"time/profile={t_four*1e3:7.2f} ms  (81 pts)")
    print(f"  Talbot   M=48     : RMS={fmt(acc_talb):>8}   "
          f"time/profile={t_talb*1e3:7.2f} ms  (81 pts)")
    print(f"  PDE   401x8000    : RMS={fmt(acc_pde):>8}   "
          f"time/profile={t_pde*1e3:7.2f} ms  (whole field)")
    print("=" * 92)
    print("notes:")
    print(" * analytic methods (Fourier, Laplace) give n(x,t) at ANY single")
    print("   point directly and reach machine precision; their 'iteration'")
    print("   count is the series/quadrature length, not a time march.")
    print(" * the PDE marches every step 0->t and returns the whole space-time")
    print("   field at once; its error floors at the spatial O(dx^2) term")
    print(f"   (~{acc_pde:.1e} at nx=401), so more nt alone cannot reach 1e-6.")
    print(" * Fourier converges Gaussian-fast (exp(-c N^2)) for t>0 because of")
    print("   the exp(-[1+(v pi L/d)^2] t/tau) factor; at t=0 it is only ~1/N^2.")

    # ---------------- convergence figure ------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    ax1.semilogy(fourier_n, fourier_err, "o-", label="Fourier Eq.36 (N terms)")
    ax1.semilogy(talbot_M, talbot_err, "s-", label="Laplace-Talbot (M nodes)")
    ax1.axhline(1e-4, color="gray", ls=":", lw=0.8)
    ax1.axhline(1e-6, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel("resolution parameter (N terms / M nodes)")
    ax1.set_ylabel(r"RMS error / $\bar n$")
    ax1.set_title("(a) analytic methods: exponential/spectral convergence")
    ax1.grid(alpha=0.3, which="both"); ax1.legend(fontsize=9)

    ax2.loglog(pde_nt, pde_err, "^-", color="tab:red", label="PDE (nt steps)")
    ax2.loglog(pde_nt, pde_err[0] * (np.array(pde_nt) / pde_nt[0]) ** -1.0,
               "k--", lw=1, alpha=0.6, label="slope -1 (O(Δt))")
    ax2.set_xlabel("time steps  nt")
    ax2.set_ylabel(r"RMS error / $\bar n$")
    ax2.set_title(f"(b) PDE: 1st-order in Δt (fit n^{r_pde:.2f})")
    ax2.grid(alpha=0.3, which="both"); ax2.legend(fontsize=9)

    fig.suptitle("Convergence of the three reverse-recovery solvers "
                 f"(t/τ={tf})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "fig_methods_convergence.png")
    fig.savefig(out, dpi=130); print("\nwrote", out)


if __name__ == "__main__":
    main()
