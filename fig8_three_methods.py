"""
fig8_three_methods.py -- Fig. 8 sweep-out profiles solved THREE independent ways.

Overlays, at the same conditions (d/L=1, iR=10 iF, t/tau = 0 ... 0.2):
  * finite-difference ambipolar PDE      (solve_transient)   -- solid lines
  * Fourier eigenfunction series Eq.36   (n_analytic)        -- dots
  * Laplace transform + Talbot inversion (n_laplace)         -- open circles

The three share no code (grid solver / sine-series / operational-calculus).
Their agreement (right panel: max |Laplace - Eq.36| and RMS |PDE - Eq.36| vs
time) is the correctness check, in the spirit of the SOI Fig.4 three-method
validation.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benda_spenke_model import Device, n_analytic, solve_transient
from laplace_solver import profile_laplace, n_laplace

HERE = os.path.dirname(__file__)
TIMES = [0.0, 0.01, 0.03, 0.10, 0.20]


def main():
    dev = Device(d_over_L=1.0)
    iR = 10 * dev.iF
    x = np.linspace(-dev.d, dev.d, 201)
    xg, tg, N = solve_transient(dev, iR, 0.22 * dev.tau, nt=9000)
    xs_sparse = np.linspace(-dev.d, dev.d, 21)     # sparse markers for Laplace

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    for tf in TIMES:
        t = tf * dev.tau
        na = n_analytic(dev, x, t if tf > 0 else 1e-9 * dev.tau, iR) / dev.nbar
        line, = ax1.plot(x / dev.d, na, lw=1.3)          # Eq.36 as the base line
        c = line.get_color()
        # PDE (solid faded, sampled) -- overlay
        k = int(round(tf * dev.tau / (tg[1] - tg[0])))
        k = min(k, len(tg) - 1)
        ax1.plot(xg[::8] / dev.d, N[k, ::8] / dev.nbar, "-", color=c, lw=3,
                 alpha=0.25)
        # Laplace (open circles at sparse x)
        tl = t if tf > 0 else 1e-4 * dev.tau
        nl = profile_laplace(dev, xs_sparse, tl, iR) / dev.nbar
        ax1.plot(xs_sparse / dev.d, nl, "o", mfc="none", mec=c, ms=6)
        ax1.text(1.01, na[-1], f"{tf:.2f}", fontsize=7, color=c, va="center")

    ax1.axhline(0, color="gray", lw=0.6)
    ax1.set_xlabel("x / d"); ax1.set_ylabel(r"$n/\bar n$")
    ax1.set_title("Fig. 8 three ways: line=Eq.36, thick faded=PDE, ∘=Laplace")
    ax1.set_xlim(-1, 1.08); ax1.set_ylim(-2.0, 1.9); ax1.grid(alpha=0.3)

    # ---- agreement vs time ----
    tfs = np.linspace(0.002, 0.20, 20)
    dLA, dPA = [], []
    for tf in tfs:
        t = tf * dev.tau
        na = n_analytic(dev, x, t, iR)
        nl = np.array([n_laplace(dev, float(xi), t, iR) for xi in x])
        k = int(round(t / (tg[1] - tg[0]))); k = min(k, len(tg) - 1)
        npde = np.interp(x, xg, N[k])
        dLA.append(np.max(np.abs(nl - na)) / dev.nbar)
        dPA.append(np.sqrt(np.mean(((npde - na) / dev.nbar) ** 2)))
    ax2.semilogy(tfs, dLA, "o-", label="max |Laplace − Eq.36|")
    ax2.semilogy(tfs, dPA, "s-", label="RMS |PDE − Eq.36|")
    ax2.set_xlabel(r"t / $\tau$"); ax2.set_ylabel(r"discrepancy / $\bar n$")
    ax2.set_title("(b) three-method agreement vs time")
    ax2.grid(alpha=0.3, which="both"); ax2.legend(fontsize=9)

    fig.suptitle("Reverse-recovery sweep-out solved 3 independent ways "
                 "(PDE / Eq.36 series / Laplace-Talbot)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "fig8_three_methods.png")
    fig.savefig(out, dpi=130); print("wrote", out)
    print(f"  max |Laplace-Eq.36|/nbar over t = {max(dLA):.2e}")
    print(f"  max RMS |PDE-Eq.36|/nbar over t = {max(dPA):.2e}")


if __name__ == "__main__":
    main()
