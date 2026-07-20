"""
fig8_sweepout.py -- reproduce Fig. 8 of Benda & Spenke 1967.

iR = 10 iF : a strong reverse (sweeping-out) current.  Now recombination is
negligible next to the fast withdrawal of carriers.  The border gradients flip
sign at t=0 (holes pulled left, electrons pulled right), so the concentration
is scooped out from BOTH edges.  The left border (p-side) empties first because
mu_n = 3 mu_p makes its gradient steeper: n(-d,t) reaches 0 at t = t0l before
n(+d,t) reaches 0 at t = t0r.  Where the quasi-neutral solution would go
negative, a space-charge zone forms -- that instant is the storage time.

Curves at t/tau = 0, 0.01, 0.03, 0.10, 0.20 (as in Fig. 8).
Solid = numerical PDE; dots = closed-form Eq. (36).  Markers = storage points.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benda_spenke_model import Device, solve_transient, n_analytic, storage_times

HERE = os.path.dirname(__file__)
TIMES = [0.0, 0.01, 0.03, 0.10, 0.20]


def main():
    dev = Device(d_over_L=1.0)
    iR = 10.0 * dev.iF
    x, t, N = solve_transient(dev, iR=iR, t_end=0.25 * dev.tau, nt=8000)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for tf in TIMES:
        k = int(round(tf * dev.tau / (t[1] - t[0])))
        k = min(k, len(t) - 1)
        prof = N[k] / dev.nbar
        ax.plot(x / dev.d, prof, lw=1.4, label=f"t/τ = {tf:.2f}")
        na = n_analytic(dev, x, tf * dev.tau, iR=iR) / dev.nbar
        ax.plot(x / dev.d, na, "k.", ms=2, alpha=0.35)

    # storage times
    t0l, t0r = storage_times(dev, iR, method="analytic")
    ax.axhline(0, color="gray", lw=0.6)
    ax.plot([-1], [0], "rv", ms=9, label=f"t0l/τ={t0l/dev.tau:.3f}  (left empties first)")
    ax.plot([+1], [0], "b^", ms=9, label=f"t0r/τ={t0r/dev.tau:.3f}")

    ax.set_xlabel("x / d")
    ax.set_ylabel(r"$n / \bar n$")
    ax.set_title(r"Fig. 8 -- sweeping-out, $i_R=10\,i_F$  (dots = Eq. 36)")
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.3, 1.8)
    ax.legend(fontsize=8, loc="upper center")
    out = os.path.join(HERE, "fig8_sweepout.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)
    print(f"storage: t0l/tau={t0l/dev.tau:.4f}  t0r/tau={t0r/dev.tau:.4f}")


if __name__ == "__main__":
    main()
