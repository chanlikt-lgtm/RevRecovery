"""
fig7_recombination_decay.py -- reproduce Fig. 7 of Benda & Spenke 1967.

iR = 0 : no sweeping-out current, so the stored plasma decays purely by
RECOMBINATION.  Starting from the forward profile (Eq. 21), the concentration
collapses toward zero with essentially the lifetime tau as time constant
(the fundamental mode ~ exp(-t/tau)); higher spatial modes decay faster, so
the profile also flattens.  This is the case Davies solved for equal
mobilities; here mu_n = 3 mu_p (B = 0.5) as everywhere in the paper.

Curves at t/tau = 0, 0.01, 0.03, 0.10, 0.30, 1.00, 2.00, 3.00 (as in Fig. 7).
Solid = numerical ambipolar PDE; dots = closed-form Eq. (36) with iR->0.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benda_spenke_model import Device, solve_transient, n_analytic

HERE = os.path.dirname(__file__)
TIMES = [0.0, 0.01, 0.03, 0.10, 0.30, 1.00, 2.00, 3.00]


def main():
    dev = Device(d_over_L=1.0)
    # iR = 0 -> zero-flux Neumann BCs (pure recombination decay)
    x, t, N = solve_transient(dev, iR=0.0, t_end=3.0 * dev.tau, nt=6000,
                              i_current=0.0)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for tf in TIMES:
        k = int(round(tf * dev.tau / (t[1] - t[0])))
        k = min(k, len(t) - 1)
        ax.plot(x / dev.d, N[k] / dev.nbar, lw=1.4)
        # analytic overlay (Eq.36, iR->small to avoid 0/0; use tiny iR)
        na = n_analytic(dev, x, tf * dev.tau, iR=1e-9 * dev.iF) / dev.nbar
        ax.plot(x / dev.d, na, "k.", ms=2, alpha=0.35)
        ax.text(1.01, N[k][-1] / dev.nbar, f"{tf:.2f}", fontsize=7, va="center")

    ax.set_xlabel("x / d")
    ax.set_ylabel(r"$n / \bar n$")
    ax.set_title(r"Fig. 7 -- decay by recombination, $i_R=0$  "
                 r"($t/\tau$ labels; dots = Eq. 36)")
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, 1.1)
    ax.set_ylim(0, 1.8)
    out = os.path.join(HERE, "fig7_recombination_decay.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
