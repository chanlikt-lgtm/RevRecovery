"""
fig4_forward_profile.py -- reproduce Fig. 4 of Benda & Spenke 1967.

Stationary high-level forward carrier concentration n(x)=p(x) across the middle
region of a p-s-n power rectifier (Eq. 21), normalised n/nbar vs x/d.  The
profile is a catenary (cosh) that is ASYMMETRIC because mu_n = 3 mu_p: the
gradient at the left (p-side, x=-d) is 3x steeper than at the right, so the
plasma piles up more on the left.  Overlaid: an independent numerical steady
state (long-time relaxation of the ambipolar PDE under forward BCs) as a check.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benda_spenke_model import Device, forward_profile, solve_transient

HERE = os.path.dirname(__file__)


def main():
    fig, ax = plt.subplots(figsize=(7, 5))
    for d_over_L, col in [(1.0, "tab:blue"), (0.5, "tab:red")]:
        dev = Device(d_over_L=d_over_L)
        x = np.linspace(-dev.d, dev.d, 401)
        n = forward_profile(dev, x) / dev.nbar
        ax.plot(x / dev.d, n, col, label=f"d/L = {d_over_L}  (Eq. 21)")

        # numeric check: relax the PDE to steady state under forward current
        xx, tt, N = solve_transient(dev, iR=None, t_end=8 * dev.tau,
                                    nt=4000, i_current=dev.iF,
                                    n0=np.full(401, dev.nbar))
        ax.plot(xx / dev.d, N[-1] / dev.nbar, col, ls="--", lw=1,
                alpha=0.7, label=f"d/L = {d_over_L}  (numeric steady state)")

    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.text(-0.98, 1.03, r"$\bar n$", color="gray")
    ax.set_xlabel("x / d")
    ax.set_ylabel(r"$n / \bar n$")
    ax.set_title("Fig. 4 -- stationary high-injection forward profile "
                 r"($\mu_n=3\mu_p$, B=0.5)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 2)
    out = os.path.join(HERE, "fig4_forward_profile.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # numeric-vs-analytic discrepancy
    dev = Device(d_over_L=1.0)
    x = np.linspace(-dev.d, dev.d, 401)
    na = forward_profile(dev, x)
    xx, tt, N = solve_transient(dev, iR=None, t_end=8 * dev.tau, nt=4000,
                                i_current=dev.iF, n0=np.full(401, dev.nbar))
    rms = np.sqrt(np.mean(((N[-1] - na) / dev.nbar) ** 2))
    print(f"numeric vs Eq.21 (d/L=1): RMS = {rms:.2e} of nbar")


if __name__ == "__main__":
    main()
