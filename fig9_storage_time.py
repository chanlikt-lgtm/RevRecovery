"""
fig9_storage_time.py -- reproduce Fig. 9 of Benda & Spenke 1967.

Storage (delay) times t0l, t0r -- the moments the left/right border
concentrations reach zero and space-charge zones begin to form (Eq. 37, 38) --
as functions of the current ratio iR/iF, for a p-i-n rectifier (negligible
doping), for d/L = 1 (a) and d/L = 1/2 (b), plotted log-log exactly as the
paper.  t0r > t0l throughout (the right/n-side empties later) because
mu_n = 3 mu_p.  At large iR/iF the sweep-out dominates and t0 ~ (iR/iF)^-1.

Solid = closed-form Eq. (36) roots; open markers = numerical PDE zero-crossing.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benda_spenke_model import Device, storage_times

HERE = os.path.dirname(__file__)
RATIOS = np.geomspace(1.0, 200.0, 22)


def curve(dev, ratios, method="analytic", **kw):
    tl, tr = [], []
    for rr in ratios:
        a, b = storage_times(dev, rr * dev.iF, method=method, **kw)
        tl.append(a / dev.tau)
        tr.append(b / dev.tau)
    return np.array(tl), np.array(tr)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, dL, lab in [(axes[0], 1.0, "(a) d = L"),
                        (axes[1], 0.5, "(b) d = L/2")]:
        dev = Device(d_over_L=dL)
        tl, tr = curve(dev, RATIOS, method="analytic")
        ax.loglog(RATIOS, tl, "r-", label=r"$t_{0l}/\tau$ (left, empties first)")
        ax.loglog(RATIOS, tr, "b-", label=r"$t_{0r}/\tau$ (right)")

        # numeric spot checks at a few ratios
        rr_chk = np.array([2.0, 10.0, 50.0])
        tlN, trN = curve(dev, rr_chk, method="numeric", nt=12000,
                         t_end=1.5 * dev.tau)
        ax.loglog(rr_chk, tlN, "ro", mfc="none", ms=7)
        ax.loglog(rr_chk, trN, "bs", mfc="none", ms=7)

        ax.set_xlabel(r"$i_R / i_F$")
        ax.set_title(lab)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlim(1, 200)
        ax.set_ylim(1e-3, 1)
        ax.legend(fontsize=9)
    axes[0].set_ylabel(r"$t_{0l}/\tau,\ t_{0r}/\tau$")
    fig.suptitle("Fig. 9 -- storage (delay) times vs current ratio "
                 "(open markers = numerical PDE)")
    fig.tight_layout()
    out = os.path.join(HERE, "fig9_storage_time.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # printed table
    print("\n d/L   iR/iF    t0l/tau   t0r/tau")
    for dL in (1.0, 0.5):
        dev = Device(d_over_L=dL)
        for rr in (2, 5, 10, 50, 100):
            a, b = storage_times(dev, rr * dev.iF, method="analytic")
            print(f" {dL:3.1f}   {rr:5d}   {a/dev.tau:8.4f}  {b/dev.tau:8.4f}")


if __name__ == "__main__":
    main()
