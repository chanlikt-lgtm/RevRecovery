"""
fig_scz_growth.py -- Section IV: growth of the space-charge zones.

The two sweeping-out boundaries advance inward from x=-d (left, holes) and
x=+d (right, electrons), starting at the storage times t0l < t0r.  Their
positions x = -a_l(t) and x = +a_r(t) and the space-charge widths w_l, w_r
grow linearly (Eqs. 46-49); the left moves mu_n/mu_p = 3x faster, so it fills
first.  When the boundaries meet (w_l + w_r = 2d) the swamped zone is gone.
This is the quantitative version of the paper's schematic Figs. 11-13.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benda_spenke_model import Device, storage_times
from space_charge_model import sc_widths, meet_time, boundary_rates

HERE = os.path.dirname(__file__)


def main():
    dev = Device(d_over_L=1.0)
    iR = 10 * dev.iF
    t0l, t0r = storage_times(dev, iR, method="analytic")
    tm = meet_time(dev, iR, t0l, t0r)
    t = np.linspace(0, tm, 400)
    wl, wr = sc_widths(dev, iR, t, t0l, t0r)
    al = dev.d - wl          # left boundary at x = -a_l
    ar = dev.d - wr          # right boundary at x = +a_r
    um = 1e4                 # cm -> um

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) boundary trajectories in the (x, t) plane
    ax1.plot(-al * um, t / dev.tau, "r", lw=2, label=r"left boundary $-a_l(t)$")
    ax1.plot(ar * um, t / dev.tau, "b", lw=2, label=r"right boundary $+a_r(t)$")
    ax1.axvline(-dev.d * um, color="k", ls=":", lw=1)
    ax1.axvline(dev.d * um, color="k", ls=":", lw=1)
    ax1.fill_betweenx(t / dev.tau, -dev.d * um, -al * um, color="r", alpha=0.12)
    ax1.fill_betweenx(t / dev.tau, ar * um, dev.d * um, color="b", alpha=0.12)
    ax1.axhline(t0l / dev.tau, color="r", ls="--", lw=0.8)
    ax1.axhline(t0r / dev.tau, color="b", ls="--", lw=0.8)
    ax1.text(-dev.d * um, t0l / dev.tau, " t0l", color="r", fontsize=8, va="bottom")
    ax1.text(-dev.d * um, t0r / dev.tau, " t0r", color="b", fontsize=8, va="bottom")
    ax1.set_xlabel("x  [um]"); ax1.set_ylabel(r"t / $\tau$")
    ax1.set_title("(a) sweeping-out boundaries (shaded = space charge)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # (b) space-charge widths vs time
    ax2.plot((t - t0l) / dev.tau, wl * um, "r", lw=2, label=r"$w_l$ (left, holes)")
    ax2.plot((t - t0r) / dev.tau, wr * um, "b", lw=2, label=r"$w_r$ (right, electrons)")
    Rl, Rr = boundary_rates(dev, iR)
    ax2.set_xlabel(r"$(t-t_0)/\tau$"); ax2.set_ylabel("space-charge width  [um]")
    ax2.set_title(f"(b) widths grow linearly, slope ratio "
                  f"{Rl/Rr:.1f} = μn/μp")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle("Section IV -- space-charge-zone growth "
                 r"($i_R=10\,i_F$, boundaries meet at "
                 f"t/τ={tm/dev.tau:.3f})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "fig_scz_growth.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
