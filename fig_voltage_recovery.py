"""
fig_voltage_recovery.py -- Section V (and VI): the reverse-recovery VOLTAGE.

Once the space-charge zones start to grow (after t0l, t0r), the reverse voltage
appears across them.  For the p-i-n rectifier each zone gives
U_sc = (2/3) sqrt(2 iR/(eps mu)) w^{3/2} (Eq. 58,59), and since w ~ (t - t0),
the total rectifier voltage rises as (t - t0)^{3/2} -- the paper's central
voltage-recovery result, confirmed experimentally in its Section VIII.

Panel (a): U_scl(t), U_scr(t), U_rect(t) -- the voltage transient; the left
(hole) zone carries ~90%+ of it.
Panel (b): log-log U_rect vs (t-t0l) showing the 3/2 slope for the p-i-n case,
and slope 2 for a uniformly doped (Schottky) device where U ~ w^2 ~ (t-t0)^2.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benda_spenke_model import Device, storage_times
from space_charge_model import (voltage_recovery, sc_widths, meet_time,
                                U_doped)

HERE = os.path.dirname(__file__)


def main():
    dev = Device(d_over_L=1.0)
    iR = 10 * dev.iF
    t0l, t0r = storage_times(dev, iR, method="analytic")
    tm = meet_time(dev, iR, t0l, t0r)
    t = np.linspace(0, tm, 500)
    U_scl, U_scr, U_rect = voltage_recovery(dev, iR, t, t0l, t0r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) voltage transient
    ax1.plot(t / dev.tau, U_scl, "r", lw=2, label=r"$U_{scl}$ (left, holes)")
    ax1.plot(t / dev.tau, U_scr, "b", lw=2, label=r"$U_{scr}$ (right, electrons)")
    ax1.plot(t / dev.tau, U_rect, "k", lw=2.4, label=r"$U_{rect}=U_{scl}+U_{scr}$")
    ax1.axvline(t0l / dev.tau, color="r", ls="--", lw=0.8)
    ax1.axvline(t0r / dev.tau, color="b", ls="--", lw=0.8)
    share = U_scl[-1] / U_rect[-1] * 100
    ax1.set_xlabel(r"t / $\tau$"); ax1.set_ylabel("voltage  [V]")
    ax1.set_title(f"(a) reverse-recovery voltage (left share {share:.0f}%)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # (b) power laws: p-i-n (3/2) vs doped Schottky (2)
    tt = np.linspace(t0l + 1e-3 * dev.tau, tm, 200)
    _, _, Ut = voltage_recovery(dev, iR, tt, t0l, t0r)
    # doped device: same kinematics, doped voltage law on each side
    wl, wr = sc_widths(dev, iR, tt, t0l, t0r)
    N = 2e15
    Ud = np.array([U_doped(dev, iR, wl_i, dev.mu_p, N) for wl_i in wl]) \
        + np.array([U_doped(dev, iR, wr_i, dev.mu_n, N) for wr_i in wr])
    x = (tt - t0l) / dev.tau
    ax2.loglog(x, Ut, "k", lw=2, label="p-i-n  U_rect")
    ax2.loglog(x, Ud, "tab:red", lw=2, label="doped (N=2e15) U_rect")
    # slope guides
    xg = np.array([x[5], x[-1]])
    ax2.loglog(xg, Ut[5] * (xg / xg[0]) ** 1.5, "k--", lw=1, alpha=0.6,
               label="slope 3/2")
    ax2.loglog(xg, Ud[5] * (xg / xg[0]) ** 2.0, "r--", lw=1, alpha=0.6,
               label="slope 2")
    ax2.set_xlabel(r"$(t-t_{0l})/\tau$"); ax2.set_ylabel("U_rect  [V]")
    ax2.set_title("(b) voltage power law: p-i-n ∝ t^{3/2}, Schottky ∝ t²")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")

    fig.suptitle("Section V/VI -- reverse-recovery voltage transient "
                 r"($i_R=10\,i_F$)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "fig_voltage_recovery.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # fit exponents
    pl_pin = np.polyfit(np.log(x), np.log(Ut), 1)[0]
    pl_dop = np.polyfit(np.log(x[x > 0.02]), np.log(Ud[x > 0.02]), 1)[0]
    print(f"fitted exponent: p-i-n = {pl_pin:.3f} (theory 1.5), "
          f"doped = {pl_dop:.3f} (theory ~2)")
    print(f"peak U_rect (at boundary meet) = {U_rect[-1]:.0f} V")


if __name__ == "__main__":
    main()
