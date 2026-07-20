"""
fig_materials_compare.py -- reverse recovery in Si vs 4H-SiC vs GaN.

Extension of the Benda & Spenke (silicon) model to wide-bandgap rectifiers,
using the cited parameter sets in materials.py.  Four panels tell the story of
going Si -> 4H-SiC -> GaN:

(a) FORWARD PROFILE.  The mobility asymmetry B = (mu_n-mu_p)/(mu_n+mu_p) rises
    0.50 -> 0.78 -> 0.94, so the high-injection plasma piles up ever more
    strongly on the p-side (left).  GaN's very low hole mobility makes it the
    most lopsided.

(b) STORAGE-TIME ASYMMETRY.  The two borders empty at different times t0l < t0r;
    the ratio t0r/t0l grows with B -- the n-side lags further and further behind
    the p-side as the material gets more asymmetric.

(c) VOLTAGE SPLIT.  During recovery the left (hole) space-charge zone carries a
    fraction b^2/(b^2+1) of the voltage (b = mu_n/mu_p): 90% (Si) -> 98% (SiC)
    -> >99.9% (GaN).  The recovery voltage becomes essentially a single-sided
    event in the wide-bandgap devices.

(d) DYNAMIC-AVALANCHE MARGIN.  The peak voltage a recovery space-charge zone can
    hold before its field reaches the critical field Ec is
    U_max = (1/3) eps mu Ec^3 / iR -- the Baliga figure of merit (eps mu Ec^3)
    emerging from the reverse-recovery physics.  SiC and GaN hold ~70-130x more
    than Si before dynamic avalanche.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benda_spenke_model import forward_profile, storage_times, Q
from materials import MATERIALS, SI

HERE = os.path.dirname(__file__)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ---- (a) forward profiles ----
    ax = axes[0, 0]
    for m in MATERIALS:
        dev = m.device(d_over_L=1.0)
        x = np.linspace(-dev.d, dev.d, 401)
        ax.plot(x / dev.d, forward_profile(dev, x) / dev.nbar, color=m.color,
                lw=2, label=f"{m.name}  (B={m.B:.2f})")
    ax.axhline(1, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("x / d   (p-side  <--  -->  n-side)")
    ax.set_ylabel(r"$n/\bar n$")
    ax.set_title("(a) forward plasma profile: asymmetry grows with B")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(-1, 1)

    # ---- (b) storage-time asymmetry t0r/t0l vs iR/iF ----
    ax = axes[0, 1]
    ratios = np.geomspace(2, 100, 10)
    for m in MATERIALS:
        dev = m.device(d_over_L=1.0)
        rr = []
        for r in ratios:
            a, b = storage_times(dev, r * dev.iF, method="analytic")
            rr.append(b / a)
        ax.semilogx(ratios, rr, color=m.color, lw=2, marker="o", ms=4,
                    label=f"{m.name}  (b={m.mu_n/m.mu_p:.0f})")
    ax.set_xlabel(r"$i_R / i_F$")
    ax.set_ylabel(r"$t_{0r} / t_{0l}$   (right lags left)")
    ax.set_title("(b) storage-time asymmetry grows with B")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # ---- (c) left voltage share vs mobility ratio b ----
    ax = axes[1, 0]
    b = np.geomspace(1, 60, 200)
    ax.semilogx(b, 100 * b**2 / (b**2 + 1), "k", lw=2,
                label=r"$b^2/(b^2+1)$")
    for m in MATERIALS:
        bb = m.mu_n / m.mu_p
        sh = 100 * bb**2 / (bb**2 + 1)
        ax.plot(bb, sh, "o", color=m.color, ms=11)
        ax.annotate(f"{m.name}\n{sh:.1f}%", (bb, sh),
                    textcoords="offset points", xytext=(-4, -34),
                    fontsize=8, color=m.color, ha="center")
    ax.set_xlabel(r"mobility ratio  $b=\mu_n/\mu_p$")
    ax.set_ylabel("left (hole) zone voltage share  [%]")
    ax.set_title("(c) recovery voltage becomes single-sided")
    ax.set_ylim(85, 101); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)

    # ---- (d) max blocking voltage before dynamic avalanche ----
    ax = axes[1, 1]
    iR = np.geomspace(50, 5000, 200)          # A/cm^2
    for m in MATERIALS:
        dev = m.device(d_over_L=1.0)
        # U_max = (1/3) eps mu_p Ec^3 / iR  (left/hole zone reaches Ec first)
        Umax = (1.0 / 3.0) * dev.eps * m.mu_p * m.Ec**3 / iR
        ax.loglog(iR, Umax, color=m.color, lw=2, label=m.name)
    ax.set_xlabel(r"reverse current density  $i_R$  [A/cm$^2$]")
    ax.set_ylabel("max recovery voltage before avalanche  [V]")
    ax.set_title(r"(d) dynamic-avalanche margin $\propto \epsilon\,\mu\,E_c^3/i_R$")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)

    fig.suptitle("Reverse recovery of P-i-N rectifiers: Si vs 4H-SiC vs GaN "
                 "(Benda-Spenke model, cited material data)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, "fig_materials_compare.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # ---- printed FOM table ----
    print("\nmaterial   B     b     left-share   eps*mu_p*Ec^3 (rel. Si)")
    fom_si = SI.eps_r * SI.mu_p * SI.Ec**3
    for m in MATERIALS:
        bb = m.mu_n / m.mu_p
        sh = 100 * bb**2 / (bb**2 + 1)
        fom = m.eps_r * m.mu_p * m.Ec**3 / fom_si
        print(f"  {m.name:<7}{m.B:.2f}  {bb:5.1f}   {sh:6.2f}%      {fom:8.1f}x")


if __name__ == "__main__":
    main()
