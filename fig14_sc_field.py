"""
fig14_sc_field.py -- Sections V-VI: field and carrier distributions in the
space-charge zones.

Left panel reproduces the paper's Fig. 14 (p-i-n): in each swept-out zone the
single remaining carrier drifts in the field of its own diffuse space charge,
giving a square-root field  F(xi) = sqrt(2 iR xi/(eps mu))  (Eq. 54) and a
carrier concentration  p(xi) = iR/(q mu F) ~ xi^{-1/2}  that is highest at the
inner boundary and falls outward (Eq. 55).  Plotted against the normalized
distance from the inner boundary, both sides look identical (the paper's
abscissa reduction), so one normalized curve serves both.

Right panel adds Section VI: with a finite doping N the field ODE
dF/dxi = (q/eps)(iR/(q mu F) + N) makes the field become LINEAR once the mobile
carrier drops below N -- a Schottky (constant space charge) zone.  Shown for
N = 0 (intrinsic), and two doping levels, illustrating the crossover.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benda_spenke_model import Device, Q
from space_charge_model import field_intrinsic, field_doped, EPS

HERE = os.path.dirname(__file__)


def main():
    dev = Device(d_over_L=1.0)
    iR = 10 * dev.iF
    w = 0.6 * dev.d
    um = 1e4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- (a) intrinsic p-i-n: field and concentration (Fig. 14) ----
    xi, F = field_intrinsic(dev, iR, w, dev.mu_p, npts=300)
    p = iR / (Q * dev.mu_p * np.maximum(F, 1e-9))
    xin = xi / w
    ax1b = ax1.twinx()
    l1, = ax1.plot(xin, F / 1e5, "k", lw=2, label="field F(ξ)")
    # space-charge carrier density is ~0.01-1% of nbar (the paper's "~1%"),
    # highest at the inner boundary and falling as xi^{-1/2} -> plot in percent.
    l2, = ax1b.plot(xin[1:], 100 * p[1:] / dev.nbar, "g", lw=2,
                    label=r"concentration p(ξ)  [% of $\bar n$]")
    ax1.set_xlabel("normalized distance from inner boundary  ξ/w")
    ax1.set_ylabel("field  F  [10⁵ V/cm]")
    ax1b.set_ylabel(r"$p/\bar n$  [%]", color="g")
    ax1b.set_ylim(0, 1.5)
    ax1b.axhline(1.0, color="g", ls=":", lw=0.8)
    ax1b.text(0.55, 1.02, "paper: ~1% of forward conc.", color="g", fontsize=7)
    ax1.set_title("(a) p-i-n zone: √-field & falling carrier (Fig. 14)")
    ax1.legend(handles=[l1, l2], fontsize=8, loc="center right")
    ax1.grid(alpha=0.3)

    # ---- (b) doping crossover: field shape (Section VI) ----
    for N, lab, c in [(0.0, "N=0 (intrinsic, √)", "k"),
                      (3e14, "N=3e14", "tab:orange"),
                      (2e15, "N=2e15 (Schottky, linear)", "tab:red")]:
        xi, F, U = field_doped(dev, iR, w, dev.mu_n, N=N, npts=500)
        ax2.plot(xi / w, F / 1e5, c, lw=2, label=f"{lab},  U={U:.0f} V")
    ax2.set_xlabel("normalized distance  ξ/w")
    ax2.set_ylabel("field  F  [10⁵ V/cm]")
    ax2.set_title("(b) doping straightens the field → Schottky zone (Sec. VI)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle("Sections V-VI -- space-charge field / carrier profiles",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "fig14_sc_field.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
