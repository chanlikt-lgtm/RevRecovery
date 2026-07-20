"""
fig_materials_extra.py -- three follow-up comparison plots for Si / 4H-SiC / GaN.

  1. fig_mat_waveform.png   -- reverse-recovery voltage waveform U_rect(t) for
                               the three materials on one ABSOLUTE-time axis.
  2. fig_mat_fig9.png       -- storage times t0l, t0r vs iR/iF for the three
                               materials, in the paper's Fig. 9 log-log style.
  3. fig_mat_avalanche.png  -- dynamic-avalanche margin (a) hole-zone (mu_p) vs
                               electron-zone (mu_n), and (b) swept over GaN's
                               uncertain hole mobility.

All at the "natural" design point d/L = 1, iF = 100 A/cm^2 (so each device is
one ambipolar diffusion length thick).  The absolute time/voltage scales then
reflect each material's own lifetime tau and permittivity -- which is itself a
result: the wide-bandgap devices recover far faster (less stored charge) at a
different voltage scale.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benda_spenke_model import storage_times, EPS0
from space_charge_model import voltage_recovery, meet_time
from materials import MATERIALS, SI, SIC, GAN

HERE = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# 1. reverse-recovery voltage waveform  U_rect(t)   (absolute time)
# ---------------------------------------------------------------------------
def fig_waveform():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    info = []
    for m in MATERIALS:
        dev = m.device(d_over_L=1.0)
        iR = 10 * dev.iF
        t0l, t0r = storage_times(dev, iR, method="analytic")
        tm = meet_time(dev, iR, t0l, t0r)
        t = np.linspace(0, tm, 400)
        _, _, U = voltage_recovery(dev, iR, t, t0l, t0r)
        # (a) absolute time, log-log (timescales span ~10^4)
        ax1.loglog(np.maximum(t, 1e-15), np.maximum(U, 1e-3), color=m.color,
                   lw=2, label=f"{m.name}")
        ax1.plot(tm, U[-1], "o", color=m.color, ms=7)
        # (b) normalized time -> shapes overlay
        ax2.plot(t / dev.tau, U / U[-1], color=m.color, lw=2, label=m.name)
        info.append((m.name, tm, U[-1], dev.d * 1e4, dev.tau))

    ax1.set_xlabel("time  t  [s]  (absolute)")
    ax1.set_ylabel(r"$U_{rect}$  [V]")
    ax1.set_title("(a) waveform on absolute time (∘ = peak at boundary meet)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, which="both")
    ax2.set_xlabel(r"normalized time  t / $\tau$")
    ax2.set_ylabel(r"$U_{rect} / U_{peak}$")
    ax2.set_title("(b) same shape (all ∝ (t−t₀)^{3/2}) vs t/τ")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle("Reverse-recovery voltage waveform: Si / 4H-SiC / GaN "
                 "(d/L=1, iR=10 iF)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "fig_mat_waveform.png")
    fig.savefig(out, dpi=130); print("wrote", out)
    print("  material   t_meet        U_peak     d[um]    tau")
    for n, tm, up, d, tau in info:
        print(f"  {n:<8} {tm:.3e} s  {up:8.1f} V  {d:7.2f}  {tau:.1e} s")


# ---------------------------------------------------------------------------
# 2. storage times t0l, t0r vs iR/iF  (paper Fig. 9 style)
# ---------------------------------------------------------------------------
def fig_fig9():
    ratios = np.geomspace(1.5, 200, 16)
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in MATERIALS:
        dev = m.device(d_over_L=1.0)
        tl, tr = [], []
        for r in ratios:
            a, b = storage_times(dev, r * dev.iF, method="analytic")
            tl.append(a / dev.tau); tr.append(b / dev.tau)
        ax.loglog(ratios, tl, color=m.color, lw=2, ls="-",
                  label=f"{m.name}  $t_{{0l}}$ (p-side)")
        ax.loglog(ratios, tr, color=m.color, lw=2, ls="--",
                  label=f"{m.name}  $t_{{0r}}$ (n-side)")
    ax.set_xlabel(r"$i_R / i_F$")
    ax.set_ylabel(r"$t_0 / \tau$")
    ax.set_title("Storage times vs current ratio (Fig. 9 style)\n"
                 "solid = t0l (p-side, empties first),  dashed = t0r (n-side)")
    ax.set_xlim(1.5, 200); ax.set_ylim(1e-4, 1)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_mat_fig9.png")
    fig.savefig(out, dpi=130); print("wrote", out)


# ---------------------------------------------------------------------------
# 3. avalanche margin: n-side vs p-side, and GaN mu_p sweep
# ---------------------------------------------------------------------------
def fig_avalanche():
    iR = np.geomspace(50, 5000, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) hole-zone (mu_p, dominant) vs electron-zone (mu_n) margin
    for m in MATERIALS:
        eps = m.eps_r * EPS0
        Up = (1.0 / 3.0) * eps * m.mu_p * m.Ec**3 / iR   # left / hole zone
        Un = (1.0 / 3.0) * eps * m.mu_n * m.Ec**3 / iR   # right / electron zone
        ax1.loglog(iR, Up, color=m.color, lw=2, ls="-",
                   label=f"{m.name}  hole zone (μp)")
        ax1.loglog(iR, Un, color=m.color, lw=1.5, ls=":",
                   label=f"{m.name}  electron zone (μn)")
    ax1.set_xlabel(r"$i_R$  [A/cm$^2$]")
    ax1.set_ylabel("max voltage before avalanche  [V]")
    ax1.set_title("(a) hole-zone (μp, solid) vs electron-zone (μn, dotted)")
    ax1.grid(alpha=0.3, which="both"); ax1.legend(fontsize=7.5)

    # (b) GaN hole-mobility uncertainty band vs Si, SiC references (hole zone)
    eps_g = GAN.eps_r * EPS0
    Ug_lo = (1.0 / 3.0) * eps_g * 30 * GAN.Ec**3 / iR
    Ug_hi = (1.0 / 3.0) * eps_g * 200 * GAN.Ec**3 / iR
    ax2.fill_between(iR, Ug_lo, Ug_hi, color=GAN.color, alpha=0.25,
                     label="GaN  (μp = 30…200)")
    ax2.loglog(iR, (1.0/3.0)*eps_g*30*GAN.Ec**3/iR, color=GAN.color, lw=1, ls="--")
    ax2.loglog(iR, (1.0/3.0)*eps_g*200*GAN.Ec**3/iR, color=GAN.color, lw=1, ls="--")
    for m in (SI, SIC):
        eps = m.eps_r * EPS0
        ax2.loglog(iR, (1.0/3.0)*eps*m.mu_p*m.Ec**3/iR, color=m.color, lw=2,
                   label=f"{m.name} (μp={m.mu_p:.0f})")
    ax2.set_xlabel(r"$i_R$  [A/cm$^2$]")
    ax2.set_ylabel("max voltage before avalanche  [V]")
    ax2.set_title("(b) GaN hole-mobility uncertainty (hole zone)")
    ax2.grid(alpha=0.3, which="both"); ax2.legend(fontsize=8)

    fig.suptitle("Dynamic-avalanche margin  ∝ ε·μ·Ec³ / iR  "
                 "(zone-mobility and GaN-μp sensitivity)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "fig_mat_avalanche.png")
    fig.savefig(out, dpi=130); print("wrote", out)
    # print the GaN electron-vs-hole FOM gap
    fom_si = SI.eps_r * SI.mu_p * SI.Ec**3
    print("\n  material  hole-zone FOM   electron-zone FOM  (rel. Si hole)")
    for m in MATERIALS:
        fp = m.eps_r * m.mu_p * m.Ec**3 / fom_si
        fn = m.eps_r * m.mu_n * m.Ec**3 / fom_si
        print(f"  {m.name:<8} {fp:8.1f}x        {fn:8.1f}x")


if __name__ == "__main__":
    fig_waveform()
    fig_fig9()
    fig_avalanche()
