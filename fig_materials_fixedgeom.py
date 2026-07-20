"""
fig_materials_fixedgeom.py -- Si / 4H-SiC / GaN at the SAME physical base.

The natural-device comparison (fig_materials_extra) fixes d/L=1, so each device
is one diffusion length thick and the absolute voltages are not comparable
(they scale with each material's own tau -> base thickness).  Here instead the
BASE THICKNESS is fixed (2d = 100 um) so the peak recovery voltages ARE directly
comparable.

Physical constraint: the Benda-Spenke model assumes the whole base is
conductivity-modulated in the forward state (d <~ L).  A 100 um base is many
diffusion lengths for the real (short) SiC/GaN lifetimes, so we instead quote,
for each material, the ambipolar lifetime it WOULD NEED to fully modulate a
100 um base (d/L = 1 => tau_req = d^2/D).  That required lifetime is itself the
key result: GaN would need ~17 us, ~10^4x its achievable ~1 ns -- i.e. GaN
cannot make a thick bipolar-modulated PiN, which is why GaN power devices are
thin/unipolar.

With the geometry fixed, the peak recovery voltage scales as 1/sqrt(eps*mu_p)
(lower for wide-bandgap), while the field it reaches is compared against each
material's critical field Ec -- the real payoff: WBG builds a bit more voltage
per micron but stays far below avalanche.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benda_spenke_model import Device, storage_times, VT
from space_charge_model import voltage_recovery, meet_time, sc_widths
from materials import MATERIALS

HERE = os.path.dirname(__file__)
BASE_UM = 100.0                     # full base thickness (um)
D_TARGET = BASE_UM / 2 * 1e-4       # half-width d (cm)
IF = 100.0
IR_RATIO = 10.0


def make_device(m):
    """Device with base fixed at 2*D_TARGET, lifetime set so d/L = 1."""
    D = 2 * m.mu_n * m.mu_p / (m.mu_n + m.mu_p) * VT      # ambipolar D (cm^2/s)
    tau_req = D_TARGET**2 / D
    dev = Device(d_over_L=1.0, mu_n=m.mu_n, mu_p=m.mu_p, tau=tau_req,
                 iF=IF, eps_r=m.eps_r)
    return dev, tau_req


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    rows = []
    for m in MATERIALS:
        dev, tau_req = make_device(m)
        iR = IR_RATIO * dev.iF
        t0l, t0r = storage_times(dev, iR, method="analytic")
        tm = meet_time(dev, iR, t0l, t0r)
        t = np.linspace(0, tm, 400)
        _, _, U = voltage_recovery(dev, iR, t, t0l, t0r)
        Upk = U[-1]
        wl, _ = sc_widths(dev, iR, tm, t0l, t0r)
        Fpk = np.sqrt(2 * iR * float(wl) / (dev.eps * m.mu_p))   # peak field (V/cm)
        rows.append((m, tau_req, tm, Upk, Fpk))
        # (a) waveform, absolute time in microseconds
        ax1.plot(t * 1e6, U, color=m.color, lw=2, label=f"{m.name}")
        ax1.plot(tm * 1e6, Upk, "o", color=m.color, ms=7)

    ax1.set_xlabel("time  t  [µs]  (absolute)")
    ax1.set_ylabel(r"$U_{rect}$  [V]")
    ax1.set_title(f"(a) recovery waveform, fixed {BASE_UM:.0f} µm base "
                  "(∘ = peak)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # (b) peak field reached vs critical field Ec (margin to dynamic avalanche)
    names = [m.name for m, *_ in rows]
    xpos = np.arange(len(names))
    Fpk = [r[4] / 1e6 for r in rows]           # MV/cm
    Ec = [r[0].Ec / 1e6 for r in rows]
    ax2.bar(xpos - 0.2, Fpk, 0.4, color=[r[0].color for r in rows],
            alpha=0.9, label="peak field reached")
    ax2.bar(xpos + 0.2, Ec, 0.4, color=[r[0].color for r in rows],
            alpha=0.35, hatch="//", label="critical field $E_c$")
    for i, r in enumerate(rows):
        frac = r[4] / r[0].Ec * 100
        ax2.text(i, max(Fpk[i], Ec[i]) * 1.05, f"{frac:.0f}% of $E_c$",
                 ha="center", fontsize=8)
    ax2.set_xticks(xpos); ax2.set_xticklabels(names)
    ax2.set_yscale("log")
    ax2.set_ylabel("field  [MV/cm]")
    ax2.set_title("(b) peak recovery field vs avalanche field $E_c$")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y", which="both")

    fig.suptitle(f"Reverse recovery at a fixed {BASE_UM:.0f} µm base: "
                 "Si / 4H-SiC / GaN  (iR = 10 iF)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "fig_mat_fixedgeom.png")
    fig.savefig(out, dpi=130); print("wrote", out)

    print(f"\nfixed base 2d = {BASE_UM:.0f} um, iF={IF:.0f}, iR/iF={IR_RATIO:.0f}")
    print(f"{'mat':<8}{'tau_req':>11}{'(real tau)':>12}{'t_meet':>11}"
          f"{'U_peak':>10}{'F_peak':>12}{'F/Ec':>8}")
    for m, tau_req, tm, Upk, Fpk in rows:
        print(f"{m.name:<8}{tau_req*1e6:>9.2f}us{m.tau*1e6:>10.4f}us"
              f"{tm*1e6:>9.3f}us{Upk:>9.0f}V{Fpk/1e6:>10.2f}MV/cm"
              f"{Fpk/m.Ec*100:>6.0f}%")
    print("\nNote: GaN's required tau (~17 us) is ~10^4x its achievable ~1 ns")
    print("-> a thick bipolar-modulated GaN PiN is not physical; GaN power")
    print("   devices are thin/unipolar.  The row is the 'if it could' limit.")


if __name__ == "__main__":
    main()
