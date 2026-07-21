"""
fig_conformal_edge.py -- conformal-mapping extension of the reverse-recovery
space-charge analysis to 2-D: field crowding at the junction / space-charge-zone
EDGE.

Where conformal mapping legitimately enters (it does NOT apply to the 1-D
diffusion transient): the doped/Schottky swept-out zone of Section VI carries a
constant space charge rho = qN, so in 2-D the potential splits (exactly as in
the SOI Nakagawa Fig-4 method, code fig4_conformal.py)

    phi(x,y) = phi_p(y) + psi(x,y),

a 1-D parabolic particular part phi_p carrying the charge (Poisson) plus a
SOURCE-FREE harmonic psi (Laplace).  psi carries the 2-D boundary shape -- and
at a device edge / junction termination the highly-doped contact forms a corner,
around which psi (hence the field) crowds.  That crowding is a genuine
conformal-mapping problem and it limits the real device: the field peaks at the
edge, so avalanche starts there, below the 1-D parallel-plane value.

Model geometry: the highly-doped contact occupies a 90-degree conducting wedge
(the metallurgical junction meeting the surface at a right angle); the
swept-out space-charge region fills the remaining 270 degrees.  The conformal
map zeta = z^{2/3} sends that 270-degree field wedge to a half-plane, where the
potential is trivial.  Mapping back, the harmonic edge potential is

    psi = C * Im(z^{2/3}),   complex potential  W(z) = -i C z^{2/3},
    E_x - i E_y = -W'(z) = i C (2/3) z^{-1/3}   =>   |E| = (2C/3) r^{-1/3},

the classic r^{-1/3} corner singularity.  Fixing C so the field far from the
corner (r = w, the zone width) equals the 1-D peak field F_1D gives the
enhancement law

    |E|(r) / F_1D = (w / r)^{1/3},

finite in practice at the corner rounding radius r_c.  Applying it to the
fixed-100um-base avalanche fields (fig_materials_fixedgeom) shows the 2-D edge
can push a device into avalanche that the 1-D bulk analysis calls safe.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benda_spenke_model import Device, storage_times, VT
from space_charge_model import meet_time, sc_widths
from materials import MATERIALS

HERE = os.path.dirname(__file__)

ALPHA = 1.5 * np.pi          # 270-degree field wedge (90-degree electrode)
P = np.pi / ALPHA            # = 2/3 : mapping exponent zeta = z^P


# ---------------------------------------------------------------------------
# conformal harmonic edge field  psi = C Im(z^P),  |E| = C P r^{P-1}
# ---------------------------------------------------------------------------
def complex_potential(z, C):
    """W(z) = -i C z^P, so phi = Re(W) = C Im(z^P) (harmonic)."""
    return -1j * C * np.power(z, P)


def field(z, C):
    """E_x - i E_y = -W'(z) = i C P z^{P-1}."""
    return 1j * C * P * np.power(z, P - 1.0)


def enhancement(r_over_w):
    """|E|(r)/F_1D = (w/r)^{1-P} = (w/r)^{1/3}."""
    return np.power(1.0 / r_over_w, 1.0 - P)


# ---------------------------------------------------------------------------
# 1-D peak avalanche field at a fixed base (from fig_materials_fixedgeom)
# ---------------------------------------------------------------------------
def one_d_peak_field(m, base_um=100.0, ir_ratio=10.0, iF=100.0):
    Q = 1.602176634e-19
    D = 2 * m.mu_n * m.mu_p / (m.mu_n + m.mu_p) * VT
    d = base_um / 2 * 1e-4
    tau = d**2 / D
    dev = Device(d_over_L=1.0, mu_n=m.mu_n, mu_p=m.mu_p, tau=tau, iF=iF,
                 eps_r=m.eps_r)
    iR = ir_ratio * dev.iF
    t0l, t0r = storage_times(dev, iR, method="analytic")
    tm = meet_time(dev, iR, t0l, t0r)
    wl, _ = sc_widths(dev, iR, tm, t0l, t0r)
    F1d = np.sqrt(2 * iR * float(wl) / (dev.eps * m.mu_p))
    return F1d, float(wl)


# ---------------------------------------------------------------------------
def verify(C=1.0):
    """Harmonicity self-check: numerical Laplacian of phi = Re(W) ~ 0."""
    h = 1e-4
    rng = np.random.default_rng(0)
    maxlap = 0.0
    for _ in range(200):
        r = rng.uniform(0.2, 1.0)
        th = rng.uniform(0.15, ALPHA - 0.15)      # interior of the wedge
        z0 = r * np.exp(1j * th)
        def phi(zz):
            return np.real(complex_potential(zz, C))
        lap = (phi(z0 + h) + phi(z0 - h) + phi(z0 + 1j * h) + phi(z0 - 1j * h)
               - 4 * phi(z0)) / h**2
        maxlap = max(maxlap, abs(lap))
    return maxlap


def main():
    # ---- (a) equipotential / field map in the 270-degree wedge ----
    C = 1.0
    n = 400
    g = np.linspace(-1.3, 1.3, n)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    th = np.mod(np.angle(Z), 2 * np.pi)
    inwedge = (th <= ALPHA) & (np.abs(Z) > 0.02)     # field region (mask 90-deg electrode)
    with np.errstate(all="ignore"):
        phi = np.real(complex_potential(Z, C))
        Emag = np.abs(field(Z, C))
    phi = np.where(inwedge, phi, np.nan)
    Emag = np.where(inwedge, Emag, np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    pc = ax1.pcolormesh(X, Y, np.log10(Emag), shading="auto", cmap="inferno")
    ax1.contour(X, Y, phi, levels=14, colors="cyan", linewidths=0.6, alpha=0.8)
    # draw the 90-degree electrode wedge (whole fourth quadrant)
    ax1.fill([0, 1.3, 1.3, 0], [0, 0, -1.3, -1.3], color="0.6", zorder=5)
    ax1.text(0.65, -0.65, "contact\n(90° wedge)", fontsize=8, ha="center", zorder=6)
    ax1.set_aspect("equal"); ax1.set_xlim(-1.3, 1.3); ax1.set_ylim(-1.3, 1.3)
    ax1.set_title("(a) edge field crowding (conformal ζ = z^{2/3})\n"
                  "cyan = equipotentials, color = log|E|")
    fig.colorbar(pc, ax=ax1, label=r"$\log_{10}|E|$ (arb.)", shrink=0.8)

    # verify harmonic
    lap = verify(C)

    # ---- (b) enhancement law + corrected avalanche margins ----
    rw = np.geomspace(1e-3, 1.0, 200)
    ax2.loglog(rw, enhancement(rw), "k", lw=2, label=r"$|E|/F_{1D}=(w/r)^{1/3}$")
    ax2.set_xlabel("distance from corner  r / w")
    ax2.set_ylabel(r"field enhancement  $|E| / F_{1D}$")
    ax2.set_title("(b) corner enhancement & edge avalanche")
    ax2.grid(alpha=0.3, which="both")

    # corrected margins: does the edge field reach Ec?
    print("=" * 78)
    print("Conformal edge-field extension  (90° junction corner, r^{-1/3} crowding)")
    print("=" * 78)
    print(f"harmonicity self-check: max |Laplacian(phi)| = {lap:.2e} (~0)")
    print(f"\nmapping exponent P = pi/alpha = {P:.4f}  (field ~ r^{{{P-1:.3f}}})")
    print("\nedge avalanche at fixed 100um base, iR=10iF, corner radius r_c=1um:")
    print(f"{'mat':<8}{'F_1D[MV/cm]':>13}{'w[um]':>8}{'F_edge(r_c)':>13}"
          f"{'Ec[MV/cm]':>11}{'edge/Ec':>9}")
    r_c_um = 1.0
    for m in MATERIALS:
        F1d, w = one_d_peak_field(m)
        w_um = w * 1e4
        enh = (w_um / r_c_um) ** (1.0 - P)
        F_edge = F1d * enh
        frac = F_edge / m.Ec * 100
        flag = "  <-- AVALANCHE" if F_edge > m.Ec else ""
        print(f"{m.name:<8}{F1d/1e6:>13.2f}{w_um:>8.1f}{F_edge/1e6:>13.2f}"
              f"{m.Ec/1e6:>11.2f}{frac:>8.0f}%{flag}")
        # mark on plot
        tag = "AVALANCHE" if F_edge > m.Ec else "safe"
        ax2.plot(r_c_um / w_um, enh, "o", color=m.color, ms=8,
                 label=f"{m.name}: edge {frac:.0f}% of Ec ({tag})")
    ax2.axvline(r_c_um / (one_d_peak_field(MATERIALS[0])[1] * 1e4), color="gray",
                ls=":", lw=0.8)
    ax2.legend(fontsize=8)
    print("\n-> the 2-D corner crowds the field ~3-4x above the 1-D bulk value,")
    print("   so edge termination (not the 1-D base) sets reverse-recovery")
    print("   ruggedness -- invisible to the paper's 1-D model.")
    print("=" * 78)

    fig.suptitle("Conformal-mapping extension: 2-D space-charge edge field "
                 "(Section VI, doped zone)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "fig_conformal_edge.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
