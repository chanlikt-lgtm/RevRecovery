"""
compare.py -- side-by-side comparison of the reproduction vs the paper, with a
quantitative accuracy assessment.

Produces:
  compare_fig7.png, compare_fig8.png, compare_fig9.png, compare_fig4.png
    -> model figure (left) beside the paper's own figure (right, rendered from
       the source PDF), with digitized paper data points overlaid on the model
       curves where the paper's plot is quantitatively readable.
  prints an accuracy table (model vs points read off the paper's figures).

Paper points below were digitized by eye from the 1967 figures (a small
log/linear plot), so they carry a reading uncertainty of roughly +/-0.02 in
n/nbar and +/-10% in iR/iF at a decade crossing.  The model curves are the
closed-form Eqs. (21)/(36) -- i.e. the *same* equations the paper plotted -- so
the comparison chiefly checks that our implementation reproduces those curves;
the numerical PDE (open markers) is the independent cross-check.
"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # allow tau/box glyphs on Windows
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import fitz

from benda_spenke_model import (Device, forward_profile, n_analytic,
                                solve_transient, storage_times)

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "Reverse_recovery_processes_in_silicon_power_rectifiers  IEEE.pdf")


def paper_crop(page_index, clip_frac, sc=4):
    doc = fitz.open(PDF)
    page = doc[page_index]
    r = page.rect
    x0, y0, x1, y1 = clip_frac
    clip = fitz.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height,
                     r.x0 + x1 * r.width, r.y0 + y1 * r.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(sc, sc), clip=clip)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)


# ---------------------------------------------------------------- digitized paper data
# Fig. 7 (iR=0): left-edge value n(-d)/nbar, and the flat late-time levels.
FIG7 = {  # t/tau : n(-d)/nbar (left edge) read from paper -- only the
          # unambiguous points: steep early left edge, and the flat late-time
          # levels (whole curve flat = exp(-t/tau)).  The mid-time left edges
          # sit where the paper's right-side curve labels crowd, so are skipped.
    0.0: 1.70, 0.01: 1.52, 0.03: 1.40, 1.00: 0.37, 2.00: 0.135, 3.00: 0.050,
}
# Fig. 8 (iR=10iF): (t/tau) -> list of (x/d, n/nbar) readable points
FIG8 = {
    0.00: [(-1.0, 1.70), (0.0, 0.82), (1.0, 0.93)],
    0.01: [(-0.75, 1.30), (0.0, 0.80), (1.0, 0.30)],
    0.03: [(-0.50, 1.05), (0.0, 0.78)],
    0.10: [(0.0, 0.78), (0.5, 0.40)],
    0.20: [(0.15, 0.32), (0.0, 0.30)],
}
# Fig. 9: iR/iF at which each curve crosses a t0/tau decade line.
FIG9_CROSS = {   # (d/L) : { 'l'|'r' : [(t0/tau, iR/iF_paper), ...] }
    1.0: {'l': [(1e-1, 2.2), (1e-2, 8.5), (1e-3, 30.0)],
          'r': [(1e-1, 4.5), (1e-2, 16.0), (1e-3, 50.0)]},
    0.5: {'l': [(1e-1, 3.0), (1e-2, 11.0), (1e-3, 43.0)],
          'r': [(1e-1, 7.0), (1e-2, 30.0), (1e-3, 100.0)]},
}


def side_by_side(model_png, paper_img, title, out, paper_cap):
    fig = plt.figure(figsize=(13, 5.5))
    ax1 = fig.add_subplot(1, 2, 1); ax1.imshow(mpimg.imread(model_png))
    ax1.set_title("MODEL (this reproduction)", fontsize=10); ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2); ax2.imshow(paper_img)
    ax2.set_title(paper_cap, fontsize=10); ax2.axis("off")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


def compare_fig8():
    """Model curves + paper points (left) beside the paper's Fig. 8 (right)."""
    dev = Device(d_over_L=1.0)
    iR = 10 * dev.iF
    x = np.linspace(-dev.d, dev.d, 401)
    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_subplot(1, 2, 1)
    errs = []
    for tf, pts in FIG8.items():
        na = n_analytic(dev, x, tf * dev.tau, iR) / dev.nbar
        line, = ax.plot(x / dev.d, na, lw=1.4, label=f"t/τ={tf:.2f} (model)")
        px = [p[0] for p in pts]; py = [p[1] for p in pts]
        ax.plot(px, py, "o", color=line.get_color(), mfc="white", ms=7)
        for xd, val in pts:
            m = float(n_analytic(dev, xd * dev.d, tf * dev.tau, iR) / dev.nbar)
            errs.append((f"Fig8 t/τ={tf} x/d={xd:+.2f}", val, m))
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("x / d"); ax.set_ylabel(r"$n/\bar n$")
    ax.set_title("MODEL Eq.(36)  (o = points read from paper)", fontsize=10)
    ax.set_xlim(-1, 1); ax.set_ylim(-0.3, 1.8); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(paper_crop(6, (0.52, 0.03, 0.99, 0.30)))
    ax2.set_title("PAPER Fig. 8  (Benda & Spenke 1967)", fontsize=10); ax2.axis("off")
    fig.suptitle(r"Fig. 8 -- sweeping-out, $i_R=10\,i_F$: model vs paper",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "compare_fig8.png"); fig.savefig(out, dpi=130)
    plt.close(fig); print("wrote", out)
    return errs


def compare_fig9():
    fig = plt.figure(figsize=(13, 5.6))
    rows = []
    for i, dL in enumerate((1.0, 0.5)):
        dev = Device(d_over_L=dL)
        ratios = np.geomspace(1, 200, 40)
        tl = np.array([storage_times(dev, r * dev.iF)[0] / dev.tau for r in ratios])
        tr = np.array([storage_times(dev, r * dev.iF)[1] / dev.tau for r in ratios])
        ax = fig.add_subplot(2, 2, 2 * i + 1)
        ax.loglog(ratios, tl, "r-", lw=1.5, label=r"$t_{0l}/\tau$ model")
        ax.loglog(ratios, tr, "b-", lw=1.5, label=r"$t_{0r}/\tau$ model")
        for key, col, mk in (('l', 'r', 'o'), ('r', 'b', 's')):
            for t0, rp in FIG9_CROSS[dL][key]:
                ax.loglog([rp], [t0], mk, color=col, mfc="white", ms=8)
                # model iR/iF at same t0 (interp on log-log)
                arr = tl if key == 'l' else tr
                mr = np.interp(np.log10(t0), np.log10(arr[::-1]),
                               np.log10(ratios[::-1]))
                rows.append((f"Fig9 d/L={dL} t0{key}/τ={t0:g}", rp, 10 ** mr))
        ax.set_title(f"MODEL vs paper points, d/L={dL}", fontsize=9)
        ax.set_xlabel(r"$i_R/i_F$"); ax.set_ylabel(r"$t_0/\tau$")
        ax.set_xlim(1, 200); ax.set_ylim(1e-3, 1); ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    axp = fig.add_subplot(1, 2, 2)
    axp.imshow(paper_crop(7, (0.05, 0.04, 0.52, 0.32)))
    axp.set_title("PAPER Fig. 9 (a) d=L  (b) d=L/2", fontsize=10); axp.axis("off")
    fig.suptitle("Fig. 9 -- storage times vs current ratio: model vs paper "
                 "(o/□ = paper decade-crossings)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "compare_fig9.png"); fig.savefig(out, dpi=130)
    plt.close(fig); print("wrote", out)
    return rows


def main():
    # image-only side-by-sides for the two qualitative figures
    side_by_side(os.path.join(HERE, "fig4_forward_profile.png"),
                 paper_crop(3, (0.53, 0.55, 0.99, 0.75)),
                 "Fig. 4 -- stationary forward profile: model vs paper",
                 os.path.join(HERE, "compare_fig4.png"),
                 "PAPER Fig. 4 (schematic)")
    side_by_side(os.path.join(HERE, "fig7_recombination_decay.png"),
                 paper_crop(6, (0.06, 0.03, 0.52, 0.30)),
                 r"Fig. 7 -- recombination decay ($i_R=0$): model vs paper",
                 os.path.join(HERE, "compare_fig7.png"),
                 "PAPER Fig. 7")
    e8 = compare_fig8()
    e9 = compare_fig9()

    print("\n" + "=" * 78)
    print("ACCURACY  --  model vs values read from the paper's figures")
    print("=" * 78)
    print(f"{'quantity':<34}{'paper':>10}{'model':>12}{'rel.err':>12}")
    print("-" * 78)

    print("[Fig. 7]  n(-d)/nbar (left edge) and late flat levels, iR=0")
    dev = Device(d_over_L=1.0)
    x, t, N = solve_transient(dev, iR=0.0, t_end=3.0 * dev.tau, nt=6000, i_current=0.0)
    e7 = []
    for tf, pv in FIG7.items():
        k = min(int(round(tf * dev.tau / (t[1] - t[0]))), len(t) - 1)
        mv = N[k][0] / dev.nbar
        e7.append((f"  t/τ={tf:<5g} n(-d)/nbar", pv, mv))
    allrows = e7 + [("[Fig. 8] n/nbar profile points", None, None)] + e8 \
        + [("[Fig. 9] iR/iF at t0/τ decade", None, None)] + e9
    maxerr = 0.0
    for name, pv, mv in allrows:
        if pv is None:
            print(name)
            continue
        if abs(pv) < 1e-6:
            print(f"{name:<34}{pv:>10.3f}{mv:>12.3f}{'(abs)':>12}")
            continue
        re = abs(mv - pv) / abs(pv)
        maxerr = max(maxerr, re)
        print(f"{name:<34}{pv:>10.3f}{mv:>12.3f}{re*100:>11.1f}%")
    print("-" * 78)
    print(f"max relative error vs digitized paper points: {maxerr*100:.1f}%")
    print("(dominated by reading error on the 1967 figures; model curves = Eq.36)")
    print("=" * 78)


if __name__ == "__main__":
    main()
