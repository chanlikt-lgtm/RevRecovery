"""
build_report.py -- assemble a side-by-side reproduction report (no LaTeX).

Stitches the model figures (fig4/fig7/fig8/fig9_*.png in this folder) next to
crops rendered on the fly from the source paper PDF, plus a title page and the
verification table.  Writes benda_spenke_report.pdf in this folder.

Renders the paper figures directly from the PDF so no extra crop files are left
lying around (keeping everything, as requested, in the one folder).
"""
import os
import io
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "Reverse_recovery_processes_in_silicon_power_rectifiers  IEEE.pdf")

import verify  # noqa: E402


def paper_crop(page_index, clip_frac):
    """Render a clipped region of a PDF page to an RGB array.
    clip_frac = (x0,y0,x1,y1) as fractions of the page rect."""
    doc = fitz.open(PDF)
    page = doc[page_index]
    r = page.rect
    x0, y0, x1, y1 = clip_frac
    clip = fitz.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height,
                     r.x0 + x1 * r.width, r.y0 + y1 * r.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img


def text_page(pdf, title, body, fontsize=11):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.08, 0.88, body, fontsize=fontsize, va="top", family="monospace")
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def pair_page(pdf, title, model_png, paper_img, cap_model, cap_paper):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax1 = fig.add_subplot(2, 1, 1)
    if os.path.exists(model_png):
        ax1.imshow(mpimg.imread(model_png))
    ax1.set_title(cap_model, fontsize=9)
    ax1.axis("off")
    ax2 = fig.add_subplot(2, 1, 2)
    if paper_img is not None:
        ax2.imshow(paper_img)
    ax2.set_title(cap_paper, fontsize=9)
    ax2.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(pdf, format="pdf")
    plt.close(fig)


def main():
    out = os.path.join(HERE, "benda_spenke_report.pdf")
    with PdfPages(out) as pdf:
        text_page(
            pdf,
            "Reproduction: Reverse Recovery in Silicon Power Rectifiers",
            "H. Benda and E. Spenke, Proc. IEEE 55(8), 1331-1354, Aug. 1967\n\n"
            "High-injection p-s-n / p-i-n rectifier.  The excess plasma obeys\n"
            "the ambipolar diffusion equation dn/dt = D n_xx - n/tau with\n"
            "high-level current boundary conditions.  Two independent solvers:\n"
            "  - closed form  Eq. (21) forward,  Eq. (36) transient\n"
            "  - numerical implicit ambipolar PDE (Neumann BCs)\n\n"
            "Reproduced figures:\n"
            "  Fig. 4  stationary forward profile (asymmetric catenary)\n"
            "  Fig. 7  decay by recombination (iR = 0)\n"
            "  Fig. 8  sweeping-out by strong reverse current (iR = 10 iF)\n"
            "  Fig. 9  storage times t0l, t0r vs iR/iF (d=L and d=L/2)\n\n"
            "See README.md and derivation.md for the full derivation.",
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            verify.main()
        text_page(pdf, "Model vs paper -- verification", buf.getvalue(), fontsize=7.5)

        # Fig. 7 & 8 are both on page index 6 (p.1337); Fig. 9 on page 7 (p.1338).
        pair_page(pdf, "Figs. 7 & 8 -- carrier removal (recombination / sweep-out)",
                  os.path.join(HERE, "fig8_sweepout.png"),
                  paper_crop(6, (0.05, 0.05, 0.98, 0.36)),
                  "MODEL Fig. 8: iR = 10 iF (dots = analytic Eq. 36)",
                  "PAPER p.1337: Fig. 7 (iR=0) and Fig. 8 (iR=10iF)")
        pair_page(pdf, "Fig. 7 -- decay by recombination (iR = 0)",
                  os.path.join(HERE, "fig7_recombination_decay.png"),
                  None,
                  "MODEL Fig. 7: t/tau = 0 ... 3.0", "")
        pair_page(pdf, "Fig. 9 -- storage (delay) times vs iR/iF",
                  os.path.join(HERE, "fig9_storage_time.png"),
                  paper_crop(7, (0.05, 0.05, 0.55, 0.34)),
                  "MODEL Fig. 9: t0l/tau, t0r/tau  (open markers = numeric PDE)",
                  "PAPER p.1338: Fig. 9 (a) d=L  (b) d=L/2")
        pair_page(pdf, "Fig. 4 -- stationary forward profile",
                  os.path.join(HERE, "fig4_forward_profile.png"),
                  None,
                  "MODEL Fig. 4: n/nbar vs x/d (Eq. 21 + numeric check)", "")

        # --- Sections IV-VII: space charge & voltage recovery ---
        text_page(
            pdf, "Sections IV-VII -- space charge & voltage recovery",
            "After the storage times t0l, t0r the quasi-neutral plasma model\n"
            "hands off to the space-charge stage:\n\n"
            "  Sec. IV  two sweeping-out boundaries advance inward; widths\n"
            "           grow linearly, left mu_n/mu_p = 3x faster (Eq. 46-49).\n"
            "  Sec. V   p-i-n: sqrt-field in each zone (Eq. 54) -> voltage\n"
            "           U_sc ~ w^{3/2} ~ (t-t0)^{3/2} (Eq. 58-61); left ~90%.\n"
            "  Sec. VI  uniform doping -> Schottky zone, U_sc ~ w^2.\n"
            "  Sec. VII diffused (exponential) doping: general doped solver.\n\n"
            "Verified: boundary-speed ratio = mu_n/mu_p exactly; left voltage\n"
            "share = 90.0% for b=3 (paper's number); fitted voltage exponents\n"
            "1.53 (p-i-n, theory 1.5) and 2.02 (Schottky, theory 2).",
        )
        pair_page(pdf, "Sec. IV -- space-charge-zone growth",
                  os.path.join(HERE, "fig_scz_growth.png"),
                  paper_crop(8, (0.05, 0.30, 0.52, 0.62)),
                  "MODEL: boundary trajectories & linear width growth",
                  "PAPER Figs. 11-13 (schematic sweeping-out)")
        pair_page(pdf, "Sec. V/VI -- space-charge field & carriers",
                  os.path.join(HERE, "fig14_sc_field.png"),
                  None,
                  "MODEL: sqrt-field + falling carrier (Fig. 14); "
                  "doping -> linear field (Fig. 16)", "")
        pair_page(pdf, "Sec. V -- reverse-recovery VOLTAGE",
                  os.path.join(HERE, "fig_voltage_recovery.png"),
                  None,
                  "MODEL: U_rect(t) ~ (t-t0)^{3/2} (p-i-n) / ^2 (Schottky); "
                  "left zone ~90%", "")

        # --- extension: wide-bandgap material comparison ---
        text_page(
            pdf, "Extension -- Si vs 4H-SiC vs GaN",
            "The silicon model is re-run with cited 4H-SiC and GaN parameter\n"
            "sets (materials.py).  Going Si -> SiC -> GaN:\n\n"
            "  * mobility asymmetry B rises 0.50 -> 0.78 -> 0.94, so the\n"
            "    forward plasma and the recovery become increasingly one-sided;\n"
            "  * the left (hole) zone's voltage share climbs 90% -> 98% -> ~100%;\n"
            "  * the recovery space charge can hold ~70-130x more voltage before\n"
            "    dynamic avalanche -- the Baliga FOM (eps mu Ec^3) emerging from\n"
            "    the reverse-recovery physics.\n\n"
            "NB GaN's hole mobility (hence B, and its hole-zone avalanche\n"
            "margin) is the most uncertain input; see materials.py.",
        )
        pair_page(pdf, "Extension -- reverse recovery: Si / 4H-SiC / GaN",
                  os.path.join(HERE, "fig_materials_compare.png"),
                  None,
                  "MODEL: (a) forward profile (b) storage-time asymmetry "
                  "(c) voltage split (d) avalanche margin", "")
        for png, cap in [
            ("fig_mat_waveform.png",
             "U_rect(t) waveform, absolute time (WBG recovers ~10^4x faster)"),
            ("fig_mat_fig9.png",
             "storage times t0l/t0r vs iR/iF (Fig. 9 style); n-side fans out"),
            ("fig_mat_avalanche.png",
             "avalanche margin: hole vs electron zone, and GaN-mu_p band"),
            ("fig_mat_fixedgeom.png",
             "fixed 100um base: comparable peaks; Si reaches 60% of Ec, "
             "SiC 17%, GaN 28%"),
        ]:
            p = os.path.join(HERE, png)
            if os.path.exists(p):
                pair_page(pdf, "Extension -- " + png, p, None, "MODEL: " + cap, "")
    print("wrote", out)


if __name__ == "__main__":
    main()
