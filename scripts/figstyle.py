#!/usr/bin/env python
r"""
figstyle -- single shared figure style for the Paper-1 revision
===============================================================

One reusable matplotlib style + palette + sizes, applied to EVERY figure so the
whole set is uniform and print-legible (revtex4-2 [aip,pof,reprint], two column).

  - serif text + Computer-Modern math, consistent with the manuscript;
  - base font 9 pt; axis tick labels 8 pt (>= the journal minimum at print size);
  - one colourblind-safe palette (Wong 2011) and line-cycle reused everywhere;
  - single-column width 3.4 in (\columnwidth), full width 7.0 in (\textwidth);
  - tight layout; vector PDF saved at TRUE physical size (no LaTeX rescaling).

Use:
    from figstyle import apply_style, savefig, PALETTE, WONG, COLUMN_WIDTH_IN, TEXT_WIDTH_IN
    plt = apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    ...
    savefig(fig, "fig_name")
"""
import os

COLUMN_WIDTH_IN = 3.4    # \columnwidth (single column)
TEXT_WIDTH_IN = 7.0      # \textwidth   (full, two-column span)

# Font floors (module minimum)
BASE_FONT_PT = 9
TICK_FONT_PT = 8
MIN_FONT_PT = 7         # no annotation/legend text below this

# Colourblind-safe palette (Wong 2011, Nature Methods)
WONG = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
PALETTE = [WONG["blue"], WONG["vermillion"], WONG["green"],
           WONG["orange"], WONG["purple"], WONG["skyblue"], WONG["black"]]
# consistent line-style cycle to pair with the palette
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "figures", "paper1")


def apply_style():
    """Set the shared rcParams and return the pyplot module."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT,
        "axes.labelsize": BASE_FONT_PT,
        "legend.fontsize": TICK_FONT_PT,
        "xtick.labelsize": TICK_FONT_PT,
        "ytick.labelsize": TICK_FONT_PT,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
    return plt


def savefig(fig, name, also_png=True):
    """Save to figures/paper1 as vector PDF (+ 600-dpi PNG) at true size."""
    os.makedirs(FIGDIR, exist_ok=True)
    exts = ("pdf", "png") if also_png else ("pdf",)
    for ext in exts:
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"),
                    bbox_inches="tight", pad_inches=0.02)
    print(f"  saved figures/paper1/{name}.pdf" + (" / .png" if also_png else ""))
    import matplotlib.pyplot as plt
    plt.close(fig)


def min_font_in_figure(fig):
    """Smallest text font size (pt) actually used in a figure -- for the acceptance
    check that nothing renders below the minimum at print size."""
    sizes = []
    for ax in fig.get_axes():
        texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        texts += list(ax.texts)
        leg = ax.get_legend()
        if leg is not None:
            texts += list(leg.get_texts())
        for t in texts:
            try:
                if t.get_text():
                    sizes.append(t.get_fontsize())
            except Exception:
                pass
    return min(sizes) if sizes else None
