"""
Individual heatmaps for each protein. One PNG per protein.
Figure width scales with number of positions (fixed cell width ~1 mm).
All font sizes and colorbar width scale with figure dimensions.
All residue positions are shown on the x-axis.

Writes: mgrb_heatmap.png, phop_heatmap.png, phoq_heatmap.png,
        pmra_heatmap.png, pmrb_heatmap.png
"""
import logging
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PROTEINS = [
    ("mgrb", "MgrB"),
    ("phop", "PhoP"),
    ("phoq", "PhoQ"),
    ("pmra", "PmrA"),
    ("pmrb", "PmrB"),
]

AA_ORDER = ["G", "A", "V", "L", "I", "P", "F", "W", "M", "S",
            "T", "C", "Y", "H", "D", "E", "N", "Q", "K", "R"]

METRIC = "ddG_total_score"
CMAP = "RdBu_r"

CELL_W = 0.040       # inches per position
MIN_PANEL_W = 3.0
MAX_PANEL_W = 16.0
PANEL_H = 3.0
LEFT_FRAC = 0.08     # left margin as fraction of fig_w
CBAR_H_FRAC = 0.06   # colorbar width as fraction of PANEL_H
CBAR_GAP_FRAC = 0.03 # gap between panel and colorbar as fraction of fig_w
RIGHT_FRAC = 0.05    # right margin as fraction of fig_w
PAD_TOP = 0.45
PAD_BOT = 0.55


def load_pivot(name):
    csv_path = os.path.join(ROOT, name, "dms_output", "DMS_report.csv")
    df = pd.read_csv(csv_path)
    df["pos_label"] = df["WT"] + df["Position_PDB"].astype(str)
    pos_order = (
        df[["Position_PDB", "pos_label"]].drop_duplicates()
        .sort_values("Position_PDB")["pos_label"].tolist()
    )
    pivot = df.pivot_table(
        index="Mutation", columns="pos_label",
        values=METRIC, aggfunc="first",
    ).reindex(index=AA_ORDER, columns=pos_order)
    pivot.columns.name = None
    pivot.index.name = None
    wt_map = (
        df[["pos_label", "WT"]].drop_duplicates()
        .set_index("pos_label")["WT"].to_dict()
    )
    return df, pivot, pos_order, wt_map


def global_vmax(panels):
    vals = pd.concat([p["df"][METRIC].dropna() for p in panels])
    return float(vals.abs().quantile(0.97))


def scaled_fonts(fig_w):
    base = max(8.0, min(13.0, 9.0 + (fig_w - 10.0) * 0.40))
    return {
        "title":      base + 1.0,
        "axis_label": base,
        "tick_y":     max(6.0, min(8.0, base - 2.0)),
        "tick_x":     max(6.0, min(8.0, base - 2.0)),
        "cbar_label": base - 1.0,
        "legend":     base - 1.0,
    }


def plot_one(panel, norm, out_path):
    n = len(panel["positions"])
    panel_w = max(MIN_PANEL_W, min(MAX_PANEL_W, n * CELL_W))

    # derive all margins from fig_w to keep proportions consistent
    # estimate fig_w first, then solve for exact value
    cbar_w = PANEL_H * CBAR_H_FRAC
    # fig_w = left + panel_w + cbar_gap + cbar_w + right
    # left = LEFT_FRAC * fig_w, cbar_gap = CBAR_GAP_FRAC * fig_w, right = RIGHT_FRAC * fig_w
    # fig_w * (1 - LEFT_FRAC - CBAR_GAP_FRAC - RIGHT_FRAC) = panel_w + cbar_w
    fixed_fracs = LEFT_FRAC + CBAR_GAP_FRAC + RIGHT_FRAC
    fig_w = (panel_w + cbar_w) / (1.0 - fixed_fracs)

    left      = LEFT_FRAC * fig_w
    cbar_gap  = CBAR_GAP_FRAC * fig_w
    right_pad = RIGHT_FRAC * fig_w   # noqa: F841 (kept for layout clarity)
    fig_h = PAD_TOP + PANEL_H + PAD_BOT

    fonts = scaled_fonts(fig_w)
    cell_pts = panel_w / n * 72.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([
        left / fig_w,
        PAD_BOT / fig_h,
        panel_w / fig_w,
        PANEL_H / fig_h,
    ])

    sns.heatmap(
        panel["pivot"], ax=ax, cmap=CMAP, norm=norm,
        cbar=False, linewidths=0.0,
        xticklabels=False, yticklabels=AA_ORDER,
    )
    ax.set_xlabel("Residue Position", fontsize=fonts["axis_label"],
                  fontweight="bold", labelpad=4)
    ax.set_ylabel("Amino Acid Substitution", fontsize=fonts["axis_label"],
                  fontweight="bold", labelpad=4)

    for j, pos in enumerate(panel["pivot"].columns):
        wt = panel["wt_map"].get(pos)
        if wt and wt in panel["pivot"].index:
            row_i = list(panel["pivot"].index).index(wt)
            ax.plot(j + 0.5, row_i + 0.5,
                    marker="o", color="black",
                    markersize=max(1.5, cell_pts * 0.25), zorder=5)

    ax.tick_params(axis="y", rotation=0, labelsize=fonts["tick_y"])

    # major ticks: labeled positions; minor ticks: all remaining positions
    cell_w_in = panel_w / n
    avg_label_chars = 3.5
    label_w_in = fonts["tick_x"] * 0.6 * avg_label_chars / 72.0
    label_step = max(1, int(label_w_in / cell_w_in) + 1)

    major_pos = [j + 0.5 for j in range(0, n, label_step)]
    major_labels = [panel["pivot"].columns[j] for j in range(0, n, label_step)]
    minor_pos = [j + 0.5 for j in range(n) if j % label_step != 0]

    ax.set_xticks(major_pos)
    ax.set_xticklabels(major_labels, rotation=90, fontsize=fonts["tick_x"])
    ax.set_xticks(minor_pos, minor=True)
    ax.tick_params(axis="x", which="major", length=4, width=0.8)
    ax.tick_params(axis="x", which="minor", length=2, width=0.5)

    ax.set_title(
        f"Deep Mutational Scanning\n{panel['title']}  (n = {n})",
        fontsize=fonts["title"], fontweight="bold", loc="center", pad=6,
    )

    cbar_left = (left + panel_w + cbar_gap) / fig_w
    cbar_ax = fig.add_axes([cbar_left, PAD_BOT / fig_h, cbar_w / fig_w, PANEL_H / fig_h])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("ddG Total Score (REU)",
                 fontsize=fonts["cbar_label"], fontweight="bold")
    cb.ax.tick_params(labelsize=fonts["tick_y"])

    wt_handle = mlines.Line2D([], [], color="black", marker="o", linestyle="None",
                               markersize=fonts["legend"] * 0.5, label="Wild-type")
    fig.legend(handles=[wt_handle], loc="upper right",
               bbox_to_anchor=(0.995, 0.995), frameon=False,
               fontsize=fonts["legend"])

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s", out_path)


def main():
    panels = []
    for name, title in PROTEINS:
        df, pivot, positions, wt_map = load_pivot(name)
        panels.append({"name": name, "title": title, "df": df,
                       "pivot": pivot, "positions": positions, "wt_map": wt_map})

    vmax = global_vmax(panels)
    logging.info("Global vmax (97th pct): %.2f REU", vmax)
    norm = Normalize(vmin=-vmax, vmax=vmax)

    for panel in panels:
        out = os.path.join(ROOT, f"{panel['name']}_heatmap.png")
        plot_one(panel, norm, out)


if __name__ == "__main__":
    main()
