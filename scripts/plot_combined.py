"""
Combined heatmap of the 5 Klebsiella DMS scans.

Vertical stack of 5 panels, panel width proportional to the number of
positions in each protein, shared y-axis (the 20 canonical amino acids),
single shared colorbar on the right.

Reads from:
    <protein>/dms_output/DMS_report.csv

Writes:
    combined_dms_heatmap.png
    combined_dms_heatmap.pdf
"""

import logging
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
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


def load_pivot(name: str):
    csv_path = os.path.join(ROOT, name, "dms_output", "DMS_report.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing DMS report for {name}: {csv_path}\n"
            f"Run `python scripts/run_dms.py {name}` first."
        )
    df = pd.read_csv(csv_path)
    df["pos_label"] = df["WT"] + df["Position_PDB"].astype(str)
    position_order = (
        df[["Position_PDB", "pos_label"]].drop_duplicates()
        .sort_values("Position_PDB")["pos_label"].tolist()
    )
    pivot = df.pivot_table(
        index="Mutation", columns="pos_label",
        values=METRIC, aggfunc="first",
    ).reindex(index=AA_ORDER, columns=position_order)
    pivot.columns.name = None
    pivot.index.name = None

    wt_map = (
        df[["pos_label", "WT"]].drop_duplicates()
        .set_index("pos_label")["WT"].to_dict()
    )
    return df, pivot, position_order, wt_map


def main():
    panels = []
    for name, title in PROTEINS:
        df, pivot, positions, wt_map = load_pivot(name)
        panels.append({
            "name": name,
            "title": title,
            "df": df,
            "pivot": pivot,
            "positions": positions,
            "wt_map": wt_map,
        })

    all_values = pd.concat([p["df"][METRIC] for p in panels])
    vmax = float(all_values.abs().quantile(0.97))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = "RdBu_r"

    n_positions = [len(p["positions"]) for p in panels]
    max_n = max(n_positions)
    min_w_frac = 0.04

    panel_h = 2.6
    panel_w_max = 18.0
    pad_top = 0.6
    pad_bottom = 0.85
    pad_between = 0.65
    left_margin = 1.4
    cbar_gap = 0.45
    cbar_w = 0.22
    right_margin = cbar_gap + cbar_w + 0.90

    fig_w = left_margin + panel_w_max + right_margin
    fig_h = pad_top + len(panels) * panel_h + (len(panels) - 1) * pad_between + pad_bottom
    fig = plt.figure(figsize=(fig_w, fig_h))

    for i, panel in enumerate(panels):
        n = n_positions[i]
        frac = max(n / max_n, min_w_frac)
        width = panel_w_max * frac
        bottom = (pad_bottom + (len(panels) - 1 - i) * (panel_h + pad_between)) / fig_h
        height = panel_h / fig_h
        left = left_margin / fig_w
        w_frac = width / fig_w

        ax = fig.add_axes([left, bottom, w_frac, height])

        pivot = panel["pivot"]
        sns.heatmap(
            pivot, ax=ax, cmap=cmap, norm=norm,
            cbar=False, linewidths=0.0,
            xticklabels=False, yticklabels=AA_ORDER,
        )
        ax.set_xlabel("")

        wt_map = panel["wt_map"]
        for col_j, pos in enumerate(pivot.columns):
            wt = wt_map.get(pos)
            if wt in pivot.index:
                row_i = pivot.index.get_loc(wt)
                ax.plot(col_j + 0.5, row_i + 0.5,
                        marker="o", color="black",
                        markersize=2.5, zorder=5)

        ax.tick_params(axis="y", rotation=0, labelsize=8)
        ax.set_ylabel("")

        if n <= 30:
            step = 5
        elif n <= 100:
            step = 10
        elif n <= 250:
            step = 25
        else:
            step = 50
        tick_idx = list(range(0, n, step))
        tick_labels = [pivot.columns[j] for j in tick_idx]
        ax.set_xticks([j + 0.5 for j in tick_idx])
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)

        ax.set_title(f"{panel['title']}  (n = {n})", fontsize=12,
                     fontweight="bold", loc="left", pad=4)

        cbar_left = (left_margin + panel_w_max + cbar_gap) / fig_w
        cbar_ax = fig.add_axes([cbar_left, bottom, cbar_w / fig_w, height])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.ax.tick_params(labelsize=8)

    fig.text(
        left_margin / fig_w * 0.30,
        (pad_bottom + (len(panels) * panel_h + (len(panels) - 1) * pad_between) / 2) / fig_h,
        "Amino acid substitution",
        rotation=90, fontsize=14, fontweight="bold",
        va="center", ha="center",
    )
    fig.text(
        (left_margin + panel_w_max / 2) / fig_w,
        (pad_bottom * 0.25) / fig_h,
        "Residue position",
        fontsize=14, fontweight="bold",
        va="center", ha="center",
    )
    fig.text(
        (left_margin + panel_w_max + cbar_gap + cbar_w + 0.18) / fig_w,
        (pad_bottom + (len(panels) * panel_h + (len(panels) - 1) * pad_between) / 2) / fig_h,
        "ddG Total Score (REU)",
        rotation=90, fontsize=12, fontweight="bold",
        va="center", ha="center",
    )

    wt_marker = mlines.Line2D(
        [], [], color="black", marker="o", linestyle="None",
        markersize=6, label="Wild-type",
    )
    fig.legend(
        handles=[wt_marker],
        loc="upper right",
        bbox_to_anchor=(1.0 - 0.01, 1.0 - 0.02),
        frameon=False, fontsize=10,
    )

    fig.suptitle(
        "Klebsiella two-component system proteins - Cartesian ddG DMS",
        fontsize=15, fontweight="bold",
        x=(left_margin + panel_w_max / 2) / fig_w,
        y=1.0 - 0.005,
    )

    out_png = os.path.join(ROOT, "combined_dms_heatmap.png")
    out_pdf = os.path.join(ROOT, "combined_dms_heatmap.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s", out_png)
    logging.info("Saved: %s", out_pdf)


if __name__ == "__main__":
    main()
