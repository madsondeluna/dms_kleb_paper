"""
TCS1 heatmap: PhoP and PhoQ side by side.
Double-column format (~180 mm). Writes: tcs1_heatmap.png, tcs1_heatmap.pdf
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

AA_ORDER = ["G", "A", "V", "L", "I", "P", "F", "W", "M", "S",
            "T", "C", "Y", "H", "D", "E", "N", "Q", "K", "R"]
METRIC = "ddG_total_score"
CMAP = "RdBu_r"
ALL_PROTEINS = ["mgrb", "phop", "phoq", "pmra", "pmrb"]


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


def global_vmax():
    vals = []
    for name in ALL_PROTEINS:
        df, _, _, _ = load_pivot(name)
        vals.append(df[METRIC].dropna())
    return float(pd.concat(vals).abs().quantile(0.97))


def add_wt_markers(ax, pivot, wt_map, panel_w_in, n):
    ms = max(1.5, min(4.0, panel_w_in / n * 72 * 0.40))
    for j, pos in enumerate(pivot.columns):
        wt = wt_map.get(pos)
        if wt and wt in pivot.index:
            row_i = list(pivot.index).index(wt)
            ax.plot(j + 0.5, row_i + 0.5, marker="o",
                    color="white", markeredgecolor="black",
                    markeredgewidth=0.4, markersize=ms, zorder=5)


def set_xticks(ax, n, columns):
    step = 5 if n <= 30 else 10 if n <= 100 else 25 if n <= 250 else 50
    idx = list(range(0, n, step))
    ax.set_xticks([j + 0.5 for j in idx])
    ax.set_xticklabels([columns[j] for j in idx], rotation=90, fontsize=6)


def main():
    proteins = [("phop", "PhoP"), ("phoq", "PhoQ")]
    panels = []
    for name, title in proteins:
        df, pivot, positions, wt_map = load_pivot(name)
        panels.append({"title": title, "pivot": pivot,
                       "positions": positions, "wt_map": wt_map})

    n_pos = [len(p["positions"]) for p in panels]
    total_n = sum(n_pos)

    FIG_W = 7.09
    LEFT = 0.55
    GAP = 0.10
    PANEL_H = 3.0
    PAD_TOP = 0.40
    PAD_BOT = 0.70
    CBAR_GAP = 0.10
    CBAR_W = 0.18
    RIGHT_PAD = 0.50

    avail = FIG_W - LEFT - (len(panels) - 1) * GAP - CBAR_GAP - CBAR_W - RIGHT_PAD
    pw = [avail * n / total_n for n in n_pos]
    fig_h = PAD_TOP + PANEL_H + PAD_BOT

    vmax = global_vmax()
    logging.info("Global vmax (97th pct): %.2f REU", vmax)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    fig = plt.figure(figsize=(FIG_W, fig_h))

    x = LEFT
    for i, panel in enumerate(panels):
        n = n_pos[i]
        w = pw[i]
        ax = fig.add_axes([x / FIG_W, PAD_BOT / fig_h, w / FIG_W, PANEL_H / fig_h])
        yticklabels = AA_ORDER if i == 0 else False
        sns.heatmap(panel["pivot"], ax=ax, cmap=CMAP, norm=norm,
                    cbar=False, linewidths=0.0,
                    xticklabels=False, yticklabels=yticklabels)
        ax.set_xlabel("")
        ax.set_ylabel("")
        add_wt_markers(ax, panel["pivot"], panel["wt_map"], w, n)
        set_xticks(ax, n, panel["pivot"].columns.tolist())
        if i == 0:
            ax.tick_params(axis="y", rotation=0, labelsize=7)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_title(f"{panel['title']}  (n = {n})", fontsize=10,
                     fontweight="bold", loc="left", pad=3)
        x += w + GAP

    fig.text(
        LEFT * 0.28 / FIG_W,
        (PAD_BOT + PANEL_H / 2) / fig_h,
        "Mutation", rotation=90, fontsize=11, fontweight="bold",
        va="center", ha="center",
    )
    fig.text(
        (LEFT + sum(pw) / 2 + GAP / 4) / FIG_W,
        PAD_BOT * 0.20 / fig_h,
        "Residue position", fontsize=11, fontweight="bold",
        va="center", ha="center",
    )

    cbar_left = (x - GAP + CBAR_GAP) / FIG_W
    cbar_ax = fig.add_axes([cbar_left, PAD_BOT / fig_h, CBAR_W / FIG_W, PANEL_H / fig_h])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("ddG Total Score (REU)", fontsize=9, fontweight="bold")
    cb.ax.tick_params(labelsize=8)

    wt_handle = mlines.Line2D([], [], color="white", marker="o", linestyle="None",
                               markeredgecolor="black", markeredgewidth=0.5,
                               markersize=5, label="Wild-type")
    fig.legend(handles=[wt_handle], loc="upper right",
               bbox_to_anchor=(0.995, 0.995), frameon=False, fontsize=9)

    for fmt in ("png", "pdf"):
        out = os.path.join(ROOT, f"tcs1_heatmap.{fmt}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        logging.info("Saved: %s", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
