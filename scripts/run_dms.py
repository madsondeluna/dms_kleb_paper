"""
Deep Mutational Scanning for Klebsiella two-component system proteins.

Targets (AlphaFold monomer models, chain A):
    mgrb : MgrB regulatory peptide       (25 aa)
    phop : PhoP response regulator       (223 aa)
    phoq : PhoQ sensor histidine kinase  (488 aa)
    pmra : PmrA response regulator       (223 aa)
    pmrb : PmrB sensor histidine kinase  (365 aa)

Pipeline:

    <name>.pdb (raw AlphaFold model)
        |
        v  pre_relax_multi_decoy()           [5 decoys, coord-cst, pick min E]
    <name>/minimum_energy.pdb
        |
        v  Run_DMS_CartesianDDG_Parallel()   [Frenz 2020, 3 replicas WT + 3 mut]
    <name>/dms_output/DMS_report.csv
        |
        v  plot_heatmap()                    [single-protein heatmap]
    <name>/<name>_dms_heatmap.png

Usage:
    python run_dms.py mgrb            # one protein
    python run_dms.py all             # all five (sequential)

Tunables at the top of the file: N_DECOYS, N_REPLICAS, NEIGHBORHOOD_RADIUS,
N_CPU, BASE_SEED.
"""

import argparse
import logging
import multiprocessing
import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

PROTEINS = {
    "mgrb": {"pdb": "mgrb.pdb", "title": "MgrB"},
    "phop": {"pdb": "phop.pdb", "title": "PhoP"},
    "phoq": {"pdb": "phoq.pdb", "title": "PhoQ"},
    "pmra": {"pdb": "pmra.pdb", "title": "PmrA"},
    "pmrb": {"pdb": "pmrb.pdb", "title": "PmrB"},
}

CHAIN = "A"
N_DECOYS = 5
N_REPLICAS = 3
NEIGHBORHOOD_RADIUS = 8.0
BASE_SEED = 12345
N_CPU = None  # None = auto (cpu_count - 2)


def _resolve_cpu(n):
    if n is None:
        return max(1, os.cpu_count() - 2)
    return max(1, n)


def clean_pdb(pdb_in: str, pdb_out: str) -> None:
    """Keep only ATOM/TER/END records (drops any HETATM if present)."""
    with open(pdb_in) as fh:
        lines = fh.readlines()
    with open(pdb_out, "w") as fh:
        for line in lines:
            if line.startswith(("ATOM", "TER", "END")):
                fh.write(line)
    logging.info("Cleaned PDB: %s -> %s", pdb_in, pdb_out)


def pre_relax(pdb_raw: str, out_dir: str) -> str:
    """Generate N_DECOYS relaxed models, return path to the lowest-energy one."""
    from utils_pyrosetta import pre_relax_multi_decoy
    best_pdb, best_score, all_scores = pre_relax_multi_decoy(
        pdb_in=pdb_raw,
        output_dir=out_dir,
        n_decoys=N_DECOYS,
        base_seed=BASE_SEED,
    )
    logging.info(
        "Pre-relax done. Best E = %.3f REU. Spread = %.3f REU.",
        best_score, max(all_scores) - min(all_scores),
    )
    return best_pdb


def run_dms(pdb_min: str, out_dir: str) -> str:
    """Cartesian ddG DMS (Frenz 2020) over the entire chain."""
    from utils_pyrosetta import Run_DMS_CartesianDDG_Parallel
    os.makedirs(out_dir, exist_ok=True)
    Run_DMS_CartesianDDG_Parallel(
        pdb=pdb_min,
        n_cpu=_resolve_cpu(N_CPU),
        positions=None,
        chain=None,
        save_structures=False,
        output_dir=out_dir,
        n_replicas=N_REPLICAS,
        neighborhood_radius=NEIGHBORHOOD_RADIUS,
        base_seed=BASE_SEED,
    )
    return os.path.join(out_dir, "DMS_report.csv")


def plot_heatmap(df: pd.DataFrame, output_path: str, title: str) -> None:
    metric = "ddG_total_score"
    aa_order = ["G", "A", "V", "L", "I", "P", "F", "W", "M", "S",
                "T", "C", "Y", "H", "D", "E", "N", "Q", "K", "R"]

    df = df.copy()
    df["pos_label"] = df["WT"] + df["Position_PDB"].astype(str)
    position_order = (
        df[["Position_PDB", "pos_label"]].drop_duplicates()
        .sort_values("Position_PDB")["pos_label"].tolist()
    )

    pivot = df.pivot_table(
        index="Mutation", columns="pos_label",
        values=metric, aggfunc="first",
    ).reindex(index=aa_order, columns=position_order)

    wt_map = (
        df[["pos_label", "WT"]].drop_duplicates()
        .set_index("pos_label")["WT"].to_dict()
    )
    wt_mask = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
    for pos, wt in wt_map.items():
        if pos in wt_mask.columns and wt in wt_mask.index:
            wt_mask.loc[wt, pos] = True

    w = max(10, len(position_order) * 0.32 + 3)
    h = max(6, len(aa_order) * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(w, h))
    vmax = df[metric].abs().quantile(0.97)

    sns.heatmap(
        pivot, ax=ax, cmap="RdBu_r", center=0.0,
        vmin=-vmax, vmax=vmax,
        linewidths=0.0,
        cbar_kws={"label": "ddG Total Score (REU)", "shrink": 0.6, "pad": 0.02},
        annot=False,
    )

    for row_i, aa in enumerate(pivot.index):
        for col_j, pos in enumerate(pivot.columns):
            if wt_mask.loc[aa, pos]:
                ax.plot(col_j + 0.5, row_i + 0.5,
                        marker="o", color="black", markersize=3, zorder=5)

    step = max(1, len(position_order) // 40)
    tick_idx = list(range(0, len(position_order), step))
    ax.set_xticks([i + 0.5 for i in tick_idx])
    ax.set_xticklabels([position_order[i] for i in tick_idx],
                       rotation=90, fontsize=10, ha="center")
    ax.set_xlabel("Position", fontsize=18, labelpad=10, fontweight="bold")
    ax.set_ylabel("Mutation", fontsize=18, labelpad=10, fontweight="bold")
    ax.set_title(f"{title} - Deep Mutational Scanning",
                 fontsize=20, pad=14, fontweight="bold")
    ax.tick_params(axis="y", rotation=0, labelsize=14)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("ddG Total Score (REU)", fontsize=14, fontweight="bold")

    wt_marker = mlines.Line2D([], [], color="black", marker="o", linestyle="None",
                              markersize=7, label="Wild-type")
    ax.legend(handles=[wt_marker], loc="upper left",
              bbox_to_anchor=(1.04, 1.0), frameon=False, fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Heatmap saved: %s", output_path)


def process_protein(name: str, plot: bool = True) -> str:
    """Run the full pipeline for one protein. Returns path to the DMS_report.csv."""
    info = PROTEINS[name]
    input_pdb = os.path.join(ROOT, "pdb", info["pdb"])
    work = os.path.join(ROOT, name)
    os.makedirs(work, exist_ok=True)

    pdb_clean   = os.path.join(work, f"{name}_clean.pdb")
    relax_dir   = os.path.join(work, "pre_relax")
    pdb_min     = os.path.join(relax_dir, "minimum_energy.pdb")
    dms_dir     = os.path.join(work, "dms_output")
    csv_path    = os.path.join(dms_dir, "DMS_report.csv")
    heatmap     = os.path.join(work, f"{name}_dms_heatmap.png")

    logging.info("=== %s ===", name)

    if not os.path.exists(pdb_clean):
        clean_pdb(input_pdb, pdb_clean)
    else:
        logging.info("Clean PDB exists, skipping: %s", pdb_clean)

    if not os.path.exists(pdb_min):
        pre_relax(pdb_clean, relax_dir)
    else:
        logging.info("Pre-relax exists, skipping: %s", pdb_min)

    if not os.path.exists(csv_path):
        run_dms(pdb_min, dms_dir)
    else:
        logging.info("DMS exists, skipping: %s", csv_path)

    if plot:
        df = pd.read_csv(csv_path)
        plot_heatmap(df, heatmap, info["title"])

    logging.info("=== Done: %s ===", name)
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="DMS pipeline for Klebsiella two-component system proteins."
    )
    parser.add_argument(
        "target",
        choices=list(PROTEINS.keys()) + ["all"],
        help="Protein to process (or 'all' for sequential run of every target).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip heatmap generation.",
    )
    args = parser.parse_args()

    if args.target == "all":
        for name in PROTEINS:
            process_protein(name, plot=not args.no_plot)
    else:
        process_protein(args.target, plot=not args.no_plot)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
