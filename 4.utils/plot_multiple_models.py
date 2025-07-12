#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import csv
import math
import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt

#DEFAULT_CSVS = [
#    "5.results/4.5/BCData/ki67_results.csv",
#    "5.results/4.1-mini-2025-04-14/BCData/ki67_results.csv",
#    "5.results/4.1-2025-04-14/BCData/ki67_results.csv",
#    "5.results/4o/BCData/ki67_results.csv",
#    "5.results/gemini1.5pro/BCData/ki67_results.csv",
#    "5.results/gemini1.5flash/BCData/ki67_results.csv",
#    "5.results/grok2vision/BCData/ki67_results.csv",
#    "5.results/claude-3-5-sonnet/BCData/ki67_results.csv",
#]

DEFAULT_CSVS = [
    "5.results/4.5/SHIDC-B-Ki-67/ki67_results.csv",
    "5.results/4.1-mini-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv",
    "5.results/4.1-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv",
    "5.results/4o/SHIDC-B-Ki-67/ki67_results.csv",
    "5.results/gemini1.5pro/SHIDC-B-Ki-67/ki67_results.csv",
    "5.results/gemini1.5flash/SHIDC-B-Ki-67/ki67_results.csv",
    "5.results/grok2vision/SHIDC-B-Ki-67/ki67_results.csv",
    "5.results/claude-3-5-sonnet/SHIDC-B-Ki-67/ki67_results.csv",
]

def load_data(csv_path: Path):
    y_true, y_pred = [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                y_true.append(float(row["true"]))
                y_pred.append(float(row["predicted"]))
            except (KeyError, ValueError):
                continue
    return y_true, y_pred

def plot_models(csv_paths, rows: int | None, cols: int | None, out_path: Path) -> None:
    n = len(csv_paths)

    if rows is None and cols is None:
        cols = min(4, n)
        rows = math.ceil(n / cols)
    elif rows is None:                      
        rows = math.ceil(n / cols)
    elif cols is None:                      
        cols = math.ceil(n / rows)

    if rows * cols < n:
        sys.exit(f"[ERROR] grid {rows}×{cols} cannot fit {n} plots")

    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 6.5 * rows), constrained_layout=False)
    fig.subplots_adjust(
                        left   = 0.015,  
                        right  = 0.99,  
                        top    = 0.88,  
                        bottom = 0.20,  
                        wspace = 0.02,  
                        hspace = 0.02)  
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    letters = string.ascii_uppercase

    for idx, csv_file in enumerate(csv_paths):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        y_true, y_pred = load_data(Path(csv_file))

        ax.scatter(y_true, y_pred, marker="x")
        ax.plot([0, 100], [0, 100], color="gray", linewidth=1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Actual Ki-67 [%]", fontsize=42)
        if idx == 0:
            ax.set_ylabel("Predicted Ki-67 [%]", fontsize=42)
        else:
            ax.set_ylabel("")
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_xticks([0, 16, 30, 100])
        ax.set_yticks([0, 16, 30, 100])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=28)

        ax.set_title(letters[idx], loc="center", fontsize=50, fontweight="bold")

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf")
    print(f"[DONE] plot saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="plot_multiple_models",
        description="Grid scatter-plot comparison of multiple ki67_results.csv files.",
        add_help=False,
    )
    parser.add_argument("csvs", nargs="*", help="Paths to ki67_results.csv files")
    parser.add_argument("--rows", type=int, help="number of subplot rows")
    parser.add_argument("--cols", type=int, help="number of subplot columns")
    parser.add_argument("--out", default="5.results/ki67_comparison_plot.pdf",
                        help="output PDF path")
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS)

    args = parser.parse_args()

    csv_list = args.csvs or DEFAULT_CSVS

    if args.csvs == []:
        print(
            "Usage:\n"
            "  python 4.utils/plot_multiple_models.py <csv1> [<csv2> ...] "
            "[--rows N] [--cols M] [--out <output.pdf>]\n"
            "Example (BCData – eight runs in a 1×8 grid):\n\n"
            "BCData:\n"
            "python 4.utils/plot_multiple_models.py 5.results/4.5/BCData/ki67_results.csv 5.results/4.1-mini-2025-04-14/BCData/ki67_results.csv 5.results/4.1-2025-04-14/BCData/ki67_results.csv 5.results/4o/BCData/ki67_results.csv 5.results/gemini1.5pro/BCData/ki67_results.csv 5.results/gemini1.5flash/BCData/ki67_results.csv 5.results/grok2vision/BCData/ki67_results.csv 5.results/claude-3-5-sonnet/BCData/ki67_results.csv --rows 1 --cols 8 --out 5.results/ki67_comparison_plot_bcdata.pdf"
            "SHIDC-B-Ki-67:\n"
            "python 4.utils/plot_multiple_models.py 5.results/4.5/SHIDC-B-Ki-67/ki67_results.csv 5.results/4.1-mini-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv 5.results/4.1-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv 5.results/4o/SHIDC-B-Ki-67/ki67_results.csv 5.results/gemini1.5pro/SHIDC-B-Ki-67/ki67_results.csv 5.results/gemini1.5flash/SHIDC-B-Ki-67/ki67_results.csv 5.results/grok2vision/SHIDC-B-Ki-67/ki67_results.csv 5.results/claude-3-5-sonnet/SHIDC-B-Ki-67/ki67_results.csv --rows 1 --cols 8 --out 5.results/ki67_comparison_plot_shidc-b-ki-67.pdf"
            "\n\nNo CSV paths supplied — using the default list.\n"
        )

    plot_models(csv_list, args.rows, args.cols, Path(args.out))
