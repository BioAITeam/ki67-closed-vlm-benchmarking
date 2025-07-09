#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_ki67_classification.py
--------------------------------
Evaluate one or more Ki-67 prediction CSV files against a shared ground-truth
Excel file.  Each image is binned into three ranges:

    • Low    : Ki-67 < 16 %
    • Medium : 16 % ≤ Ki-67 < 30 %
    • High   : Ki-67 ≥ 30 %

The script reports

    • Accuracy
    • Precision, recall, F1 (per-class and macro)
    • Confusion matrix
    • Class-distribution (GT vs predictions)
    • Comparative tables when ≥ 2 prediction files are provided
    • Optional multi-panel PDF with all confusion matrices
"""
from __future__ import annotations

import argparse
import math
import string
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ────────────────────────────────────────────────────────────────────────────
#                               EXAMPLE USAGE
# ────────────────────────────────────────────────────────────────────────────
USAGE_EXAMPLE = """
────────────────────────  EXAMPLE USAGE FOR BCData ─────────────────────────
python 4.utils/evaluate_ki67_classification.py \
 --gt 1.data_access/data_full_BCData_summary.xlsx \
 --pred \
  5.results/4.5/BCData/ki67_results.csv \
  5.results/4.1-mini-2025-04-14/BCData/ki67_results.csv \
  5.results/4.1-2025-04-14/BCData/ki67_results.csv \
  5.results/4o/BCData/ki67_results.csv \
  5.results/gemini1.5pro/BCData/ki67_results.csv \
  5.results/gemini1.5flash/BCData/ki67_results.csv \
  5.results/grok2vision/BCData/ki67_results.csv \
  5.results/claude-3-5-sonnet/BCData/ki67_results.csv \
 --labels "GPT-4.5,GPT-4.1 mini,GPT-4.1,GPT-4o,\
Gemini 1.5 Pro,Gemini 1.5 Flash,Grok-2 Vision,Claude 3.5 Sonnet" \
 --low 16 --mid 30 \
 --cm-out 5.results/conf_matrix_grid_bcdata.pdf \
 --cm-rows 1 --cm-cols 8
\n\n─────────────────────────────────────────────────────────────────────────
────────────────────────  EXAMPLE USAGE FOR SHIDC-B-Ki-67 ───────────────────
python 4.utils/evaluate_ki67_classification.py \
 --gt 1.data_access/data_full_SHIDC-B-Ki-67_summary.xlsx \
 --pred \
  5.results/4.5/SHIDC-B-Ki-67/ki67_results.csv \
  5.results/4.1-mini-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv \
  5.results/4.1-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv \
  5.results/4o/SHIDC-B-Ki-67/ki67_results.csv \
  5.results/gemini1.5pro/SHIDC-B-Ki-67/ki67_results.csv \
  5.results/gemini1.5flash/SHIDC-B-Ki-67/ki67_results.csv \
  5.results/grok2vision/SHIDC-B-Ki-67/ki67_results.csv \
  5.results/claude-3-5-sonnet/SHIDC-B-Ki-67/ki67_results.csv \
 --labels "GPT-4.5,GPT-4.1 mini,GPT-4.1,GPT-4o,\
Gemini 1.5 Pro,Gemini 1.5 Flash,Grok-2 Vision,Claude 3.5 Sonnet" \
 --low 16 --mid 30 \
 --cm-out 5.results/conf_matrix_grid_shidc_b_ki67.pdf \
 --cm-rows 1 --cm-cols 8
─────────────────────────────────────────────────────────────────────────────\n\n
"""

# ────────────────────────────────────────────────────────────────────────────
#                                   HELPERS
# ────────────────────────────────────────────────────────────────────────────
def ki67_to_class(val: float, low: float, mid: float) -> int:
    """Return 0/1/2 according to cut-offs."""
    if val < low:
        return 0                          # low
    if val < mid:                         # 16 ≤ val < 30
        return 1                          # medium
    return 2                              # high  (≥ 30)

def read_ground_truth(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERROR] ground-truth file not found → {path}")
    df = pd.read_excel(path, engine="openpyxl")
    if "image" not in df.columns:
        df = df.rename(columns={"image_name": "image"})
    df["ki67_value_0_100"] = (
        df["ki67_value_0_100"].astype(str).str.replace(",", ".").astype(float)
    )
    return df[["image", "ki67_value_0_100"]]

def read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERROR] prediction file not found → {path}")
    df = pd.read_csv(path)
    if "image" not in df.columns:
        df = df.rename(columns={"image_name": "image"})
    df["predicted"] = df["predicted"].astype(str).str.replace(",", ".").astype(float)
    return df[["image", "predicted"]]

# ────────────────────────────────────────────────────────────────────────────
#                                 METRICS
# ────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    y_true_cls: pd.Series, y_pred_cls: pd.Series, labels: List[int] = [0, 1, 2]
) -> Dict[str, float | List | pd.DataFrame]:
    acc = accuracy_score(y_true_cls, y_pred_cls)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, labels=labels, zero_division=0
    )
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)
    return dict(
        accuracy=acc,
        precision=p,
        recall=r,
        f1=f1,
        macro_precision=mp,
        macro_recall=mr,
        macro_f1=mf1,
        confusion=cm,
    )

def evaluate_pair(
    gt_path: Path, pred_path: Path, low_thr: float, mid_thr: float
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    gt   = read_ground_truth(gt_path)
    pred = read_predictions(pred_path)
    merged = pd.merge(gt, pred, on="image", how="inner")
    if merged.empty:
        sys.exit(f"[ERROR] no matching image names between GT and {pred_path}")
    y_true_cls = merged["ki67_value_0_100"].apply(ki67_to_class, args=(low_thr, mid_thr))
    y_pred_cls = merged["predicted"].apply(ki67_to_class, args=(low_thr, mid_thr))
    return y_true_cls, y_pred_cls, compute_metrics(y_true_cls, y_pred_cls)

# ────────────────────────────────────────────────────────────────────────────
#                       CONFUSION-MATRIX GRID (PDF)
# ────────────────────────────────────────────────────────────────────────────
def save_conf_matrix_grid(
    names: List[str],
    metrics_list: List[Dict[str, float]],
    out_path: Path,
    rows: int | None,
    cols: int | None,
) -> None:
    n = len(names)
    if rows is None and cols is None:
        cols = min(4, n)
        rows = math.ceil(n / cols)
    elif rows is None:
        rows = math.ceil(n / cols)
    elif cols is None:
        cols = math.ceil(n / rows)
    if rows * cols < n:
        sys.exit(f"[ERROR] grid {rows}×{cols} cannot fit {n} matrices")

    # tighter horizontal spacing
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(3.4 * cols, 3.6 * rows),
        gridspec_kw={"wspace": 0.05, "hspace": 0.15}
    )
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    tick_labels = ["low", "medium", "high"]
    for idx, m in enumerate(metrics_list):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        cm = m["confusion"]
        vmax = cm.max()

        # grayscale background
        ax.imshow(cm, cmap="Greys", interpolation="nearest",
                  vmin=0, vmax=vmax)

        ax.set_xticks(range(3), tick_labels, fontsize=14, fontweight="bold")
        if c == 0:                                           # show y-ticks only in first col
            ax.set_yticks(range(3), tick_labels, fontsize=14, fontweight="bold")
            ax.set_ylabel("Actual", fontsize=14, fontweight="bold")
        else:
            ax.set_yticks([])

        ax.set_xlabel("Predicted", fontsize=14, fontweight="bold")

        # panel letter
        ax.set_title(string.ascii_uppercase[idx],
                     fontsize=20, fontweight="bold", pad=6)

        # annotate counts with adaptive colour
        for i in range(3):
            for j in range(3):
                val = int(cm[i, j])
                color = "white" if val > 0.5 * vmax else "black"
                ax.text(j, i, val, ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)

    # hide unused panels
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"[INFO] confusion-matrix grid saved → {out_path.resolve()}\n")

# ────────────────────────────────────────────────────────────────────────────
#                                PRINTING
# ────────────────────────────────────────────────────────────────────────────
def print_single_report(name: str, low: float, mid: float,
                        m: Dict[str, float | List | pd.DataFrame]) -> None:
    classes = ["low", "medium", "high"]
    print(f"\n====================  {name}  ====================")
    print(f"Cut-offs: low < {low} , {low} ≤ medium < {mid} , high ≥ {mid}")
    print(f"Accuracy               : {m['accuracy']:.4f}")
    print(f"Macro Precision        : {m['macro_precision']:.4f}")
    print(f"Macro Recall           : {m['macro_recall']:.4f}")
    print(f"Macro F1-score         : {m['macro_f1']:.4f}\n")
    print("Per-class metrics")
    for i, cls in enumerate(classes):
        print(f"{cls:>6}  P={m['precision'][i]:.4f}  "
              f"R={m['recall'][i]:.4f}  F1={m['f1'][i]:.4f}")

def print_perclass_table(names: List[str], mlist: List[Dict[str, float]]) -> None:
    labels = ["low", "medium", "high"]
    rows = {
        f"{cls.capitalize()}_{metric.capitalize()}": [m[metric][i] for m in mlist]
        for metric in ("precision", "recall", "f1")
        for i, cls in enumerate(labels)
    }
    print("\n====================  PER-CLASS METRICS  ====================")
    print(pd.DataFrame(rows, index=names).T.round(4).to_string())

def print_macro_table(names: List[str], mlist: List[Dict[str, float]]) -> None:
    df = pd.DataFrame({
        "Accuracy":        [m["accuracy"]        for m in mlist],
        "Macro_Precision": [m["macro_precision"] for m in mlist],
        "Macro_Recall":    [m["macro_recall"]    for m in mlist],
        "Macro_F1":        [m["macro_f1"]        for m in mlist],
    }, index=names).T
    print("\n====================  MACRO METRICS  ====================")
    print(df.round(4).to_string())

# ────────────────────────────────────────────────────────────────────────────
#                                    CLI
# ────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_ki67_classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__ + "\n" + USAGE_EXAMPLE,
    )
    p.add_argument("--gt",   required=True, type=Path, help="Ground-truth Excel file")
    p.add_argument("--pred", required=True, nargs="+", type=Path,
                   help="One or more ki67_results.csv prediction files")
    p.add_argument("--labels", type=str,
                   help="Comma-separated model names (same order as --pred)")
    p.add_argument("--low", type=float, default=16.0,
                   help="Low / medium cut-off (default 16)")
    p.add_argument("--mid", type=float, default=30.0,
                   help="Medium / high cut-off (default 30)")
    p.add_argument("--cm-out",  type=Path,
                   help="Path for confusion-matrix PDF")
    p.add_argument("--cm-rows", type=int,
                   help="Rows in confusion-matrix grid")
    p.add_argument("--cm-cols", type=int,
                   help="Columns in confusion-matrix grid")
    return p

def validate_args(args: argparse.Namespace) -> None:
    if args.low <= 0 or args.mid <= 0:
        sys.exit("[ERROR] --low and --mid must be > 0\n" + USAGE_EXAMPLE)
    if args.low >= args.mid:
        sys.exit("[ERROR] require low < mid (e.g. 16 30)\n" + USAGE_EXAMPLE)
    if args.labels and len([x for x in args.labels.split(",") if x.strip()]) != len(args.pred):
        sys.exit("[ERROR] number of --labels does not match --pred files\n"
                 + USAGE_EXAMPLE)

# ────────────────────────────────────────────────────────────────────────────
#                                   MAIN
# ────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    parser = build_parser()
    if argv is None and len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.stderr.write(USAGE_EXAMPLE + "\n")
        sys.exit(1)

    args = parser.parse_args(argv)
    validate_args(args)

    n_files = len(args.pred)
    names = (
        [x.strip() for x in args.labels.split(",") if x.strip()]
        if args.labels
        else (list(string.ascii_uppercase[:n_files])
              if n_files <= 26 else [p.stem for p in args.pred])
    )

    metrics_list: List[Dict[str, float]] = []
    for name, pred_path in zip(names, args.pred):
        _, _, metrics = evaluate_pair(args.gt, pred_path, args.low, args.mid)
        metrics_list.append(metrics)
        print_single_report(name, args.low, args.mid, metrics)

    if n_files > 1:
        print_perclass_table(names, metrics_list)
        print_macro_table(names, metrics_list)

    if args.cm_out:
        save_conf_matrix_grid(
            names, metrics_list, args.cm_out,
            rows=args.cm_rows, cols=args.cm_cols,
        )

if __name__ == "__main__":
    main()
