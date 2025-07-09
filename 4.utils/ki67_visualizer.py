#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# definir el porcentaje de ancho de imagen para el tamaño del punto
DOT_SIZE_PCT = 0.0085  # 0.85% del ancho de la imagen

COLORS = {1: (255, 0, 0), 2: (0, 255, 0)}
ARROW = dict(
    arrowstyle='-|>',
    connectionstyle="arc3",
    mutation_scale=25,
    linewidth=3,
    color='gray'
)

USAGE_EXAMPLE = (
    "Usage: python 4.utils/ki67_visualizer.py "
    "-i 1.data_access/data_full_BCData/3.data_processed/343.jpg "  #  4.5 %
    "-j 1.data_access/data_full_BCData/3.data_processed/343.json " #  4.5 %
    "-i 1.data_access/data_full_BCData/3.data_processed/281.jpg "  # 21.37 %
    "-j 1.data_access/data_full_BCData/3.data_processed/281.json " # 21.37 %
    "-i 1.data_access/data_full_BCData/3.data_processed/104.jpg "  # 91.67 %
    "-j 1.data_access/data_full_BCData/3.data_processed/104.json " # 91.67 %

    "-i 1.data_access/data_full_SHIDC-B-Ki-67/2.data_processed/p2_0219_4.jpg "   #  7.24 %
    "-j 1.data_access/data_full_SHIDC-B-Ki-67/2.data_processed/p2_0219_4.json "  #  7.24 %
    "-i 1.data_access/data_full_SHIDC-B-Ki-67/2.data_processed/p13_0093_6.jpg "  # 21.74 %
    "-j 1.data_access/data_full_SHIDC-B-Ki-67/2.data_processed/p13_0093_6.json " # 21.74 %
    "-i 1.data_access/data_full_SHIDC-B-Ki-67/2.data_processed/p15_0038_3.jpg "  # 67.05 %
    "-j 1.data_access/data_full_SHIDC-B-Ki-67/2.data_processed/p15_0038_3.json " # 67.05 %

    "--out 5.results/ki67_comparison_plots_values.pdf"
)

def load_annotations(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_image(path: str | Path):
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"[ERROR] cannot open image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def annotate(img, cells):
    """
    Dibuja un círculo en cada coordenada de célula
    con radio proporcional al ancho de la imagen.
    """
    out = img.copy()
    h, w, _ = out.shape
    # calcular radio en px según porcentaje del ancho
    radius = max(1, int(w * DOT_SIZE_PCT))

    for c in cells:
        lbl, x, y = c.get("label_id"), c.get("x"), c.get("y")
        if lbl in (1, 2) and x is not None and y is not None:
            cv2.circle(out, (int(x), int(y)), radius, COLORS[lbl], -1)
    return out

def stats(cells):
    pos = sum(1 for c in cells if c.get("label_id") == 1)
    neg = sum(1 for c in cells if c.get("label_id") == 2)
    tot = pos + neg
    return pos, neg, round(pos / tot * 100, 2) if tot else 0.0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ki67_visualizer",
        description="Generate a PDF report with original and Ki-67-annotated images",
    )
    p.add_argument("-i", "--image", action="append", metavar="IMG",
                   help="path to input image")
    p.add_argument("-j", "--json", action="append", metavar="JSON",
                   help="path to JSON annotations")
    p.add_argument("--out", default="ki67_report.pdf", metavar="PDF",
                   help="output PDF file")
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.image or not args.json or len(args.image) != len(args.json):
        print(USAGE_EXAMPLE)
        sys.exit(1)

    cases = list(zip(string.ascii_uppercase, args.image, args.json))

    print("\nCase summary")
    for letter, img_p, js_p in cases:
        pos, neg, idx = stats(load_annotations(js_p))
        print(f"{letter}: {pos:4d} positive | {neg:4d} negative | Ki-67 = {idx:6.2f} %")

    cols = len(cases)
    fig = plt.figure(figsize=(3.2 * cols, 8))
    gs = GridSpec(
            2, cols,
            hspace=0.28,
            wspace=0.05,
            left=0.02,
            right=0.98
    )

    for i, (letter, img_p, js_p) in enumerate(cases):
        img = load_image(img_p)
        ann = annotate(img, load_annotations(js_p))

        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(img);  ax1.axis("off")
        ax1.set_title(letter, fontsize=22, fontweight="bold", pad=14)

        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(ann); ax2.axis("off")

        p1, p2 = ax1.get_position(), ax2.get_position()
        x = p1.x0 + p1.width / 2
        y1, y2 = p1.y0 - .015, p2.y0 + p2.height + .015
        fig.patches.append(
            patches.FancyArrowPatch((x, y1), (x, y2),
                                    transform=fig.transFigure,
                                    clip_on=False,
                                    **ARROW)
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"\n[OK] PDF saved → {out.resolve()}\n")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(USAGE_EXAMPLE)
        sys.exit(1)
    main()
