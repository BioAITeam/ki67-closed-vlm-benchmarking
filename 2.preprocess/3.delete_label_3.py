#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Delete entries whose ``label_id`` equals 3 from each *.json* file
and copy both the cleaned *.json* and its matching *.jpg* image
into a destination folder.
"""

import json
import shutil
import sys
from pathlib import Path


def clean_json(src_json: Path, dst_json: Path) -> None:
    """Remove label_id == 3 entries and save to *dst_json*."""
    with src_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = [item for item in data if item.get("label_id") != 3]

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with dst_json.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=4)

    #print(f"[OK] Cleaned JSON: {src_json.name} -> {dst_json}")


def copy_image(src_json: Path, dst_dir: Path) -> None:
    """Copy the jpg image that shares the stem with *src_json*."""
    src_img = src_json.with_suffix(".jpg")
    if not src_img.exists():
        print(f"[WARN] Image not found for {src_json.stem}.json (expected {src_img.name})")
        return

    dst_img = dst_dir / src_img.name
    shutil.copy2(src_img, dst_img)
    #print(f"[OK] Copied Image: {src_img.name} -> {dst_img}")


def process_folder(src_dir: Path, dst_dir: Path) -> None:
    json_files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".json")
    if not json_files:
        print("[INFO] No .json files found in the source folder.")
        return

    for src_json in json_files:
        dst_json = dst_dir / src_json.name
        clean_json(src_json, dst_json)
        copy_image(src_json, dst_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python 2.preprocess/3.delete_label_3.py <annotations_src> <processed_dataset>\n"
            "Example:\n"
            "  python 2.preprocess/3.delete_label_3.py "
            "1.data_access/data_sample_SHIDC-B-Ki-67/1.bare_images/Test "
            "1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed"
        )
        sys.exit(1)

    annotations_src = Path(sys.argv[1]).resolve()
    processed_dst = Path(sys.argv[2]).resolve()

    if not annotations_src.is_dir():
        sys.exit(f"[ERROR] Source folder not found: {annotations_src}")

    print("[START] Removing label_id == 3 and copying images…")
    process_folder(annotations_src, processed_dst)
    print("[DONE] Processing completed.")
