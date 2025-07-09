#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
from pathlib import Path
from collections import Counter

_IMAGE_RE = re.compile(r"image\s*:\s*([^,]+\.jpg)", re.I)

def read_images_from_txt(txt_path: Path) -> list[str]:
    imgs: list[str] = []
    with txt_path.open(encoding="utf-8") as f:
        for ln in f:
            m = _IMAGE_RE.search(ln)
            if m:
                imgs.append(m.group(1).strip().lower())
    return imgs

def find_duplicates(imgs: list[str]) -> list[str]:
    counts = Counter(imgs)
    return sorted([img for img, n in counts.items() if n > 1])

def main(txt_file: str) -> None:
    path = Path(txt_file).resolve()
    if not path.is_file():
        sys.exit(f"[ERROR] File not found: {path}")

    images = read_images_from_txt(path)
    dups   = find_duplicates(images)

    print(f"\nAnalysed file : {path}")
    print(f"Total lines with images : {len(images)}")
    print(f"Unique image filenames   : {len(set(images))}")

    if dups:
        print("\n  Duplicates detected:")
        for img in dups:
            print(f" - {img}")
    else:
        print("\n No duplicate images found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python 4.utils/check_duplicates_in_txt.py <ki67_log.txt>\n\n"
            "Example:\n"
            "  python 4.utils/check_duplicates_in_txt.py "
            "5.results/4.5/BCData/ki67_log.txt"
        )
        sys.exit(1)

    main(sys.argv[1])
