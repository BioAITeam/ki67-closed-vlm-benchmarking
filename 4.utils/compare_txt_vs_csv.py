import csv
import re
import sys
from pathlib import Path

def read_images_from_csv(csv_path: Path) -> set[str]:
    images = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None) 
        for row in reader:
            if row and row[0].strip():
                images.add(row[0].strip())
    return images

def read_images_from_txt(txt_path: Path) -> set[str]:
    images = set()
    pattern = re.compile(r"^=+\s*(.+?\.(?:jpg|jpeg|png))\s*=+$", re.I)
    with txt_path.open(encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                images.add(m.group(1).strip())
    return images

def compare(csv_path: str, txt_path: str) -> None:
    csv_p = Path(csv_path).resolve()
    txt_p = Path(txt_path).resolve()

    if not csv_p.is_file():
        print(f"CSV file not found: {csv_p}")
        return
    if not txt_p.is_file():
        print(f"TXT file not found: {txt_p}")
        return

    csv_images = read_images_from_csv(csv_p)
    txt_images = read_images_from_txt(txt_p)

    missing = txt_images - csv_images
    if missing:
        print("\nImages found in TXT but missing in CSV:")
        for img in sorted(missing):
            print(f" - {img}")
    else:
        print("All images referenced in TXT are present in CSV.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python 4.utils/compare_txt_vs_csv.py <results.csv> <llm_responses.txt>\n"
            "Example:\n"
            "  python 4.utils/compare_txt_vs_csv.py "
            "5.results/4.5/bcdata/ki67_results.csv "
            "5.results/4.5/bcdata/llm_responses.txt"
        )
        sys.exit(1)

    compare(sys.argv[1], sys.argv[2])
