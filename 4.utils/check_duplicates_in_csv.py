import csv
import sys
from collections import Counter
from pathlib import Path

def detect_duplicates(csv_path: str) -> None:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        print(f"File not found: {csv_path}")
        return

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "image" not in reader.fieldnames:
            print("Column 'image' not found in the CSV file.")
            return

        images = [row["image"] for row in reader if row.get("image")]

    duplicates = {img: cnt for img, cnt in Counter(images).items() if cnt > 1}

    if duplicates:
        print("Duplicated images found:")
        for img, cnt in duplicates.items():
            print(f" - {img}: {cnt} times")
    else:
        print("No duplicated images found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python 4.utils/check_duplicates_in_csv.py <results.csv>\n"
            "Example:\n"
            "  python 4.utils/check_duplicates_in_csv.py "
            "5.results/4.5/bcdata/ki67_results.csv"
        )
        sys.exit(1)

    detect_duplicates(sys.argv[1])