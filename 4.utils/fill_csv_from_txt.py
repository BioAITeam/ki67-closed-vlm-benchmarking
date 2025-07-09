import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

def read_existing_csv(csv_path: Path) -> Tuple[List[Dict[str, str]], Set[str]]:
    if not csv_path.is_file():
        return [], set()

    records, images = [], set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = (row.get("image") or row.get("imagen") or "").strip()
            records.append(row)
            if image:
                images.add(image)
    return records, images

def read_llm_txt(txt_path: Path) -> Dict[str, str]:
    blocks: Dict[str, str] = {}
    current, content = None, []
    re_image = re.compile(r"([^\s]+?\.(?:jpg|jpeg|png))", re.I)

    with txt_path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("====="):
                if current and content:
                    blocks[current] = "\n".join(content).strip()
                content = []
                m = re_image.search(line)
                current = m.group(1).strip() if m else None
            else:
                content.append(line.rstrip())

        if current and content:
            blocks[current] = "\n".join(content).strip()

    return blocks

_KI67_RE = re.compile(r"Ki[\s-]?67[^%]*?([0-9]+(?:\.[0-9]+)?)\s*%", flags=re.I | re.S)

def extract_index(text: str) -> Optional[float]:
    m = _KI67_RE.search(text)
    return float(m.group(1)) if m else None

def calculate_true_index(json_path: Path) -> float:
    with json_path.open() as f:
        data = json.load(f)
    pos = sum(1 for c in data if c.get("label_id") == 1)
    neg = sum(1 for c in data if c.get("label_id") == 2)
    total = pos + neg
    return round((pos / total) * 100, 2) if total else 0.0

def update_csv(csv_path: str, txt_path: str, json_folder: str) -> None:
    csv_path = Path(csv_path).resolve()
    txt_path = Path(txt_path).resolve()
    json_folder = Path(json_folder).resolve()

    if not csv_path.is_file():
        sys.exit(f"Error: CSV '{csv_path}' does not exist.")
    if not txt_path.is_file():
        sys.exit(f"Error: TXT '{txt_path}' does not exist.")
    if not json_folder.is_dir():
        sys.exit(f"Error: JSON folder '{json_folder}' does not exist.")

    output_csv = csv_path.parent / "ki67_results_filled.csv"

    existing_records, images_in_csv = read_existing_csv(csv_path)
    llm_responses = read_llm_txt(txt_path)

    new_rows: List[Dict[str, str]] = []

    for image, response_text in llm_responses.items():
        if image in images_in_csv:
            continue  # already registered

        predicted = extract_index(response_text)
        if predicted is None:
            print(f"Warning: Ki-67 not found in {image}; skipped.")
            continue

        json_path = json_folder / f"{Path(image).stem}.json"
        if not json_path.is_file():
            print(f"Warning: JSON not found for {image}; skipped.")
            continue

        actual = calculate_true_index(json_path)
        new_rows.append(
            {
                "image": image,
                "predicted": f"{predicted:.2f}",
                "true": f"{actual:.2f}",
            }
        )
        print(f"{image} → predicted: {predicted:.2f}% | true: {actual:.2f}%")

    fieldnames = ["image", "predicted", "true"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_records:
            writer.writerow(
                {
                    "image": row.get("image") or row.get("imagen"),
                    "predicted": row.get("predicted") or row.get("predicho"),
                    "true": row.get("true") or row.get("verdadero"),
                }
            )
        for row in new_rows:
            writer.writerow(row)

    print(f"\nUpdated CSV saved at: {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "  python 4.utils/fill_csv_from_txt.py <results.csv> <llm_responses.txt> <json_folder>\n"
            "Example:\n"
            "  python 4.utils/fill_csv_from_txt.py "
            "5.results/4.5/bcdata/ki67_results.csv "
            "5.results/4.5/bcdata/llm_responses.txt "
            "1.data_access/data_sample/3.data_processed"
        )
        sys.exit(1)

    csv_in, txt_in, json_dir = sys.argv[1:4]
    update_csv(csv_in, txt_in, json_dir) 