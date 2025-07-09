import sys
from pathlib import Path
import pandas as pd

def verify_images_in_csv(image_folder: str, csv_file: str) -> None:
    image_dir = Path(image_folder).resolve()
    csv_path = Path(csv_file).resolve()

    if not image_dir.is_dir():
        print(f"Directory not found: {image_dir}")
        return
    if not csv_path.is_file():
        print(f"CSV file not found: {csv_path}")
        return

    extensions = {".jpg", ".jpeg", ".png"}
    images_in_folder = {p.name for p in image_dir.iterdir() if p.suffix.lower() in extensions}

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Accept either 'image' or 'imagen' column
    col = "image" if "image" in df.columns else ("imagen" if "imagen" in df.columns else None)
    if col is None:
        print("The CSV does not contain a column named 'image' or 'imagen'.")
        return

    images_in_csv = set(df[col].dropna().astype(str))
    missing = images_in_folder - images_in_csv

    if missing:
        print("Images present in the folder but missing from the CSV:")
        for img in sorted(missing):
            print(f"- {img}")
    else:
        print("All images in the folder are listed in the CSV.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python 4.utils/verify_images_in_csv.py <image_folder> <results.csv>\n"
            "Example:\n"
            "  python 4.utils/verify_images_in_csv.py "
            "1.data_access/data_sample/3.data_processed "
            "5.results/4.5/bcdata/ki67_results.csv"
        )
        sys.exit(1)

    verify_images_in_csv(sys.argv[1], sys.argv[2])