import os
import sys
from pathlib import Path

def count_json_files(folder_path: str) -> None:
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    json_files = [p for p in folder.iterdir() if p.is_file() and p.suffix == ".json"]
    print(f"Number of .json files in '{folder_path}': {len(json_files)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python 4.utils/count_jsons.py <folder_with_jsons>\n"
            "Example:\n"
            "  python 4.utils/count_jsons.py 1.data_access/data_sample/3.data_processed"
        )
        sys.exit(1)

    count_json_files(sys.argv[1])