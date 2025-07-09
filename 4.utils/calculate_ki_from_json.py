import json
import sys

def calculate_ki_from_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        positives = sum(1 for item in data if item.get("label_id") == 1)
        negatives = sum(1 for item in data if item.get("label_id") == 2)
        total = positives + negatives

        if total == 0:
            ki67_index = 0.0
        else:
            ki67_index = round((positives / total) * 100, 2)

        print(f"Immunopositive cells: {positives}")
        print(f"Immunonegative cells: {negatives}")
        print(f"Ki-67 Index: {ki67_index:.2f}%")

    except FileNotFoundError:
        print(f"File not found: {json_path}")
    except json.JSONDecodeError:
        print("Error in JSON decoding.")
    except Exception as e:
        print(f"Unknown error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python 4.utils/calculate_ki_from_json.py <json_path>\n"
            "Example:\n"
            "  python 4.utils/calculate_ki_from_json.py "
            "1.data_access/data_sample/3.data_processed/8.json"
        )
        sys.exit(1)

    calculate_ki_from_json(sys.argv[1])
