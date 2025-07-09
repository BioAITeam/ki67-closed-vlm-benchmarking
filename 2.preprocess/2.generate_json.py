import os
import json
import h5py
import sys

def extract_coordinates_from_h5(file_path, label_id):
    coordinates = []
    try:
        with h5py.File(file_path, 'r') as f:
            if 'coordinates' in f:
                coords = f['coordinates'][:]
                for x, y in coords:
                    coordinates.append({
                        'x': int(x),
                        'y': int(y),
                        'label_id': label_id
                    })
            else:
                print(f"[WARNING] File '{file_path}' does not contain the 'coordinates' dataset.")
    except Exception as e:
        print(f"[ERROR] Could not read '{file_path}': {e}")
    return coordinates

def process_folders(positive_dir, negative_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    positive_files = {f for f in os.listdir(positive_dir) if f.endswith('.h5')}
    negative_files = {f for f in os.listdir(negative_dir) if f.endswith('.h5')}

    common_files = positive_files & negative_files

    if not common_files:
        print("[INFO] No matching .h5 files found in both folders.")
        return

    only_in_positive = positive_files - negative_files
    only_in_negative = negative_files - positive_files

    if only_in_positive:
        print(f"[WARNING] Files only in POSITIVE: {', '.join(sorted(only_in_positive))}")
    if only_in_negative:
        print(f"[WARNING] Files only in NEGATIVE: {', '.join(sorted(only_in_negative))}")

    for name in sorted(common_files):
        pos_path = os.path.join(positive_dir, name)
        neg_path = os.path.join(negative_dir, name)

        data = []
        data += extract_coordinates_from_h5(pos_path, label_id=1)
        data += extract_coordinates_from_h5(neg_path, label_id=2)

        output_name = os.path.splitext(name)[0]
        output_path = os.path.join(output_dir, f"{output_name}.json")
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"[OK] JSON created: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write JSON '{output_path}': {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "  python 2.preprocess/2.generate_json.py <positive_dir> <negative_dir> <processed_dataset>\n"
            "Example:\n"
            "  python 2.preprocess/2.generate_json.py "
            "1.data_access/data_sample/2.annotations/test/positive "
            "1.data_access/data_sample/2.annotations/test/negative "
            "1.data_access/data_sample/3.data_processed"
        )
        sys.exit(1)

    positive_dir = sys.argv[1]
    negative_dir = sys.argv[2]
    output_dir = sys.argv[3]

    print("[START] Processing HDF5 files...")
    process_folders(positive_dir, negative_dir, output_dir)
    print("[DONE] Processing completed.")
