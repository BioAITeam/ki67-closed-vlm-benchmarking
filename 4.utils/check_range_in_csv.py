import csv
import sys

def read_images_from_csv(csv_path):
    images = set()
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)   
        for row in reader:
            if row and row[0].strip().lower().endswith(".jpg"):
                images.add(row[0].strip().lower())
    return images

def check_missing_range(csv_path, start, end):
    images = read_images_from_csv(csv_path)
    missing = []

    for i in range(start, end + 1):
        filename = f"{i}.jpg"
        if filename not in images:
            missing.append(filename)

    print(f"\n🔍 Checking range from {start}.jpg to {end}.jpg")
    if missing:
        print("Missing images in CSV:")
        for img in missing:
            print(f" - {img}")
    else:
        print("All images in range are present in the CSV.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "  python 4.utils/check_range_in_csv.py <results.csv> <start_id> <end_id>\n"
            "Example:\n"
            "  python 4.utils/check_range_in_csv.py "
            "5.results/4.5/bcdata/ki67_results.csv 0 25"
        )
        sys.exit(1)

    check_missing_range(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))