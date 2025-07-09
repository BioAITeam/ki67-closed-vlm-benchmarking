import os
from PIL import Image
import sys
from pathlib import Path

def convert_png_to_jpg(source_folder, target_folder):
    try:
        os.makedirs(target_folder, exist_ok=True)

        files = [f for f in os.listdir(source_folder) if f.lower().endswith('.png')]
        
        if not files:
            print("[INFO] No .png files found in the source folder.")
            return

        for file in files:
            png_path = os.path.join(source_folder, file)
            base_name = os.path.splitext(file)[0]
            jpg_path = os.path.join(target_folder, f"{base_name}.jpg")

            try:
                with Image.open(png_path) as img:
                    if img.mode in ("RGBA", "P"): 
                        img = img.convert("RGB")
                    img.save(jpg_path, "JPEG")
                print(f"[OK] Converted: {file} -> {jpg_path}")
            except Exception as e:
                print(f"[ERROR] Failed to convert {file}: {e}")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python 2.preprocess/1.convert_images.py <images_src> <processed_dataset>\n"
            "Example:\n"
            "  python 2.preprocess/1.convert_images.py "
            "1.data_access/data_sample/1.images/test "
            "1.data_access/data_sample/3.data_processed"
        )
        sys.exit(1)

    src = Path(sys.argv[1]).resolve()
    dst = Path(sys.argv[2]).resolve()
    if not src.is_dir():
        sys.exit(f"Source folder not found: {src}")

    convert_png_to_jpg(src, dst)