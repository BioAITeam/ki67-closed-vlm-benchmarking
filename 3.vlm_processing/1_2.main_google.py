#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re, csv, json, time, argparse, sys
from datetime import datetime
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY not found.")
genai.configure(api_key=api_key)

# model_name = "gemini-1.5-flash"
model_name = "gemini-1.5-pro"

THIS_DIR = Path(__file__).parent
SYSTEM_PROMPT = (THIS_DIR / "system_prompt.txt").read_text(encoding="utf-8")
USER_PROMPT   = (THIS_DIR / "user_prompt.txt").read_text(encoding="utf-8")

_KI67 = re.compile(r"Ki[\s-]?67[^%]*?([0-9]+(?:\.[0-9]+)?)\s*%", re.I | re.S)

def extract_predicted_index(text: str) -> float:
    m = _KI67.search(text)
    if m:
        return float(m.group(1))
    vals = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if vals:
        return float(vals[-1])
    raise ValueError("Ki-67 value not found.")

def calculate_true_index(json_path: Path) -> float:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    pos = sum(1 for c in data if c.get("label_id") == 1)
    neg = sum(1 for c in data if c.get("label_id") == 2)
    return round((pos / (pos + neg)) * 100, 2) if pos + neg else 0.0

def predict_with_gemini(img: Path) -> Tuple[float, str]:
    image_bytes = img.read_bytes()
    model = genai.GenerativeModel(
        f"models/{model_name}",
        safety_settings=[
            {"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_NONE"},
            {"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"},
            {"category":"HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold":"BLOCK_NONE"},
            {"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_ONLY_HIGH"},
        ])
    rsp = model.generate_content(
        [SYSTEM_PROMPT, {"mime_type":"image/jpeg","data":image_bytes}, USER_PROMPT],
        generation_config=genai.types.GenerationConfig(temperature=0.0)
    )
    txt = rsp.text or ""
    return extract_predicted_index(txt), txt

def main(dataset: Path,
         parent_out: Path | None = None,
         resume_dir: Path | None = None,
         pause_sec: float = 0.0) -> None:

    safe_model = model_name.replace("/", "_")
    if resume_dir:
        out_dir = resume_dir.resolve()
    else:
        ts = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        base = parent_out.resolve() if parent_out else THIS_DIR
        out_dir = base / f"output_google_{safe_model}_date_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

    log_path  = out_dir / "ki67_log.txt"
    csv_path  = out_dir / "ki67_results.csv"
    llm_path  = out_dir / "llm_responses.txt"
    plot_path = out_dir / "ki67_pred_vs_true.png"

    processed: set[str] = set()
    if csv_path.exists():
        with csv_path.open() as f:
            next(f, None)
            processed = {row.split(",")[0].strip().lower() for row in f if row.strip()}

    images = sorted(p for p in dataset.iterdir()
                    if p.suffix.lower() in {".jpg",".jpeg",".png"})
    pending = [p for p in images if p.name.lower() not in processed]

    with log_path.open("a", encoding="utf-8") as logf, \
         csv_path.open("a", newline="", encoding="utf-8") as csvf:

        writer = csv.writer(csvf)
        if csv_path.stat().st_size == 0:
            writer.writerow(["image","predicted","true"]); csvf.flush()

        trues, preds = [], []

        for idx, img in enumerate(pending, 1):
            fname = img.name
            json_path = dataset / f"{img.stem}.json"
            if not json_path.is_file():
                print(f"[WARN] {fname}: JSON missing – skipped")
                continue

            try:
                true_val = calculate_true_index(json_path)
                pred_val, full_resp = predict_with_gemini(img)

                with llm_path.open("a", encoding="utf-8") as rf:
                    rf.write(f"\n===== {fname} =====\n{full_resp.strip()}\n")

                # write & flush every call
                logf.write(f"image:{fname}, predicted:{pred_val:.2f}, true:{true_val:.2f}\n")
                logf.flush()
                writer.writerow([fname,f"{pred_val:.2f}",f"{true_val:.2f}"])
                csvf.flush()

                trues.append(true_val); preds.append(pred_val)
                print(f"[OK] {idx}/{len(pending)} {fname}")

            except Exception as e:
                print(f"[ERROR] {fname}: {e}")

            if pause_sec:
                time.sleep(pause_sec)

    if trues and preds:
        plt.figure(figsize=(6,6))
        plt.scatter(trues, preds, marker="x")
        plt.plot([0,100],[0,100], linewidth=1, linestyle="--")
        plt.xlabel("True Ki-67 (%)"); plt.ylabel("Predicted Ki-67 (%)")
        plt.title("Ki-67 Predicted vs True")
        plt.tight_layout(); plt.savefig(plot_path, dpi=300); plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python 3.vlm_processing/1_2.main_google.py "
            "<processed_dataset> [<output_parent_dir>] "
            "[--resume <existing_out_dir>] [--pause <seconds>]\n\n"
            "Examples:\n"
            "  # fresh run (creates new folder)\n"
            "  python 3.vlm_processing/1_2.main_google.py "
            "1.data_access/data_sample/3.data_processed 5.results\n\n"
            "  # resume, continue only missing images\n"
            "  python 3.vlm_processing/1_2.main_google.py "
            "1.data_access/data_sample/3.data_processed "
            "--resume 5.results/gemini\n\n"
            "  # resume with 1-second pause\n"
            "  python 3.vlm_processing/1_2.main_google.py "
            "1.data_access/data_sample/3.data_processed "
            "--resume 5.results/gemini --pause 1.0"
        )
        sys.exit(1)

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("dataset")
    ap.add_argument("output_parent", nargs="?", default=None)
    ap.add_argument("--resume")
    ap.add_argument("--pause", type=float, default=0.0)
    args = ap.parse_args()

    ds = Path(args.dataset).resolve()
    if not ds.is_dir():
        sys.exit(f"[ERROR] Dataset folder not found: {ds}")

    resume_dir = Path(args.resume).resolve() if args.resume else None
    if resume_dir and not resume_dir.is_dir():
        sys.exit(f"[ERROR] Resume directory not found: {resume_dir}")

    parent_dir = Path(args.output_parent).resolve() if args.output_parent else None
    if resume_dir and args.output_parent:
        print("[INFO] --resume given → ignoring <output_parent_dir>")

    main(ds, parent_dir, resume_dir, args.pause)
