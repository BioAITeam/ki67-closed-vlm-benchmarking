#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re, csv, json, time, base64, argparse, sys
from datetime import datetime
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
if XAI_API_KEY is None:
    raise ValueError("XAI_API_KEY not found.")
API_URL   = "https://api.x.ai/v1/chat/completions"
MODEL_NAME = "grok-2-vision-latest"

THIS_DIR    = Path(__file__).parent
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

def _encode_image(img: Path) -> Tuple[str,str]:
    mime = "jpeg" if img.suffix.lower() in {".jpg",".jpeg"} else "png"
    return mime, base64.b64encode(img.read_bytes()).decode()

def predict_with_grok(img: Path) -> Tuple[float,str]:
    mime, b64 = _encode_image(img)
    payload = {
        "model": MODEL_NAME,
        "temperature": 0,
        "max_tokens": 1024,
        "messages":[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":[
                {"type":"text","text":USER_PROMPT},
                {"type":"image_url",
                 "image_url":{"url":f"data:image/{mime};base64,{b64}"}}]},
        ],
    }
    headers = {"Authorization":f"Bearer {XAI_API_KEY}",
               "Content-Type":"application/json"}
    r = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    if not r.ok:
        raise RuntimeError(f"xAI error {r.status_code}: {r.text}")
    txt = r.json()["choices"][0]["message"]["content"]
    return extract_predicted_index(txt), txt

def main(dataset: Path,
         parent_out: Path|None = None,
         resume_dir: Path|None = None,
         pause_sec: float = 0.0) -> None:

    safe_model = MODEL_NAME.replace("/","_")
    if resume_dir:
        out_dir = resume_dir.resolve()
    else:
        ts   = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        base = parent_out.resolve() if parent_out else THIS_DIR
        out_dir = base / f"output_xai_{safe_model}_date_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

    log_path  = out_dir / "ki67_log.txt"
    csv_path  = out_dir / "ki67_results.csv"
    llm_path  = out_dir / "llm_responses.txt"
    plot_path = out_dir / "ki67_pred_vs_true.png"

    processed:set[str] = set()
    if csv_path.exists():
        with csv_path.open() as f:
            next(f,None)
            processed = {row.split(",")[0].strip().lower() for row in f if row.strip()}

    images  = sorted(p for p in dataset.iterdir()
                     if p.suffix.lower() in {".jpg",".jpeg",".png"})
    pending = [p for p in images if p.name.lower() not in processed]

    with log_path.open("a",encoding="utf-8") as logf, \
         csv_path.open("a",newline="",encoding="utf-8") as csvf:

        writer = csv.writer(csvf)
        if csv_path.stat().st_size == 0:
            writer.writerow(["image","predicted","true"]); csvf.flush()

        trues:list[float] = []; preds:list[float] = []

        for idx,img in enumerate(pending,1):
            fname = img.name
            json_path = dataset / f"{img.stem}.json"
            if not json_path.is_file():
                print(f"[WARN] {fname}: JSON missing – skipped")
                continue
            try:
                true_val = calculate_true_index(json_path)
                pred_val, resp = predict_with_grok(img)

                with llm_path.open("a",encoding="utf-8") as rf:
                    rf.write(f"\n===== {fname} =====\n{resp.strip()}\n")

                logf.write(f"image:{fname}, predicted:{pred_val:.2f}, true:{true_val:.2f}\n")
                logf.flush()
                writer.writerow([fname,f"{pred_val:.2f}",f"{true_val:.2f}"]); csvf.flush()

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
        plt.title("Ki-67 Predicted vs True (xAI)")
        plt.tight_layout(); plt.savefig(plot_path, dpi=300); plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python 3.vlm_processing/1_3.main_xai.py "
            "<processed_dataset> [<output_parent_dir>] "
            "[--resume <existing_out_dir>] [--pause <seconds>]\n\n"
            "Examples:\n"
            "  # fresh run\n"
            "  python 3.vlm_processing/1_3.main_xai.py "
            "1.data_access/data_sample/3.data_processed 5.results\n\n"
            "  # resume, continue only missing images\n"
            "  python 3.vlm_processing/1_3.main_xai.py "
            "1.data_access/data_sample/3.data_processed "
            "--resume 5.results/grok\n\n"
            "  # resume with 1-second pause\n"
            "  python 3.vlm_processing/1_3.main_xai.py "
            "1.data_access/data_sample/3.data_processed "
            "--resume 5.results/grok --pause 1.0"
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
