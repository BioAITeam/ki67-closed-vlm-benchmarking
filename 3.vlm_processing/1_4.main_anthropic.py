#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re, csv, json, time, base64, argparse, sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import anthropic

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key is None:
    raise ValueError("ANTHROPIC_API_KEY not found.")
client = anthropic.Anthropic(api_key=api_key)

THIS_DIR      = Path(__file__).parent
SYSTEM_PROMPT = (THIS_DIR / "system_prompt.txt").read_text(encoding="utf-8")
USER_PROMPT   = (THIS_DIR / "user_prompt.txt").read_text(encoding="utf-8")

MODEL_NAME = "claude-3-5-sonnet-20240620"

_KI67 = re.compile(r"Ki[\s-]?67[^%]*?([0-9]+(?:\.[0-9]+)?)\s*%", re.I | re.S)

def extract_predicted_index(txt: str) -> float:
    m = _KI67.search(txt)
    if m:
        return float(m.group(1))
    vals = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", txt)
    if vals:
        return float(vals[-1])
    raise ValueError("Ki-67 value not found.")

def calculate_true_index(json_path: Path) -> float:
    with json_path.open(encoding="utf-8") as f:
        data: List[dict] = json.load(f)
    pos = sum(1 for c in data if c.get("label_id") == 1)
    neg = sum(1 for c in data if c.get("label_id") == 2)
    return round((pos / (pos + neg)) * 100, 2) if pos + neg else 0.0

def encode_image(img: Path) -> Tuple[str,str]:
    mime = "jpeg" if img.suffix.lower() in {".jpg",".jpeg"} else "png"
    return mime, base64.b64encode(img.read_bytes()).decode()

def predict_with_claude(img: Path) -> Tuple[float,str]:
    mime,b64 = encode_image(img)
    rsp = client.messages.create(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role":"user",
            "content":[
                {"type":"image",
                 "source":{"type":"base64","media_type":f"image/{mime}","data":b64}},
                {"type":"text","text":USER_PROMPT}
            ]
        }]
    )
    txt = rsp.content[0].text.strip()
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
        out_dir = base / f"output_anthropic_{safe_model}_date_{ts}"
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

        trues: List[float] = []; preds: List[float] = []

        for idx,img in enumerate(pending,1):
            fname = img.name
            json_path = dataset / f"{img.stem}.json"
            if not json_path.is_file():
                print(f"[WARN] {fname}: JSON missing – skipped")
                continue
            try:
                true_val = calculate_true_index(json_path)
                pred_val, resp = predict_with_claude(img)

                with llm_path.open("a",encoding="utf-8") as rf:
                    rf.write(f"\n===== {fname} =====\n{resp}\n")

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
        plt.title("Ki-67 Predicted vs True (Claude)")
        plt.tight_layout(); plt.savefig(plot_path, dpi=300); plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python 3.vlm_processing/1_4.main_anthropic.py "
            "<processed_dataset> [<output_parent_dir>] "
            "[--resume <existing_out_dir>] [--pause <seconds>]\n\n"
            "Examples:\n"
            "  # fresh run\n"
            "  python 3.vlm_processing/1_4.main_anthropic.py "
            "1.data_access/data_sample/3.data_processed 5.results\n\n"
            "  # resume missing images with 1-second pause\n"
            "  python 3.vlm_processing/1_4.main_anthropic.py "
            "1.data_access/data_sample/3.data_processed "
            "--resume 5.results/claude --pause 1.0"
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
