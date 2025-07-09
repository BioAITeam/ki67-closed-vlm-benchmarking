#!/usr/bin/env python
# coding: utf-8
"""
Time- & token-usage benchmark on N sample images for any supported VLM.

Supported providers / model-prefixes
  • OpenAI      (gpt-…)
  • Google      (gemini-…)
  • xAI         (grok-…)
  • Anthropic   (claude-…)
"""

import argparse
import base64
import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List
import tiktoken
import requests
from dotenv import load_dotenv
from PIL import Image
from requests.exceptions import RequestException

this_dir = Path(__file__).parent
SYSTEM_PROMPT = (this_dir / "../3.vlm_processing/system_prompt.txt").read_text(encoding="utf-8")
USER_PROMPT   = (this_dir / "../3.vlm_processing/user_prompt.txt").read_text(encoding="utf-8")

_KI67_RE       = re.compile(r"Ki[\s-]?67[^%]*?([0-9]+(?:\.[0-9]+)?)\s*%", re.I | re.S)
_CELL_RE_POS   = re.compile(r"Immunopositive cells?:\s*(\d+)", re.I)
_CELL_RE_NEG   = re.compile(r"Immunonegative cells?:\s*(\d+)", re.I)

def _extract_predicted_index(text: str) -> float:
    m = _KI67_RE.search(text)
    if m:
        return float(m.group(1))
    perc = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    return float(perc[-1]) if perc else 0.0

def extract_cell_counts_and_index(text: str) -> Tuple[int, int, float]:
    pos = int(_CELL_RE_POS.search(text).group(1)) if _CELL_RE_POS.search(text) else 0
    neg = int(_CELL_RE_NEG.search(text).group(1)) if _CELL_RE_NEG.search(text) else 0
    ki  = _extract_predicted_index(text)
    return pos, neg, ki

try:
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_text_tokens(txt: str) -> int:
        return len(_enc.encode(txt))
except Exception:  # pragma: no cover
    def _count_text_tokens(txt: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", txt))

def _image_tokens(img_path: Path) -> int:
    with Image.open(img_path) as im:
        w, h = im.size
    approx = 170 * (w * h / 1_000_000)
    return max(85, int(round(approx / 85.0)) * 85)

class ProviderBase:
    def predict(self, model_id: str, img: Path):
        raise NotImplementedError

    @staticmethod
    def _encode_image(img: Path) -> Tuple[str, str]:
        mime = "jpeg" if img.suffix.lower() in {".jpg", ".jpeg"} else "png"
        return mime, base64.b64encode(img.read_bytes()).decode()

# ────────────────────────── OpenAI ────────────────────────────────────────────
class OpenAIProvider(ProviderBase):
    def __init__(self):
        from openai import OpenAI
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.cli = OpenAI(api_key=key)

    def predict(self, model_id: str, img: Path):
        mime, b64 = self._encode_image(img)
        start = time.time()
        rsp = self.cli.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text",      "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}]},
            ],
            temperature=0, seed=64, max_tokens=1024,
        )
        dur = time.time() - start
        txt = rsp.choices[0].message.content
        pos, neg, ki = extract_cell_counts_and_index(txt)
        usage = {"prompt_tokens": rsp.usage.prompt_tokens,
                 "completion_tokens": rsp.usage.completion_tokens,
                 "total_tokens": rsp.usage.total_tokens,
                 "duration_s": dur}
        return pos, neg, ki, txt, usage

# ──────────────────────── Google Gemini ───────────────────────────────────────
class GoogleProvider(ProviderBase):
    def __init__(self):
        import google.generativeai as genai
        load_dotenv()
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set.")
        genai.configure(api_key=key)
        self.genai = genai

    def predict(self, model_id: str, img: Path):
        mime, b64 = self._encode_image(img)
        prompt_tok = (_count_text_tokens(SYSTEM_PROMPT)
                      + _count_text_tokens(USER_PROMPT)
                      + _image_tokens(img))
        model = self.genai.GenerativeModel(f"models/{model_id}")
        image_part = {"mime_type": f"image/{mime}", "data": base64.b64decode(b64)}
        start = time.time()
        rsp = model.generate_content(
            [SYSTEM_PROMPT, image_part, USER_PROMPT],
            generation_config=self.genai.types.GenerationConfig(temperature=0.0)
        )
        dur = time.time() - start
        txt = rsp.text or ""
        comp = _count_text_tokens(txt)
        tot  = prompt_tok + comp
        pos, neg, ki = extract_cell_counts_and_index(txt)
        usage = {"prompt_tokens": prompt_tok,
                 "completion_tokens": comp,
                 "total_tokens": tot,
                 "duration_s": dur}
        return pos, neg, ki, txt, usage

# ────────────────────────── xAI Grok ──────────────────────────────────────────
class XAIProvider(ProviderBase):
    API_URL = "https://api.x.ai/v1/chat/completions"

    def __init__(self):
        load_dotenv()
        self.key = os.getenv("XAI_API_KEY")
        if not self.key:
            raise RuntimeError("XAI_API_KEY not set.")

    def predict(self, model_id: str, img: Path):
        mime, b64 = self._encode_image(img)
        prompt_tok = (_count_text_tokens(SYSTEM_PROMPT)
                      + _count_text_tokens(USER_PROMPT)
                      + _image_tokens(img))
        payload = {
            "model": model_id, "temperature": 0, "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}]},
            ],
        }
        headers = {"Authorization": f"Bearer {self.key}",
                   "Content-Type": "application/json"}
        start = time.time()
        r = requests.post(self.API_URL, json=payload, headers=headers, timeout=60)
        try:
            r.raise_for_status()
        except RequestException as e:
            raise RuntimeError(f"xAI error: {e}") from e
        dur = time.time() - start
        txt = r.json()["choices"][0]["message"]["content"]
        comp = _count_text_tokens(txt)
        tot  = prompt_tok + comp
        pos, neg, ki = extract_cell_counts_and_index(txt)
        usage = {"prompt_tokens": prompt_tok,
                 "completion_tokens": comp,
                 "total_tokens": tot,
                 "duration_s": dur}
        return pos, neg, ki, txt, usage

# ────────────────────── Anthropic Claude ──────────────────────────────────────
class AnthropicProvider(ProviderBase):
    def __init__(self):
        import anthropic
        load_dotenv()
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self.cli = anthropic.Anthropic(api_key=key)

    def predict(self, model_id: str, img: Path):
        mime, b64 = self._encode_image(img)
        prompt_tok = (_count_text_tokens(SYSTEM_PROMPT)
                      + _count_text_tokens(USER_PROMPT)
                      + _image_tokens(img))
        start = time.time()
        rsp = self.cli.messages.create(
            model=model_id,
            system=SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{mime}",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            }],
        )
        dur = time.time() - start
        txt = rsp.content[0].text.strip()
        comp = _count_text_tokens(txt)
        tot  = prompt_tok + comp
        pos, neg, ki = extract_cell_counts_and_index(txt)
        usage = {"prompt_tokens": prompt_tok,
                 "completion_tokens": comp,
                 "total_tokens": tot,
                 "duration_s": dur}
        return pos, neg, ki, txt, usage

# ───────────────────── Provider registry / defaults ───────────────────────────
PROVIDERS = {
    "gpt":    ("OpenAI",        OpenAIProvider),
    "gemini": ("Google Gemini", GoogleProvider),
    "grok":   ("xAI Grok",      XAIProvider),
    "claude": ("Anthropic",     AnthropicProvider),
}
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"

def _provider_for(model_id: str) -> ProviderBase:
    for pfx, (_, cls) in PROVIDERS.items():
        if model_id.lower().startswith(pfx):
            return cls()
    raise ValueError(f"Unknown model '{model_id}'. Prefixes: {', '.join(PROVIDERS)}")

def _slice_images(dataset: Path, n_spec: str | int) -> List[Path]:
    imgs = [p for p in sorted(dataset.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (dataset / f"{p.stem}.json").is_file()]
    total = len(imgs)
    if total == 0:
        return []

    if isinstance(n_spec, str):
        t = n_spec.strip().lower()
        if t in {"all", "-1"}:
            n = total
        else:
            n = int(t)
    else:
        n = n_spec

    if n == -1:
        n = total
    elif n < -1:
        n = max(0, total + n + 1)

    if n == 0:
        raise ValueError("Resulting sample size is 0 — nothing to process.")
    if n > total:
        n = total

    return imgs[:n]

def _analyze_samples(dataset: Path, out_parent: Path,
                     model_id: str, n_spec: str | int = 10) -> None:
    prov = _provider_for(model_id)
    imgs = _slice_images(dataset, n_spec)
    if not imgs:
        sys.exit("No valid images found.")

    print(f"Provider : {prov.__class__.__name__}")
    print(f"Model    : {model_id}")
    print(f"Samples  : {len(imgs)}\n")

    ts = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    out_dir = out_parent / f"time_benchmark_{model_id}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / "time_stats.csv"
    resp_path = out_dir / "raw_responses.txt"

    times: List[float] = []; inT=[]; outT=[]; totT=[]
    with csv_path.open("w", newline="", encoding="utf-8") as cf, \
         resp_path.open("w", encoding="utf-8") as rf:
        wr = csv.writer(cf)
        wr.writerow(["image","prompt_tokens","completion_tokens",
                     "total_tokens","time_s",
                     "ki67","pos_cells","neg_cells"])
        for idx, img in enumerate(imgs, 1):
            print(f"[{idx}/{len(imgs)}] {img.name}")
            try:
                pos, neg, ki, txt, usage = prov.predict(model_id, img)
            except Exception as e:
                print(f"  {e}")
                continue

            wr.writerow([img.name, usage["prompt_tokens"],
                         usage["completion_tokens"], usage["total_tokens"],
                         f"{usage['duration_s']:.2f}",
                         f"{ki:.2f}", pos, neg])

            rf.write(f"\n===== {img.name} =====\n")
            rf.write(f"Time: {usage['duration_s']:.2f}s | "
                     f"Tokens: {usage['total_tokens']}\n")
            rf.write(txt.strip() + "\n")

            times.append(usage["duration_s"])
            inT.append(usage["prompt_tokens"])
            outT.append(usage["completion_tokens"])
            totT.append(usage["total_tokens"])

    if times:
        print("\nAVERAGE METRICS")
        print(f"Avg time           : {sum(times)/len(times):.2f}s")
        print(f"Avg prompt tokens  : {sum(inT)/len(inT):.0f}")
        print(f"Avg completion tok : {sum(outT)/len(outT):.0f}")
        print(f"Avg total tokens   : {sum(totT)/len(totT):.0f}")
        print(f"Results saved in   : {out_dir}")

def main() -> None:
    pa = argparse.ArgumentParser(
        description="Average execution-time & token usage on N samples "
                    "with any supported VLM."
    )
    pa.add_argument("dataset", help="folder with processed images & JSONs")
    pa.add_argument("output",  help="parent folder to store results")
    pa.add_argument("-m", "--model", default=DEFAULT_MODEL,
                    help=f"model id (default {DEFAULT_MODEL})")
    pa.add_argument("-n", "--num", default="10",
                    help="number of samples: positive int, 'all', -1 (all), "
                         "-2 (all-1), etc.  default: 10")
    args = pa.parse_args()

    ds = Path(args.dataset).resolve()
    op = Path(args.output).resolve()
    if not ds.is_dir():
        sys.exit(f"Dataset folder not found: {ds}")
    if not op.is_dir():
        sys.exit(f"Output parent dir not found: {op}")

    try:
        _analyze_samples(ds, op, args.model, args.num)
    except ValueError as err:
        sys.exit(f"Error: {err}")

if __name__ == "__main__":    
    # Expected call:
    #   python 4.utils/calculate_time_average.py <processed_dataset> <output_parent_dir> [--model <model_id>] [--n <int>]
    #
    # Tested model IDs
    # ── OpenAI ───────────────────────────────────────────────────────────────
    #   gpt-4o-2024-11-20
    #   gpt-4.1-2025-04-14
    #   gpt-4.1-mini-2025-04-14   ← default
    #   gpt-4.5-preview
    #
    # ── Google Gemini ────────────────────────────────────────────────────────
    #   gemini-1.5-pro
    #   gemini-1.5-flash
    #
    # ── xAI Grok ─────────────────────────────────────────────────────────────
    #   grok-2-vision-latest
    #
    # ── Anthropic Claude ─────────────────────────────────────────────────────
    #   claude-3-5-sonnet-20240620
    #
    # Complete example invocations (same dataset, different VLMs)
    # -----------------------------------------------------------------------
    # OpenAI:
    #   python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4o-2024-11-20 --n all
    #   python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4.1-2025-04-14 --n all
    #   python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4.1-mini-2025-04-14 --n all
    #   python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4.5-preview --n 10
    #
    # Google Gemini:
    #   python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gemini-1.5-pro --n all
    #   python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gemini-1.5-flash --n all
    #
    # xAI Grok:
    #   python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model grok-2-vision-latest --n all
    #
    # Anthropic Claude:
    #   python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model claude-3-5-sonnet-20240620 --n -1
    # -----------------------------------------------------------------------

    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "  python 4.utils/calculate_time_average.py <processed_dataset> <output_parent_dir> "
            "[--model <model_id>] [--n <int>]\n\n"
            "Model examples:\n"
            "  • OpenAI      : gpt-4o-2024-11-20 | gpt-4.1-2025-04-14 | "
            "gpt-4.1-mini-2025-04-14 | gpt-4.5-preview\n"
            "  • Gemini      : gemini-1.5-pro | gemini-1.5-flash\n"
            "  • xAI Grok    : grok-2-vision-latest\n"
            "  • Anthropic   : claude-3-5-sonnet-20240620\n\n"
            "Example (default OpenAI – gpt-4.1-mini-2025-04-14):\n"
            "  python 4.utils/calculate_time_average.py "
            "1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results\n\n"
            "Example (OpenAI):\n"
            "  python 4.utils/calculate_time_average.py "
            "1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4o-2024-11-20 --n all\n"
            "  python 4.utils/calculate_time_average.py "
            "1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4.1-2025-04-14 --n all\n"
            "  python 4.utils/calculate_time_average.py "
            "1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4.1-mini-2025-04-14 --n all\n"
            "  python 4.utils/calculate_time_average.py "
            "1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4.5-preview --n all\n\n"
            "Example (Google Gemini):\n"
            "  python 4.utils/calculate_time_average.py "
            "1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gemini-1.5-pro --n all\n"
            "  python 4.utils/calculate_time_average.py "
            "1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gemini-1.5-flash --n all\n\n"
            "Example (xAI Grok):\n"
            "  python 4.utils/calculate_time_average.py "
            "1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model grok-2-vision-latest --n all\n\n"
            "Example (Anthropic Claude):\n"
            "  python 4.utils/calculate_time_average.py "
            "1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model claude-3-5-sonnet-20240620 --n -1\n\n"
            "If --model is omitted, the default is gpt-4.1-mini-2025-04-14.\n"
            "Use --n to change the number of sampled images (default: 10)."
        )
        sys.exit(1)

    main() 