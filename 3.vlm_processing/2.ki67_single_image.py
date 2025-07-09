#!/usr/bin/env python
# coding: utf-8
"""
Single-image Ki-67 estimation with **four** VLM providers:

  • OpenAI GPT-4 vision models   (prefix: gpt-)
  • Google Gemini 1.5 vision     (prefix: gemini-)
  • xAI Grok vision              (prefix: grok-)
  • Anthropic Claude 3.5 Sonnet  (prefix: claude-)

A provider is selected automatically from the *prefix* of `--model`.
"""

import argparse
import base64
import os
import re
import sys
import time
from pathlib import Path
from typing import Tuple

import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException
from PIL import Image

# ── Prompts ────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).parent
SYSTEM_PROMPT = (THIS_DIR / "system_prompt.txt").read_text(encoding="utf-8")
USER_PROMPT   = (THIS_DIR / "user_prompt.txt").read_text(encoding="utf-8")

# ── Helpers ────────────────────────────────────────────────────────────────────
_KI67_RE = re.compile(r"Ki[\s-]?67[^%]*?([0-9]+(?:\.[0-9]+)?)\s*%", re.I | re.S)
def extract_predicted_index(text: str) -> float:
    m = _KI67_RE.search(text)
    if m:
        return float(m.group(1))
    any_perc = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if any_perc:
        return float(any_perc[-1])
    raise ValueError("Ki-67 value not found in model output.\n" + text)

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_text_tokens(s: str) -> int:
        return len(_enc.encode(s))
except Exception:                       # pragma: no cover
    def _count_text_tokens(s: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", s, re.UNICODE))

def _image_tokens(img_path: Path) -> int:
    """≈170 tokens per megapixel, rounded up to the nearest multiple of 85."""
    with Image.open(img_path) as im:
        w, h = im.size
    tokens = 170 * (w * h / 1_000_000)
    return max(85, int(round(tokens / 85.0)) * 85)

# ── Provider base class ───────────────────────────────────────────────────────
class ProviderBase:
    def predict(self, model_id: str, img_path: Path):
        raise NotImplementedError

    @staticmethod
    def _encode_image(img: Path) -> Tuple[str, str]:
        mime = "jpeg" if img.suffix.lower() in {".jpg", ".jpeg"} else "png"
        return mime, base64.b64encode(img.read_bytes()).decode()

# ── OpenAI ─────────────────────────────────────────────────────────────────────
class OpenAIProvider(ProviderBase):
    def __init__(self):
        from openai import OpenAI
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=key)

    def predict(self, model_id: str, img_path: Path):
        mime, b64 = self._encode_image(img_path)
        start = time.time()
        rsp = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/{mime};base64,{b64}"}}]},
            ],
            temperature=0,
            seed=64,
            max_tokens=1024,
        )
        dur = time.time() - start
        txt = rsp.choices[0].message.content
        usage = {"prompt_tokens": rsp.usage.prompt_tokens,
                 "completion_tokens": rsp.usage.completion_tokens,
                 "total_tokens": rsp.usage.total_tokens,
                 "duration_s": dur}
        return extract_predicted_index(txt), txt, usage

# ── Google Gemini ─────────────────────────────────────────────────────────────
class GoogleProvider(ProviderBase):
    def __init__(self):
        import google.generativeai as genai
        load_dotenv()
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set.")
        genai.configure(api_key=key)
        self.genai = genai

    def predict(self, model_id: str, img_path: Path):
        mime, b64 = self._encode_image(img_path)

        prompt_tokens_txt  = _count_text_tokens(SYSTEM_PROMPT) + _count_text_tokens(USER_PROMPT)
        prompt_tokens_img  = _image_tokens(img_path)
        prompt_tokens_total = prompt_tokens_txt + prompt_tokens_img

        model = self.genai.GenerativeModel(f"models/{model_id}")
        image_part = {"mime_type": f"image/{mime}", "data": base64.b64decode(b64)}

        start = time.time()
        rsp = model.generate_content(
            [SYSTEM_PROMPT, image_part, USER_PROMPT],
            generation_config=self.genai.types.GenerationConfig(temperature=0.0)
        )
        dur  = time.time() - start
        txt  = rsp.text or ""
        completion_tokens = _count_text_tokens(txt)
        total_tokens = prompt_tokens_total + completion_tokens

        return extract_predicted_index(txt), txt, {
            "prompt_tokens": prompt_tokens_total,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_s": dur,
        }

# ── xAI Grok ──────────────────────────────────────────────────────────────────
class XAIProvider(ProviderBase):
    API_URL = "https://api.x.ai/v1/chat/completions"

    def __init__(self):
        load_dotenv()
        self.key = os.getenv("XAI_API_KEY")
        if not self.key:
            raise RuntimeError("XAI_API_KEY not set.")

    def predict(self, model_id: str, img_path: Path):
        mime, b64 = self._encode_image(img_path)

        prompt_tokens_txt  = _count_text_tokens(SYSTEM_PROMPT) + _count_text_tokens(USER_PROMPT)
        prompt_tokens_img  = _image_tokens(img_path)
        prompt_tokens_total = prompt_tokens_txt + prompt_tokens_img

        payload = {
            "model": model_id,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/{mime};base64,{b64}"}}]},
            ],
        }
        headers = {"Authorization": f"Bearer {self.key}",
                   "Content-Type": "application/json"}

        start = time.time()
        r = requests.post(self.API_URL, json=payload, headers=headers, timeout=60)
        try:
            r.raise_for_status()
        except RequestException as e:
            raise RuntimeError(f"xAI request error: {e}") from e
        dur = time.time() - start

        txt = r.json()["choices"][0]["message"]["content"]
        completion_tokens = _count_text_tokens(txt)
        total_tokens = prompt_tokens_total + completion_tokens

        return extract_predicted_index(txt), txt, {
            "prompt_tokens": prompt_tokens_total,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_s": dur,
        }

# ── Anthropic Claude ──────────────────────────────────────────────────────────
class AnthropicProvider(ProviderBase):
    def __init__(self):
        import anthropic
        load_dotenv()
        self.key = os.getenv("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic(api_key=self.key)

    def predict(self, model_id: str, img_path: Path):
        mime, b64 = self._encode_image(img_path)

        prompt_tokens_txt  = _count_text_tokens(SYSTEM_PROMPT) + _count_text_tokens(USER_PROMPT)
        prompt_tokens_img  = _image_tokens(img_path)
        prompt_tokens_total = prompt_tokens_txt + prompt_tokens_img

        start = time.time()
        rsp = self.client.messages.create(
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
        completion_tokens = _count_text_tokens(txt)
        total_tokens = prompt_tokens_total + completion_tokens

        return extract_predicted_index(txt), txt, {
            "prompt_tokens": prompt_tokens_total,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_s": dur,
        }

# ── Provider registry ─────────────────────────────────────────────────────────
PROVIDERS = {
    "gpt":    ("OpenAI",        OpenAIProvider),
    "gemini": ("Google Gemini", GoogleProvider),
    "grok":   ("xAI Grok",      XAIProvider),
    "claude": ("Anthropic",     AnthropicProvider),
}
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"

def provider_for(model_id: str) -> ProviderBase:
    for pfx, (_, cls) in PROVIDERS.items():
        if model_id.lower().startswith(pfx):
            return cls()
    raise ValueError(f"Unknown model '{model_id}'. Valid prefixes: {', '.join(PROVIDERS)}")

# ── Main routine ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ki-67 prediction on a single image using "
                    "OpenAI, Google, xAI, or Anthropic."
    )
    parser.add_argument("image", help="path to image (.jpg/.jpeg/.png)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help="model identifier (prefix decides provider)")
    args = parser.parse_args()

    img = Path(args.image).expanduser().resolve()
    if not img.is_file():
        sys.exit(f"File '{img}' not found.")

    try:
        provider = provider_for(args.model)
    except ValueError as e:
        opts = "\n  - " + "\n  - ".join(f"{p}: {d[0]}" for p, d in PROVIDERS.items())
        sys.exit(str(e) + "\nAvailable options:" + opts)

    print(f"Model: {args.model} ({provider.__class__.__name__})\n")

    try:
        ki67, raw, usage = provider.predict(args.model, img)
    except Exception as e:
        sys.exit(f"Prediction failed: {e}")

    print("=" * 60)
    print(raw.strip())
    print("=" * 60)
    print(f"Ki-67 Index       : {ki67:.2f}%")
    print(f"Execution time    : {usage['duration_s']:.2f}s")
    print(f"Prompt tokens     : {usage['prompt_tokens']}")
    print(f"Completion tokens : {usage['completion_tokens']}")
    print(f"Total tokens      : {usage['total_tokens']}")
    print("=" * 60)

if __name__ == "__main__":
    # Expected call:
    #   python 3.vlm_processing/2.ki67_single_image.py <image_path> [--model <model_id>]
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
    # Complete example invocations (same image, different VLMs)
    # -----------------------------------------------------------------------
    # OpenAI:
    #   python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4o-2024-11-20
    #   python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4.1-2025-04-14
    #   python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4.1-mini-2025-04-14
    #   python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4.5-preview
    #
    # Google Gemini:
    #   python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gemini-1.5-pro
    #   python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gemini-1.5-flash
    #
    # xAI Grok:
    #   python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model grok-2-vision-latest
    #
    # Anthropic Claude:
    #   python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model claude-3-5-sonnet-20240620
    #
    if len(sys.argv) not in (2, 4) or (len(sys.argv) == 4 and sys.argv[2] not in {"--model", "-m"}):
        print(
            "Usage:\n"
            "  python 3.vlm_processing/2.ki67_single_image.py <image_path> [--model <model_id>]\n\n"
            "Model examples:\n"
            "  • OpenAI      : gpt-4o-2024-11-20 | gpt-4.1-2025-04-14 | "
            "gpt-4.1-mini-2025-04-14 | gpt-4.5-preview\n"
            "  • Gemini      : gemini-1.5-pro | gemini-1.5-flash\n"
            "  • xAI Grok    : grok-2-vision-latest\n"
            "  • Anthropic   : claude-3-5-sonnet-20240620\n\n"
            "Example (default OpenAI - gpt-4.1-mini-2025-04-14):\n"
            "  python 3.vlm_processing/2.ki67_single_image.py "
            "1.data_access/data_sample_BCData/3.data_processed/8.jpg\n\n"
            "Example (OpenAI):\n"
            "  python 3.vlm_processing/2.ki67_single_image.py "
            "1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4.1-mini-2025-04-14\n"
            "  python 3.vlm_processing/2.ki67_single_image.py "
            "1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4.1-2025-04-14\n"
            "  python 3.vlm_processing/2.ki67_single_image.py "
            "1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4.5-preview\n"
            "  python 3.vlm_processing/2.ki67_single_image.py "
            "1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4o-2024-11-20\n\n"
            "Example (Google Gemini):\n"
            "  python 3.vlm_processing/2.ki67_single_image.py "
            "1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gemini-1.5-pro\n"
            "  python 3.vlm_processing/2.ki67_single_image.py "
            "1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gemini-1.5-flash\n\n"
            "Example (xAI Grok):\n"
            "  python 3.vlm_processing/2.ki67_single_image.py "
            "1.data_access/data_sample_BCData/3.data_processed/8.jpg --model grok-2-vision-latest\n\n"
            "Example (Anthropic Claude):\n"
            "  python 3.vlm_processing/2.ki67_single_image.py "
            "1.data_access/data_sample_BCData/3.data_processed/8.jpg --model claude-3-5-sonnet-20240620\n\n"
            "If --model is omitted, the default is gpt-4.1-mini-2025-04-14."
        )
        sys.exit(1)

    main() 