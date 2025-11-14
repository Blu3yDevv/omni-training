# scripts/build_large_dataset.py

from __future__ import annotations

import json
import os
import random
from typing import Dict, List

from datasets import load_dataset

# Where to save the final training file
OUTPUT_PATH = "data/processed/omni_train_large.jsonl"

# Max examples to use (None = use all)
MAX_SAMPLES = 50000  # good starting point for Kaggle

# OmniAI system prompt (adjust as you like)
OMNI_SYSTEM_PROMPT = (
    "You are OmniAI (Omni Nano), a direct, structured assistant. "
    "You explain things clearly and help with general tasks, "
    "study planning, and general reasoning. You are honest and do not make up "
    "facts when unsure. Your tone is calm and not overly formal."
)


def format_alpaca_like(example: Dict) -> Dict:
    """
    Convert an Alpaca-style example (instruction / input / output)
    into your chat-style JSON schema.

    Expected keys:
      - instruction (str)
      - input (str)
      - output (str)
    """
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output_text = (example.get("output") or "").strip()

    if input_text:
        user_content = instruction + "\n\n" + input_text
    else:
        user_content = instruction

    user_content = user_content.strip()
    if not user_content or not output_text:
        return None  # skip bad/empty

    messages = [
        {"role": "system", "content": OMNI_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text.strip()},
    ]

    return {"messages": messages}


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 1) Load an Alpaca-style dataset from Hugging Face
    # NOTE: If this errors, double-check the dataset ID on Hugging Face.
    print("[build_large_dataset] Loading dataset 'yahma/alpaca-cleaned'...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")

    print(f"[build_large_dataset] Loaded {len(ds)} examples.")

    # 2) Shuffle and optionally subsample
    indices = list(range(len(ds)))
    random.shuffle(indices)

    if MAX_SAMPLES is not None:
        indices = indices[:MAX_SAMPLES]

    print(f"[build_large_dataset] Using {len(indices)} examples after subsample.")

    # 3) Convert to chat format and write JSONL
    num_written = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for idx in indices:
            ex = ds[int(idx)]
            formatted = format_alpaca_like(ex)
            if formatted is None:
                continue
            f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"[build_large_dataset] Wrote {num_written} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
