# omni-training/scripts/format_datasets.py
import json
from pathlib import Path
from typing import List, Dict

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/omni_train.jsonl")

SYSTEM_PROMPT = (
    "You are Omni Nano, a direct, structured, honest assistant. "
    "You explain clearly, use headings and bullets when useful, and avoid fluff."
)

def load_your_raw_sources() -> List[Dict]:
    # TODO: replace with real loaders (e.g. HF Datasets, local JSON, etc.)
    # Each item should become a {user, assistant, tags} dict.
    # Example:
    return [
        {
            "id": "toy-1",
            "user": "Explain QLoRA in simple terms.",
            "assistant": "QLoRA is a way to fine-tune a large model cheaply by freezing ...",
            "tags": ["ml", "qlora"]
        }
    ]

def main():
    items = load_your_raw_sources()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in items:
            row = {
                "id": item["id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": item["assistant"]},
                ],
                "tags": item.get("tags", []),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
