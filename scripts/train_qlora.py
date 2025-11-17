import os
import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_PATH = "data/processed/omni_train.jsonl"
OUTPUT_DIR = "outputs/omni_nano_qlora"

HF_TOKEN = os.environ.get("HF_TOKEN")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=HF_TOKEN,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return tokenizer, model

def format_example(example: Dict, tokenizer):
    # Convert messages -> a single training string with special formatting.
    messages = example["messages"]

    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text_parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            text_parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            text_parts.append(f"<|assistant|>\n{content}\n")

    full_text = "\n".join(text_parts)
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=2048,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def load_training_dataset(tokenizer):
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    ds = ds.map(lambda ex: format_example(ex, tokenizer), batched=False)
    return ds

def main():
    tokenizer, model = load_tokenizer_and_model()
    train_ds = load_training_dataset(tokenizer)

 training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1.0,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    eval_strategy="no",      # <— NEW NAME
    report_to="none",
)



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
