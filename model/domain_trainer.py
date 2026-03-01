

import gc
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODELS = {
    "phi-2":     "microsoft/phi-2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
}

CPU_SAFE_MODELS = {"tinyllama", "qwen-0.5b"}


def train_domain_lora(
    model_name: str,
    dataset: Dataset,
    output_dir: str,
):
    has_gpu = torch.cuda.is_available()

    if not has_gpu and model_name not in CPU_SAFE_MODELS:
        raise MemoryError(
            f"Model '{model_name}' is too large to fine-tune on CPU with 4 GB RAM. "
            "Use 'tinyllama' or 'qwen-0.5b'."
        )

    model_id = MODELS[model_name]

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Load model fresh for training (low memory) ────────────────────────────
    print("[trainer] Loading model for fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,   # CPU requires float32
        device_map=None,             # keep on CPU
        low_cpu_mem_usage=True,      # stream weights → lower peak RAM
    )

    # Gradient checkpointing: recompute activations instead of storing them
    # Cuts ~40% RAM at cost of ~20% slower training — worth it on 4 GB
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # required when using grad checkpointing + PEFT

    # ── LoRA (minimal footprint) ──────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=2,                  # very small rank → tiny adapter
        lora_alpha=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Tokenize dataset ──────────────────────────────────────────────────────
    def tokenize(x):
        out = tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=128,      # was 256 → halves sequence memory
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    # ── Training args (CPU-safe) ───────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,       # minimum batch
        gradient_accumulation_steps=4,       # effective batch=4 without RAM cost
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        fp16=False,                          # CPU does not support fp16 training
        bf16=False,
        dataloader_pin_memory=False,         # pin_memory needs CUDA
        no_cuda=not has_gpu,                 # explicit CPU-only flag
        optim="adamw_torch",                 # lightweight optimizer
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
    )
    trainer.train()

    # ── Save adapter only (not full model weights) ────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[trainer] LoRA adapter saved → {output_dir}")

    # ── Unload training model immediately to free RAM ─────────────────────────
    del model
    del trainer
    gc.collect()
    if has_gpu:
        torch.cuda.empty_cache()
    print("[trainer] Training model unloaded. RAM freed.")
