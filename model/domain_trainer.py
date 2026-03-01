"""
model/domain_trainer.py
=======================
Memory-safe LoRA fine-tuning for 4 GB CPU environments.

Key optimizations:
- gradient_checkpointing: recompute activations instead of storing → -40% RAM
- max_length=128: half the sequence memory vs 256
- gradient_accumulation_steps=4: effective batch=4 with batch_size=1
- low_cpu_mem_usage=True: streams weights during load → lower peak RAM
- del model + gc.collect() after training: self-destructs to free RAM
- Hard blocks phi-2 on CPU (needs 6-7 GB, won't fit in 4 GB)
"""

import gc
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
) -> None:
    """
    Fine-tunes a LoRA adapter on the provided dataset.
    Saves only the adapter (not full model weights) to output_dir.
    Self-destructs after saving to free RAM for inference model reload.

    Raises MemoryError if phi-2 is requested on CPU.
    """
    has_gpu = torch.cuda.is_available()

    if not has_gpu and model_name not in CPU_SAFE_MODELS:
        raise MemoryError(
            f"Model '{model_name}' requires ~6 GB RAM to fine-tune on CPU. "
            f"Use 'tinyllama' or 'qwen-0.5b' for 4 GB environments."
        )

    model_id = MODELS[model_name]

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token     = tokenizer.eos_token
        tokenizer.pad_token_id  = tokenizer.eos_token_id

    # ── Load model (RAM-safe) ─────────────────────────────────────────────────
    print(f"[trainer] Loading {model_name} for fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,   # CPU requires float32
        device_map=None,             # keep on CPU
        low_cpu_mem_usage=True,      # streams weights → lower peak RAM
    )

    # ── LoRA config (minimal rank = tiny adapter, minimal extra RAM) ──────────
    lora_cfg = LoraConfig(
        r=2,                  # very small rank → tiny adapter footprint
        lora_alpha=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # IMPORTANT: gradient checkpointing and enable_input_require_grads MUST be
    # called AFTER get_peft_model(). get_peft_model() replaces the model object,
    # so any hooks set before it are lost and cause:
    #   "element 0 of tensors does not require grad and does not have a grad_fn"
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model.print_trainable_parameters()

    # ── Tokenize ──────────────────────────────────────────────────────────────
    def tokenize(x):
        out = tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=128,       # was 256 → halves sequence tensor memory
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    # ── Training args ─────────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,       # minimum memory per step
        gradient_accumulation_steps=4,       # effective batch=4 without RAM cost
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        fp16=False,                          # CPU does not support fp16 training
        bf16=False,
        dataloader_pin_memory=False,         # requires CUDA
        no_cuda=not has_gpu,
        optim="adamw_torch",                 # lightweight standard optimizer
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
    )
    trainer.train()

    # ── Save adapter only ─────────────────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[trainer] LoRA adapter saved → {output_dir}")

    # ── Self-destruct to free RAM ─────────────────────────────────────────────
    del trainer
    del model
    gc.collect()
    if has_gpu:
        torch.cuda.empty_cache()
    print("[trainer] Training model unloaded. RAM freed.")
