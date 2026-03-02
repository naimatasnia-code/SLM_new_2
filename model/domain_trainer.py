

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

# Valid optimizer names accepted by HuggingFace Trainer
OPTIMIZER_MAP = {
    "adam":     "adamw_torch",
    "adamw":    "adamw_torch",
    "sgd":      "sgd",
    "rmsprop":  "rmsprop",
    "adafactor": "adafactor",
    "adamw_torch": "adamw_torch",
}


def train_domain_lora(
    model_name: str,
    dataset: Dataset,
    output_dir: str,
    lora_params: dict = None,   # ← dynamic params from UI
) -> None:
    """
    Fine-tunes a LoRA adapter on the provided dataset.
    Saves only the adapter (not full model weights) to output_dir.
    Self-destructs after saving to free RAM for inference model reload.

    Args:
        model_name  : one of MODELS keys
        dataset     : HuggingFace Dataset with a 'text' column
        output_dir  : path to save the LoRA adapter
        lora_params : dict of UI-supplied parameters (all optional, safe defaults used)

    Raises MemoryError if phi-2 is requested on CPU.
    """
    # ── Defaults (safe for 4 GB CPU) ─────────────────────────────────────────
    p = lora_params or {}

    lora_rank      = int(p.get("lora_rank",      8))
    lora_alpha     = int(p.get("lora_alpha",     16))
    lora_dropout   = float(p.get("lora_dropout",  0.05))
    target_modules = p.get("target_modules",     ["q_proj", "v_proj"])
    learning_rate  = float(p.get("learning_rate", 2e-4))
    batch_size     = int(p.get("batch_size",      1))
    epochs         = int(p.get("epochs",          1))
    grad_ckpt      = bool(p.get("gradient_checkpointing", True))

    # Normalize optimizer name → HuggingFace key
    raw_optim = str(p.get("optimizer", "adamw_torch")).lower()
    optimizer = OPTIMIZER_MAP.get(raw_optim, "adamw_torch")

    # ── Hardware check ────────────────────────────────────────────────────────
    has_gpu = torch.cuda.is_available()

    if not has_gpu and model_name not in CPU_SAFE_MODELS:
        raise MemoryError(
            f"Model '{model_name}' requires ~6 GB RAM to fine-tune on CPU. "
            f"Use 'tinyllama' or 'qwen-0.5b' for 4 GB environments."
        )

    # Enforce minimum memory safety on CPU regardless of UI input
    if not has_gpu:
        batch_size = min(batch_size, 1)   # never exceed 1 on CPU
        lora_rank  = min(lora_rank,  8)   # cap rank at 8 on CPU
        grad_ckpt  = True                 # always on for CPU

    model_id = MODELS[model_name]

    print(f"[trainer] LoRA config → rank={lora_rank}, alpha={lora_alpha}, "
          f"dropout={lora_dropout}, modules={target_modules}")
    print(f"[trainer] Training config → lr={learning_rate}, batch={batch_size}, "
          f"epochs={epochs}, optimizer={optimizer}, grad_ckpt={grad_ckpt}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Load model (RAM-safe) ─────────────────────────────────────────────────
    print(f"[trainer] Loading {model_name} for fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,   # CPU requires float32
        device_map=None,             # keep on CPU
        low_cpu_mem_usage=True,      # streams weights → lower peak RAM
    )

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # IMPORTANT: gradient_checkpointing and enable_input_require_grads MUST be
    # called AFTER get_peft_model(). get_peft_model() replaces the model object,
    # so any hooks set before it are lost and cause:
    #   "element 0 of tensors does not require grad and does not have a grad_fn"
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    # ── Tokenize ──────────────────────────────────────────────────────────────
    def tokenize(x):
        out = tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=128,       # halves sequence tensor memory vs 256
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    # ── Training args ─────────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 4 // batch_size),  # keep effective batch ~4
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        optim=optimizer,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        fp16=False,                          # CPU does not support fp16 training
        bf16=False,
        dataloader_pin_memory=False,         # requires CUDA
        use_cpu=not has_gpu
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
