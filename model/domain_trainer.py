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
    "adam":        "adamw_torch",
    "adamw":       "adamw_torch",
    "sgd":         "sgd",
    "rmsprop":     "rmsprop",
    "adafactor":   "adafactor",
    "adamw_torch": "adamw_torch",
}


def train_domain_lora(
    model_name: str,
    dataset: Dataset,
    output_dir: str,
    lora_params: dict = None,
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

    # ── FIX 1: Aggressive memory cleanup BEFORE anything else ─────────────────
    # This ensures any previously loaded inference model or leftover tensors
    # are fully cleared before we attempt to load the (heavier) training model.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[trainer] Pre-training memory cleanup done.")

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

    # ── FIX 2: Stricter CPU memory safety caps ────────────────────────────────
    # On CPU float32, every extra rank unit and sequence length costs real RAM.
    # These hard caps prevent OOM before trainer.train() is even reached.
    if not has_gpu:
        batch_size = min(batch_size, 1)    # never exceed 1 on CPU
        lora_rank  = min(lora_rank,  4)    # FIX: capped to 4 (was 8) → saves ~200 MB
        epochs     = min(epochs,     1)    # FIX: cap epochs to 1 on CPU
        grad_ckpt  = True                  # always on for CPU
        print(
            f"[trainer] CPU mode: enforced caps → "
            f"batch=1, lora_rank={lora_rank}, epochs=1, grad_ckpt=True"
        )

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

    # ── FIX 3: Tokenize with max_length=64 on CPU (was 128) ──────────────────
    # Halving sequence length cuts activation tensor memory by ~4x during
    # backprop (activations scale quadratically with sequence length).
    max_seq_len = 128 if has_gpu else 64

    def tokenize(x):
        out = tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    print(f"[trainer] Tokenizing with max_length={max_seq_len} ...")
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
        use_cpu=not has_gpu,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
    )

    print("[trainer] Starting training...")
    trainer.train()
    print("[trainer] Training complete.")

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
