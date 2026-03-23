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

OPTIMIZER_MAP = {
    "adam":        "adamw_torch",
    "adamw":       "adamw_torch",
    "sgd":         "sgd",
    "rmsprop":     "rmsprop",
    "adafactor":   "adafactor",
    "adamw_torch": "adamw_torch",
}


def _make_inputs_require_grad(module, input, output):
    """
    Forward hook attached to the embedding layer.
    Forces the output tensor to require gradients so that
    gradient checkpointing does not sever the computation graph.

    This is the correct fix for:
      RuntimeError: element 0 of tensors does not require grad
                    and does not have a grad_fn
    which occurs when gradient_checkpointing_enable() is called
    alongside LoRA on CPU (no autocast, pure float32).
    """
    output.requires_grad_(True)


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

    # ── Pre-training memory cleanup ───────────────────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[trainer] Pre-training memory cleanup done.")

    # ── Defaults ──────────────────────────────────────────────────────────────
    p = lora_params or {}

    lora_rank      = int(p.get("lora_rank",      8))
    lora_alpha     = int(p.get("lora_alpha",     16))
    lora_dropout   = float(p.get("lora_dropout",  0.05))
    target_modules = p.get("target_modules",     ["q_proj", "v_proj"])
    learning_rate  = float(p.get("learning_rate", 2e-4))
    batch_size     = int(p.get("batch_size",      1))
    epochs         = int(p.get("epochs",          1))
    grad_ckpt      = bool(p.get("gradient_checkpointing", True))

    raw_optim = str(p.get("optimizer", "adamw_torch")).lower()
    optimizer = OPTIMIZER_MAP.get(raw_optim, "adamw_torch")

    # ── Hardware check ────────────────────────────────────────────────────────
    has_gpu = torch.cuda.is_available()

    if not has_gpu and model_name not in CPU_SAFE_MODELS:
        raise MemoryError(
            f"Model '{model_name}' requires ~6 GB RAM to fine-tune on CPU. "
            f"Use 'tinyllama' or 'qwen-0.5b' for 4 GB environments."
        )

    # ── CPU safety caps ───────────────────────────────────────────────────────
    if not has_gpu:
        batch_size = min(batch_size, 1)
        lora_rank  = min(lora_rank,  4)
        epochs     = min(epochs,     1)
        grad_ckpt  = True
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

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"[trainer] Loading {model_name} for fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
    )

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # ── Gradient checkpointing setup (THE KEY FIX) ────────────────────────────
    # Problem:  gradient_checkpointing_enable() + LoRA on CPU severs the
    #           autograd graph at the embedding boundary, causing:
    #           "element 0 of tensors does not require grad and does not have a grad_fn"
    #
    # Wrong fix: model.enable_input_require_grads() — unreliable on some
    #            transformers versions with pure float32 / no autocast.
    #
    # Correct fix: register a forward hook directly on the embedding layer that
    #              calls requires_grad_(True) on its output every forward pass.
    #              This is version-agnostic and works on CPU and GPU.
    if grad_ckpt:
        model.gradient_checkpointing_enable()

        # Find embedding layer — covers LLaMA, Qwen, Phi, Mistral, Falcon, etc.
        embedding_layer = None
        for name, module in model.named_modules():
            if "embed_tokens" in name or "embedding" in name.lower():
                embedding_layer = module
                break

        if embedding_layer is not None:
            embedding_layer.register_forward_hook(_make_inputs_require_grad)
            print("[trainer] Gradient hook attached to embedding layer.")
        else:
            # Fallback for architectures with non-standard embedding names
            model.enable_input_require_grads()
            print("[trainer] Fallback: enable_input_require_grads() used.")

    model.print_trainable_parameters()

    # ── Tokenize ──────────────────────────────────────────────────────────────
    # max_length=64 on CPU: activation memory scales quadratically with
    # sequence length, halving it saves ~4x backprop RAM.
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
        gradient_accumulation_steps=max(1, 4 // batch_size),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        optim=optimizer,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
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

    # ── Free RAM ──────────────────────────────────────────────────────────────
    del trainer
    del model
    gc.collect()
    if has_gpu:
        torch.cuda.empty_cache()
    print("[trainer] Training model unloaded. RAM freed.")