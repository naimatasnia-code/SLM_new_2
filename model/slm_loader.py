

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

MODELS = {
    "phi-2":      "microsoft/phi-2",
    "tinyllama":  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen-0.5b":  "Qwen/Qwen2.5-0.5B-Instruct",
}

# Models confirmed to run on 4 GB RAM (CPU)
CPU_SAFE_MODELS = {"tinyllama", "qwen-0.5b"}


def load_slm(
    model_name: str,
    lora_path: str | None = None,
    quantized: bool = True,
):
    torch.set_grad_enabled(False)    # global grad off → saves ~30 % RAM

    model_id = MODELS[model_name]
    has_gpu  = torch.cuda.is_available()
    use_quant = quantized and has_gpu

    if not has_gpu:
        print(f"[slm_loader] No GPU detected → CPU mode (float32).")
        if model_name not in CPU_SAFE_MODELS:
            print(
                f"[slm_loader] WARNING: '{model_name}' may exceed 4 GB RAM on CPU. "
                "Recommend: 'tinyllama' or 'qwen-0.5b' for low-memory environments."
            )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Quantization config (GPU only) ────────────────────────────────────────
    quant_config = None
    if use_quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,   # extra ~10 % VRAM saving
            bnb_4bit_quant_type="nf4",
        )

    # ── Model load ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if has_gpu else None,
        quantization_config=quant_config,
        torch_dtype=torch.float16 if has_gpu else torch.float32,
        low_cpu_mem_usage=True,              # stream weights → lower peak RAM
    )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return tokenizer, model
