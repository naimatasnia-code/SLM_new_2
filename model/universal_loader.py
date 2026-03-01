"""
model/universal_loader.py  –  Updated
- Was loading with no quantization or memory settings at all (memory bomb)
- Now mirrors slm_loader settings: low_cpu_mem_usage, float32 on CPU
- Adds missing pad_token fix
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODELS = {
    "phi-2":      "microsoft/phi-2",
    "tinyllama":  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen-0.5b":  "Qwen/Qwen2.5-0.5B-Instruct",
}


def load_model(model_name: str, lora_path: str | None = None):
    torch.set_grad_enabled(False)

    model_id = MODELS[model_name]
    has_gpu  = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if has_gpu else None,
        torch_dtype=torch.float16 if has_gpu else torch.float32,
        low_cpu_mem_usage=True,
    )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return tokenizer, model
