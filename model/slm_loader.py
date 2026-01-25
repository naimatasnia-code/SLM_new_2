from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

MODELS = {
    "phi-2": "microsoft/phi-2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

def load_slm(
    model_name: str,
    lora_path: str | None = None,
    quantized: bool = True
):

    torch.set_grad_enabled(False)
    model_id = MODELS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=quantized,   # 
        torch_dtype=torch.float16
    )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return tokenizer, model
