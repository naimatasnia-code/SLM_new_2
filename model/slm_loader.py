from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_slm(
    model_name: str,
    lora_path: str | None = None,
    quantized: bool = True
):
    torch.set_grad_enabled(False)
    model_id = MODELS[model_name]

    has_gpu = torch.cuda.is_available()
    use_quant = quantized and has_gpu

    if not has_gpu:
        print("No GPU detected → running SLM in CPU mode (quantization disabled)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    #Quantization config (only if GPU exists)
    quant_config = None
    if use_quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if has_gpu else None,
        quantization_config=quant_config,
        torch_dtype=torch.float16 if has_gpu else torch.float32
    )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return tokenizer, model
