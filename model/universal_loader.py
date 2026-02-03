from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
MODELS = { "phi-2": "microsoft/phi-2", "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct" }

def load_model(model_name, lora_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return tokenizer, model
