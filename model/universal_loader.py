from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from model.model_registry import MODEL_REGISTRY

def load_model(model_name, lora_path=None):
    model_name = MODEL_REGISTRY.get(model_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return tokenizer, model
