from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(model_name, lora_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return tokenizer, model
