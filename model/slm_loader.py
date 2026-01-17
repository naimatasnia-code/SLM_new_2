from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "phi-2": "microsoft/phi-2",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
}

def load_slm(model_name: str):
    model_id = MODELS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )

    model.eval()
    return tokenizer, model
