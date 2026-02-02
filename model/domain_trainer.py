import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def train_domain_lora(model_id, dataset: Dataset, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # CPU model load
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=None
    )

    lora = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.1,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)

    def tokenize(x):
        out = tokenizer(x["text"], truncation=True, padding="max_length", max_length=256)
        out["labels"] = out["input_ids"]
        return out

    dataset = dataset.map(tokenize)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
