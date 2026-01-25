import torch
from core.prompt import build_prompt

class MedicalAgent:
    def __init__(self, tokenizer, model, retriever):
        self.tokenizer = tokenizer
        self.model = model
        self.retriever = retriever

    def answer(self, question: str):
        docs = self.retriever.invoke(question)
        context = "\n".join(d.page_content[:800] for d in docs)

        prompt = build_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = output[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        return {
            "answer": answer,
            "prompt_tokens": len(inputs["input_ids"][0]),
            "completion_tokens": len(generated_ids),
            "total_tokens": len(inputs["input_ids"][0]) + len(generated_ids)
        }
