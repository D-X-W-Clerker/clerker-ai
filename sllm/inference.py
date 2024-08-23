import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatGeneration:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

        if self.model.config.eos_token_id is None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.terminators = [
            self.model.config.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("")
        ]

    def generate(self, instruction, max_new_tokens=2048, do_sample=True, temperature=0.6, top_p=0.9):
        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''

        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{instruction}"}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            attention_mask=attention_mask
        )

        return self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

def main(instruction_text):
    generator = ChatGeneration(model_id="MLP-KTLim/llama-3-Korean-Bllossom-8B", device_map="auto")
    response = generator.generate(instruction_text)
    
    return response

if __name__ == "__main__":
  instruction_text = sys.argv[1]
  print(instruction_text)
  main(instruction_text)
