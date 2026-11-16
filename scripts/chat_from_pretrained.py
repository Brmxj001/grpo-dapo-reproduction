#!/usr/bin/env python3
"""不依赖 src.model_loader，直接用 transformers 加载本地模型跑一次推理。

可在任意目录运行：python scripts/chat_from_pretrained.py
""" 

from pathlib 
import Path 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = Path(__file__).resolve().parents[1] / "models/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()

messages = [{"role": "user", "content": "2+3=?"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        # 显式置空，避免模型 generation_config 中的采样默认值触发告警
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
