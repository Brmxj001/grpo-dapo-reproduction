#!/usr/bin/env python3
"""加载模型，跑一次固定输入，打印回答。

运行：python scripts/chat.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_loader import load_model, load_tokenizer

tokenizer = load_tokenizer()
model = load_model()

messages = [{"role": "user", "content": "2+3=?"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )

print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
