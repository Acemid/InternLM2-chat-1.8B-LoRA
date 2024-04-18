# InternLM2-chat-1.8B-LoRA
# -- coding: utf-8 --
# -------------------------------
# @Author : Ning Zhang
# @Email : zhang_n@zju.edu.cn
# -------------------------------
# @File : app.py.py
# @Time : 2024/4/18 14:25
# -------------------------------

import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# download internlm2 to the base_path directory using git tool
base_path = './internlm2-chat-1.8b'
os.system(f'git clone https://code.openxlab.org.cn/kino/InternLM2-chat-1.8B-LoRA.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')



tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="InternLM2-Chat-1.8B-LoRA-kino",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory, characterized by kino.  
                 """,
                 ).queue(1).launch()

