#! -*- coding: utf-8 -*-
# @Time    : 2024/12/29 15:24pip install modelscope --upgrade
# @Author  : LiuGan
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
model_id = "qwen/Qwen2-1.5B-Instruct"
model_dir = "./qwen/Qwen2-1___5B-Instruct"

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download(model_id, cache_dir="./", revision="master")