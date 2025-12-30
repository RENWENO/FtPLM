#! -*- coding: utf-8 -*-
# @Time    : 2025/12/3 16:30
# @Author  : LiuGan

from transformers import AutoTokenizer, AutoModelForMaskedLM

from huggingface_hub import snapshot_download
snapshot_download(repo_id="bert-base-chinese",local_dir="./BERT")

