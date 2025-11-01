from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

MODEL_NAME = "ruslanmv/Medical-Llama3-8B"
LOCAL_MODEL_DIR = "local_medical_llama"

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ---------------- SAVE LOCALLY ----------------
model.save_pretrained(LOCAL_MODEL_DIR)
tokenizer.save_pretrained(LOCAL_MODEL_DIR)

print(f"Model and tokenizer saved to {LOCAL_MODEL_DIR}")
