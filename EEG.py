#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG-style EEG + SAPS + BPRS Report Generator with Flask API
Automatically downloads OpenNeuro ds003944, processes EEG data, and serves clinical reports via HTTP.
"""

import os
import json
import threading
import pandas as pd
import numpy as np
import mne
from scipy.signal import welch
from flask import Flask, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer,AutoModelForSeq2SeqLM
import openneuro as on  # Correct import
app = Flask(__name__)

# ---------------- CONFIG ----------------
REMOTE_MODEL = "Mahalingam/DistilBart-Med-Summary"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 500
OUTPUT_DIR = "ds003944_local"
PATIENT_GROUP = "default_group"
DATASET_ID='ds003944'
# ---------------- PATHS ----------------
SAPS_JSON_PATH = os.path.join(OUTPUT_DIR, "phenotype", "saps.json")
SAPS_TSV_PATH = os.path.join(OUTPUT_DIR, "phenotype", "saps.tsv")
BPRS_JSON_PATH = os.path.join(OUTPUT_DIR, "phenotype", "bprs.json")
BPRS_TSV_PATH = os.path.join(OUTPUT_DIR, "phenotype", "bprs.tsv")
EEG_DIR = os.path.join(OUTPUT_DIR)  # EEG files will be inside subfolders after download

# ---------------- DOWNLOAD DATASET ----------------
@app.route("/")
@app.route("/download_dataset")
def download_dataset():
    """Download the OpenNeuro dataset if not already downloaded."""
    marker_file = os.path.join(OUTPUT_DIR, "dataset_description.json")

    if os.path.exists(marker_file):
        print(f"[INFO] Dataset {DATASET_ID} already exists in {OUTPUT_DIR}, skipping download.")
        return True

    print(f"[INFO] Downloading OpenNeuro dataset {DATASET_ID} to {OUTPUT_DIR}...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        on.download(dataset=DATASET_ID, target_dir=OUTPUT_DIR, keep_dir_structure=True)
        print(f"[INFO] Dataset {DATASET_ID} downloaded successfully to {OUTPUT_DIR}.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        return False
# ---------------- EEG FEATURE EXTRACTION ----------------
def extract_features_from_file(vhdr_file):
    try:
        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)
        raw.filter(1., 40., fir_design="firwin", verbose=False)
        data = raw.get_data()
        sfreq = raw.info["sfreq"]
        freqs, psd = welch(data, sfreq, nperseg=min(2048, data.shape[1]), axis=1)
        bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12),
                 "beta": (12, 30), "gamma": (30, 40)}
        features = {}
        total = np.trapz(psd, freqs, axis=-1).mean() + 1e-12
        for band, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            features[band] = float(np.mean(np.trapz(psd[:, mask], freqs[mask], axis=-1) / total))
        return features
    except Exception as e:
        print(f"[ERROR] {vhdr_file}: {e}")
        return None

# ---------------- LOAD SCALE SCORES ----------------
def load_scores(tsv_path, subj_id, json_path):
    try:
        with open(json_path, encoding="utf-8") as f:
            definitions = json.load(f)
        df = pd.read_csv(tsv_path, sep="\t", dtype=str)
        row = df[df["participant_id"] == subj_id]
        if row.empty:
            return None
        record = row.to_dict(orient="records")[0]
        mapped = {}
        for k, v in record.items():
            if k == "participant_id":
                continue
            q_info = definitions.get(k, {})
            levels = q_info.get("Levels", {}) if isinstance(q_info, dict) else {}
            desc = q_info.get("Description", "") if isinstance(q_info, dict) else ""
            display_value = str(v)
            meaning = ""
            try:
                if str(v).isdigit():
                    meaning = levels.get(str(int(v)), "")
                    if meaning:
                        display_value = f"{v} ({meaning})"
                else:
                    meaning = levels.get(str(v), "")
                    if meaning:
                        display_value = f"{v} ({meaning})"
            except Exception:
                pass
            mapped[k] = {"value": display_value, "desc": desc}
        return mapped
    except Exception as e:
        print(f"[ERROR] Loading scores {tsv_path}: {e}")
        return None

# ---------------- MODEL WRAPPER ----------------
def model_call(prompt, model, tokenizer, stage):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    kwargs = dict(**inputs, max_new_tokens=MAX_TOKENS, streamer=streamer)
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    result = ""
    print(f"\n=== {stage.upper()} ===\n")
    for new_text in streamer:
        print(new_text, end="", flush=True)
        result += new_text
    print("\n")
    return result.strip()

# ---------------- BUILD PROMPTS ----------------
def build_context(features, saps, bprs, group):
    def fmt(scale):
        return "\n".join(f"- {k}: {v['value']} â†’ {v['desc']}" for k, v in (scale or {}).items())
    return f"""
Patient Group: {group or 'unspecified'}

qEEG Features:
{json.dumps(features, indent=2)}

SAPS:
{fmt(saps)}

BPRS:
{fmt(bprs)}
"""

def build_stage_prompts(context):
    return {
        "Diagnosis": f"You are a clinical psychiatrist. Given this patient data:\n{context}\nList top 3 DSM-5 diagnoses with justification.",
        "Evidence": f"Context:\n{context}\nExplain EEG and clinical score evidence supporting each diagnosis.",
        "Risk": f"Context:\n{context}\nAssess patient risks (suicide, hospitalization, relapse, cognitive decline).",
        "Treatment": f"Context:\n{context}\nSuggest a treatment plan: medications, psychotherapy, neuromodulation, lifestyle, and social support.",
        "Summary": f"Context:\n{context}\nWrite a concise clinical summary paragraph suitable for a medical record."
    }

# ---------------- FLASK SERVER ----------------

@app.route("/generate_report/<subj_id>", methods=["GET"])
def generate_report(subj_id):
    download_dataset()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(REMOTE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        REMOTE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    # Find EEG file
    vhdr_file = None
    for root, _, files in os.walk(EEG_DIR):
        for f in files:
            if f.lower().endswith(".vhdr") and subj_id in f:
                vhdr_file = os.path.join(root, f)
                break
        if vhdr_file:
            break

    if not vhdr_file:
        return jsonify({"error": f"EEG file for {subj_id} not found"}), 404

    # Extract features and load scores
    features = extract_features_from_file(vhdr_file)
    saps = load_scores(SAPS_TSV_PATH, subj_id, SAPS_JSON_PATH)
    bprs = load_scores(BPRS_TSV_PATH, subj_id, BPRS_JSON_PATH)

    context = build_context(features, saps, bprs, PATIENT_GROUP)
    prompts = build_stage_prompts(context)

    report_parts = {}
    for stage, prompt in prompts.items():
        report_parts[stage] = model_call(prompt, model, tokenizer, stage)

    final_report = "\n\n".join(f"## {stage}\n{txt}" for stage, txt in report_parts.items())
    report_file = os.path.join(OUTPUT_DIR, f"{subj_id}_BioMed3B_Report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(final_report)

    return jsonify({"message": f"Report generated for {subj_id}", "report_file": report_file})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)