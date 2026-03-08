#!/usr/bin/env python3
"""
Download and prepare Bambara datasets from HuggingFace and Kaggle
"""
from datasets import load_dataset
import kagglehub
import os
import shutil

OUTPUT_DIR = "./raw"
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def download_bambara_lm_qa():
    """Download bambara-lm-qa dataset"""
    print("Downloading oza75/bambara-lm-qa...")
    ds = load_dataset("oza75/bambara-lm-qa", split="train")
    output_path = f"{OUTPUT_DIR}/bambara-lm-qa.jsonl"
    ds.to_json(output_path)
    print(f"Saved to {output_path} ({len(ds)} examples)")
    return len(ds)

def download_bambara_texts():
    """Download bambara-texts dataset (may require HF token)"""
    print("Downloading oza75/bambara-texts...")
    try:
        ds = load_dataset("oza75/bambara-texts", split="train")
        output_path = f"{OUTPUT_DIR}/bambara-texts.jsonl"
        ds.to_json(output_path)
        print(f"Saved to {output_path} ({len(ds)} examples)")
        return len(ds)
    except Exception as e:
        print(f"Error: {e}")
        print("This dataset may require authentication. Run: huggingface-cli login")
        return 0

def download_bambara_mt():
    """Download bambara-mt translation dataset"""
    print("Downloading oza75/bambara-mt...")
    try:
        ds = load_dataset("oza75/bambara-mt", split="train")
        output_path = f"{OUTPUT_DIR}/bambara-mt.jsonl"
        ds.to_json(output_path)
        print(f"Saved to {output_path} ({len(ds)} examples)")
        return len(ds)
    except Exception as e:
        print(f"Error: {e}")
        return 0

def download_bambara_french_kaggle():
    """Download bambara-french-parallel-dataset from Kaggle and copy to workspace"""
    print("Downloading ozaresearch1/bambara-french-parallel-dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("ozaresearch1/bambara-french-parallel-dataset")
        print(f"Path to dataset files: {path}")
        # Copy to workspace
        dest = os.path.join(WORKSPACE_ROOT, "bambara-french-parallel-dataset")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(path, dest)
        print(f"Copied to {dest}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure kagglehub is installed (pip install kagglehub) and Kaggle API is configured.")
        return 0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total = 0
    total += download_bambara_lm_qa()
    total += download_bambara_texts()
    total += download_bambara_mt()
    download_bambara_french_kaggle()
    
    print(f"\n=== Total: {total} examples downloaded ===")

if __name__ == "__main__":
    main()
