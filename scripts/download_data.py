#!/usr/bin/env python3
"""
Download and prepare Bambara datasets from HuggingFace
"""
from datasets import load_dataset
import os

OUTPUT_DIR = "./raw"

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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total = 0
    total += download_bambara_lm_qa()
    total += download_bambara_texts()
    total += download_bambara_mt()
    
    print(f"\n=== Total: {total} examples downloaded ===")

if __name__ == "__main__":
    main()
