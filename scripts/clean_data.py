#!/usr/bin/env python3
"""
Clean and format Bambara datasets for training
"""
import json
import os
from pathlib import Path

INPUT_DIR = "./raw"
OUTPUT_DIR = "./processed"

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save to JSONL format"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove very short texts
    if len(text.strip()) < 3:
        return ""
    return text.strip()

def process_qa(data):
    """Process Q&A dataset"""
    processed = []
    for item in data:
        question = clean_text(item.get('question', ''))
        answer = clean_text(item.get('answer', ''))
        
        if question and answer:
            processed.append({
                'question': question,
                'answer': answer,
                'source': 'bambara-lm-qa'
            })
    return processed

def process_texts(data):
    """Process texts dataset"""
    processed = []
    for item in data:
        text = clean_text(item.get('text', ''))
        if text:
            processed.append({
                'text': text,
                'source': 'bambara-texts'
            })
    return processed

def process_mt(data):
    """Process translation dataset"""
    processed = []
    for item in data:
        bambara = clean_text(item.get('bambara', ''))
        french = clean_text(item.get('french', ''))
        
        if bambara and french:
            processed.append({
                'bambara': bambara,
                'french': french,
                'source': 'bambara-mt'
            })
    return processed

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each dataset
    qa_file = f"{INPUT_DIR}/bambara-lm-qa.jsonl"
    if os.path.exists(qa_file):
        print("Processing Q&A dataset...")
        data = load_jsonl(qa_file)
        processed = process_qa(data)
        save_jsonl(processed, f"{OUTPUT_DIR}/bambara-qa.jsonl")
        print(f"  → {len(processed)} examples")
    
    texts_file = f"{INPUT_DIR}/bambara-texts.jsonl"
    if os.path.exists(texts_file):
        print("Processing texts dataset...")
        data = load_jsonl(texts_file)
        processed = process_texts(data)
        save_jsonl(processed, f"{OUTPUT_DIR}/bambara-texts.jsonl")
        print(f"  → {len(processed)} examples")
    
    mt_file = f"{INPUT_DIR}/bambara-mt.jsonl"
    if os.path.exists(mt_file):
        print("Processing translation dataset...")
        data = load_jsonl(mt_file)
        processed = process_mt(data)
        save_jsonl(processed, f"{OUTPUT_DIR}/bambara-mt.jsonl")
        print(f"  → {len(processed)} examples")
    
    print("\n=== Processing complete ===")

if __name__ == "__main__":
    main()
