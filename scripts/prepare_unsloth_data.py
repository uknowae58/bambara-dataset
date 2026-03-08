#!/usr/bin/env python3
"""
Prepare Bambara data for Unsloth training, organized by GOAL.

Each dataset has a specific purpose (see README in each folder). Data is joined
by goal, not blindly merged.

Goals:
  mt   Machine translation (Bambara ↔ French). High-quality parallel corpora.
  qa   Q&A / instruction-following in Bambara. bambara-lm-qa (requires pyarrow).
  all  Generate both mt and qa separately.

Output format: Alpaca-style JSONL
  {"instruction": "...", "input": "...", "output": "...", "source": "..."}

Usage:
  python scripts/prepare_unsloth_data.py --goal mt
  python scripts/prepare_unsloth_data.py --goal qa
  python scripts/prepare_unsloth_data.py --goal all
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Paths ---
BAMBARA_FRENCH_PARALLEL = REPO_ROOT / "bambara-french-parallel-dataset" / "bambara-french-parallel.json"
BAMADABA_CSV            = REPO_ROOT / "bambara-french-parallel-dataset" / "bamadaba-parallel.csv"
TRANSCRIPTIONS_CSV      = REPO_ROOT / "bambara-french-parallel-dataset" / "transcriptions.csv"
BAYELMABAGA_TSV         = REPO_ROOT / "bayelemabaga-bambara-dataset" / "bayelemabaga" / "train" / "train.tsv"
BAMBARA_FRENCH_TRAIN    = REPO_ROOT / "bambara-french" / "train.json"
BAMACOURS_GUIDE         = REPO_ROOT / "bayelemabaga-bambara-dataset" / "bamacours-apprendre-bambara-guide.json"
TRANSCRIPT_PARQUET      = REPO_ROOT / "Transcript_Bambara_French" / "data" / "train-00000-of-00001.parquet"
BAMBARA_LM_QA_SFT       = REPO_ROOT / "bambara-lm-qa" / "sft" / "train-00000-of-00001.parquet"
BAMBARA_LM_QA_DEFAULT   = REPO_ROOT / "bambara-lm-qa" / "data" / "train-00000-of-00001.parquet"


def normalize(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def pair_key(bambara: str, french: str) -> str:
    return hashlib.sha256(f"{normalize(bambara)}|||{normalize(french)}".encode("utf-8")).hexdigest()


# ============== GOAL: MT (Machine Translation) ==============

def load_mt_pairs() -> list[tuple[str, str, str]]:
    """Returns (bambara, french, source) from all curated parallel corpora."""
    pairs: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def add(b: str, f: str, src: str) -> None:
        b, f = normalize(b), normalize(f)
        if len(b) < 2 or len(f) < 2:
            return
        k = pair_key(b, f)
        if k in seen:
            return
        seen.add(k)
        pairs.append((b, f, src))

    if BAMBARA_FRENCH_PARALLEL.exists():
        with open(BAMBARA_FRENCH_PARALLEL, "r", encoding="utf-8") as f:
            for item in json.load(f):
                if isinstance(item, dict):
                    b, fr = item.get("bambara", ""), item.get("french", "")
                    if b and fr:
                        add(b, fr, "bambara_french_parallel")

    if BAMADABA_CSV.exists():
        with open(BAMADABA_CSV, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                fr = row.get("french", row.get("French", ""))
                b  = row.get("bambara", row.get("Bambara", ""))
                if fr and b:
                    add(b, fr, "bamadaba")

    if TRANSCRIPTIONS_CSV.exists():
        with open(TRANSCRIPTIONS_CSV, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                b  = row.get("bambara", row.get("Bambara", ""))
                fr = row.get("french", row.get("French", ""))
                if b and fr:
                    add(b, fr, "transcriptions")

    if BAYELMABAGA_TSV.exists():
        with open(BAYELMABAGA_TSV, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    add(parts[0], parts[1], "bayelemabaga")

    if BAMBARA_FRENCH_TRAIN.exists():
        with open(BAMBARA_FRENCH_TRAIN, "r", encoding="utf-8") as f:
            for item in json.load(f):
                if isinstance(item, dict):
                    b, fr = item.get("bambara", ""), item.get("french", "")
                    if b and fr:
                        add(b, fr, "bambara_french_train")

    if TRANSCRIPT_PARQUET.exists():
        _load_parquet_pairs(TRANSCRIPT_PARQUET, add, "transcript_parquet")

    return pairs


def load_bamacours_mt() -> list[dict]:
    """Bamacours guide as MT instruction examples."""
    out = []
    if not BAMACOURS_GUIDE.exists():
        return out
    with open(BAMACOURS_GUIDE, "r", encoding="utf-8") as f:
        data = json.load(f)
    for p in data.get("phrases_essentielles", []):
        b, fr = normalize(p.get("bambara", "")), normalize(p.get("french", ""))
        if b and fr:
            out.append({"instruction": "Traduis en bambara.", "input": fr, "output": b, "source": "bamacours_phrases"})
            out.append({"instruction": "Traduis en français.", "input": b, "output": fr, "source": "bamacours_phrases"})
    for key in ("voyelles", "consonnes_frequentes"):
        for item in data.get("alphabet", {}).get(key, []):
            ex, sens = normalize(item.get("exemple", "")), normalize(item.get("sens", ""))
            if ex and sens:
                out.append({"instruction": "Comment dit-on en bambara ?", "input": sens, "output": ex, "source": "bamacours_alphabet"})
    for gr in data.get("grammaire", {}).get("exemples_ordre", []):
        b, fr = normalize(gr.get("bambara", "")), normalize(gr.get("french", ""))
        if b and fr:
            out.append({"instruction": "Traduis en bambara.", "input": fr, "output": b, "source": "bamacours_grammaire"})
            out.append({"instruction": "Traduis en français.", "input": b, "output": fr, "source": "bamacours_grammaire"})
    for item in data.get("bambara_vs_soninke", {}).get("mini_lexique", []):
        fr, b = normalize(item.get("francais", "")), normalize(item.get("bambara", ""))
        if b and fr:
            out.append({"instruction": "Traduis en bambara.", "input": fr, "output": b, "source": "bamacours_lexique"})
            out.append({"instruction": "Traduis en français.", "input": b, "output": fr, "source": "bamacours_lexique"})
    return out


def to_mt_rows(pairs: list[tuple[str, str, str]]) -> list[dict]:
    rows = []
    for b, fr, src in pairs:
        rows.append({"instruction": "Traduis en bambara.", "input": fr, "output": b, "source": src})
        rows.append({"instruction": "Traduis en français.", "input": b, "output": fr, "source": src})
    return rows


# ============== GOAL: Q&A (Instruction-following) ==============

def load_qa_rows() -> list[dict]:
    """Load bambara-lm-qa. Prefers SFT config (already Alpaca-style)."""
    rows = []
    if BAMBARA_LM_QA_SFT.exists():
        for r in _read_parquet_rows(BAMBARA_LM_QA_SFT):
            inst = normalize(r.get("instruction", ""))
            inp  = normalize(r.get("input", ""))
            out  = normalize(r.get("output", ""))
            if inst and out:
                rows.append({
                    "instruction": inst,
                    "input": inp,
                    "output": out,
                    "source": r.get("source_dataset", "bambara_lm_qa_sft"),
                })
    elif BAMBARA_LM_QA_DEFAULT.exists():
        for r in _read_parquet_rows(BAMBARA_LM_QA_DEFAULT):
            q = normalize(r.get("question", ""))
            a = normalize(r.get("answer", ""))
            if q and a:
                rows.append({
                    "instruction": "Réponds à la question en bambara.",
                    "input": q,
                    "output": a,
                    "source": "bambara_lm_qa",
                })
    return rows


# ============== Helpers ==============

def _read_parquet_rows(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        cols = t.column_names
        return [dict(zip(cols, [t.column(c)[i].as_py() for c in cols])) for i in range(len(t))]
    except ImportError:
        try:
            from datasets import load_dataset
            ds = load_dataset("parquet", data_files=str(path), split="train")
            return [dict(zip(ds.column_names, row)) for row in ds]
        except ImportError:
            print(f"  [skip] {path.name}: install pyarrow or datasets to read parquet")
            return []
    except Exception as e:
        print(f"  [skip] {path.name}: {e}")
        return []


def _load_parquet_pairs(path: Path, add_fn, source: str) -> None:
    for r in _read_parquet_rows(path):
        b  = str(r.get("bambara", r.get("Bambara", r.get("bam", ""))))
        fr = str(r.get("french",  r.get("French",  r.get("fr", ""))))
        if b and fr:
            add_fn(b, fr, source)


def _split_and_write(rows: list[dict], output_dir: Path, prefix: str, val_ratio: float, seed: int, max_length: int) -> None:
    if max_length > 0:
        rows = [r for r in rows if len(r.get("input", "")) <= max_length and len(r.get("output", "")) <= max_length]
    random.seed(seed)
    random.shuffle(rows)
    n_val = max(0, int(len(rows) * val_ratio))
    train_rows = rows[:-n_val] if n_val else rows
    val_rows   = rows[-n_val:] if n_val else []
    train_path = output_dir / f"unsloth_{prefix}_train.jsonl"
    val_path   = output_dir / f"unsloth_{prefix}_eval.jsonl"
    for p, data in [(train_path, train_rows), (val_path, val_rows)]:
        with open(p, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {prefix}: {len(train_rows)} train, {len(val_rows)} eval -> {train_path.name}, {val_path.name}")


def main():
    ap = argparse.ArgumentParser(description="Prepare Bambara data for Unsloth training")
    ap.add_argument("--goal", choices=["mt", "qa", "all"], default="mt",
                    help="mt=translation, qa=Q&A/instruction-following, all=both")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: data/)")
    ap.add_argument("--val-ratio", type=float, default=0.05,
                    help="Fraction for evaluation split (default: 0.05)")
    ap.add_argument("--max-length", type=int, default=0,
                    help="Max characters per input/output field (0 = no filter)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (REPO_ROOT / "data").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    goals = ["mt", "qa"] if args.goal == "all" else [args.goal]

    for goal in goals:
        if goal == "mt":
            print("Loading MT (machine translation) sources...")
            pairs = load_mt_pairs()
            guide = load_bamacours_mt()
            rows  = to_mt_rows(pairs) + guide
            print(f"  {len(pairs)} pairs + {len(guide)} bamacours examples -> {len(rows)} rows")
            _split_and_write(rows, output_dir, "mt", args.val_ratio, args.seed, args.max_length)

        elif goal == "qa":
            print("Loading Q&A (bambara-lm-qa) sources...")
            rows = load_qa_rows()
            print(f"  {len(rows)} rows")
            if rows:
                _split_and_write(rows, output_dir, "qa", args.val_ratio, args.seed, args.max_length)
            else:
                print("  No data found. Run: pip install pyarrow")

    print(f"\nDone. Load in Unsloth:")
    print(f'  dataset = load_dataset("json", data_files={{"train": "data/unsloth_<goal>_train.jsonl", "eval": "..."}})')


if __name__ == "__main__":
    main()
