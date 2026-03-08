# Bambara Dataset

Datasets pour l'entraînement de modèles IA en langue bambara (Bambara–French parallel data, Q&A).

## Prérequis

```bash
pip install -r requirements.txt
# Pour l'entraînement Unsloth : pip install unsloth trl
```

## Préparer les données pour Unsloth (GPU)

Les données sont préparées **par objectif** (chaque dataset a un usage spécifique, voir les README dans chaque dossier) :

```bash
# Traduction (MT) : corpus parallèles de qualité
python scripts/prepare_unsloth_data.py --goal mt

# Q&A / instruction-following : bambara-lm-qa (nécessite pip install pyarrow)
python scripts/prepare_unsloth_data.py --goal qa
```

**Sortie :**
- MT : `data/unsloth_mt_train.jsonl`, `data/unsloth_mt_eval.jsonl`
- Q&A : `data/unsloth_qa_train.jsonl`, `data/unsloth_qa_eval.jsonl`

Options : `--output-dir`, `--val-ratio`, `--max-length`.

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files={
    "train": "data/unsloth_mt_train.jsonl",
    "eval": "data/unsloth_mt_eval.jsonl"
})
# Colonnes : instruction, input, output (format Alpaca)
```

## Entraînement Unsloth

```bash
pip install unsloth datasets trl
python scripts/train_unsloth.py
```

Options : `--model`, `--output-dir`, `--max-seq-length`, `--batch-size`, `--no-4bit`.

## Sources par objectif

| Objectif | Sources | Exemples |
|----------|---------|----------|
| **mt** (traduction) | bambara-french-parallel, bamadaba, transcriptions, bayelemabaga, bambara-french/train.json, bamacours | ~157k paires → ~150k train |
| **qa** (Q&A) | bambara-lm-qa (sft ou default, parquet) | ~173k |

## Structure du dépôt

```
bambara-dataset/
├── data/                             # Généré par prepare_unsloth_data.py
│   ├── unsloth_mt_train.jsonl        # ~150k exemples (traduction)
│   ├── unsloth_mt_eval.jsonl         # ~8k exemples
│   ├── unsloth_qa_train.jsonl        # Q&A (si pyarrow installé)
│   └── unsloth_qa_eval.jsonl
├── bambara-french-parallel-dataset/  # Corpus parallèle Bambara–Français
├── bambara-french/                   # Paires supplémentaires (train.json)
├── bambara-french_sentence-pairs/    # Paires NLLB/OPUS (référence uniquement)
├── bambara-lm-qa/                    # Q&A en bambara (parquet)
├── bayelemabaga-bambara-dataset/     # Corpus bayelemabaga + guide bamacours
├── Transcript_Bambara_French/        # Transcriptions (parquet)
├── scripts/
│   ├── prepare_unsloth_data.py       # Prépare les données → JSONL Alpaca
│   ├── train_unsloth.py              # Entraînement Unsloth sur GPU
│   └── clean_data.py
├── requirements.txt
└── README.md
```

## Licence

MIT
