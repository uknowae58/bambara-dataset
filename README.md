# Bambara Dataset

Datasets pour l'entraînement de modèles IA en langue bambara.

## Sources

| Dataset | Description | Taille |
|---------|-------------|--------|
| oza75/bambara-lm-qa | Q&A en bambara | 824k |
| oza75/bambara-texts | Textes divers | 617k |
| oza75/bambara-mt | Traduction BM↔FR | ~5M |

## Structure

```
bambara-dataset/
├── raw/              # Données brutes
├── processed/        # Données traitées
├── scripts/          # Scripts de traitement
└── README.md
```

## Utilisation

```bash
# Télécharger les datasets
python scripts/download_data.py

# Nettoyer les données
python scripts/clean_data.py
```

## Licence

MIT
