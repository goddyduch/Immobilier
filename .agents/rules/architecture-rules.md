---
trigger: always_on
---

# Règles techniques communes — IA-Immobilier

- **Environnement Python (Mac)** : toujours utiliser `/Users/user/Downloads/IA-immobilier/.venv/bin/python`.
- **Base de données** : PostgreSQL (`localhost:5432`, base `immobilier`) + exports miroirs `csv/`.
- **Intégrité des données** : ne jamais mélanger `prix affiché` et `prix signé DVF` sans le mentionner explicitement ; ne jamais entraîner sans déduplication préalable.
- **Validation obligatoire** : aucune modification du pipeline ML ne doit être validée sans mesure comparative avant/après (`R²`, `MAE`, `MdAPE`, `GroupKFold`).
- **Respect des périmètres** : chaque agent édite uniquement les fichiers de son domaine ; pour des modifications transverses, coordonner via le coordinateur.
