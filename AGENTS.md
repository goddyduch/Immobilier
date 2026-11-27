# Équipe IA-Immobilier — Routage des Agents

Projet : Estimation du prix immobilier (Poitiers puis multi-villes) et détection fiable des biens sous-évalués (décote) ou surévalués.

## Règle de délégation

### Pôle Direction & Stratégie
- **Positionnement, benchmark concurrentiel, pricing, Go-to-Market** → `strategy-expert`
- **PMF, proposition de valeur, parcours différenciant, rétention** → `product-expert`
- **Scalabilité, risques tech/juridiques, Build vs Buy, moat technique** → `cto`
- **Arbitrage final, cap business, ROI, Go/No-Go stratégique** → `director`

### Pôle Exécution & Engineering
- **Priorisation, specs métier, UX fonctionnelle, Go/No-Go** → `product-manager`
- **Expériences ML, validation scientifique, intervalles conformes, bruit agents** → `research-engineer`
- **Scraping, géocodage, déduplication, PostgreSQL/PostGIS, Open Data** → `data-engineer`
- **Features, modèles ML, AVM comparables, LLM, OpenCLIP, SHAP** → `ai-engineer`
- **API FastAPI, logique métier backend, schémas Pydantic, tests API** → `software-engineer`
- **Dashboard Dash / Application React, dataviz, UX explications** → `frontend-engineer`
- **Docker, CI/CD, MLflow, monitoring drift, crons, RGPD** → `mlops-engineer`

## Workflow standard d'une nouvelle fonctionnalité
1. `product-manager` définit le besoin, les cas limites et les critères d'acceptation.
2. `research-engineer` valide l'approche statistique/métrique si la prédiction ou le scoring est impacté.
3. `data-engineer` et `ai-engineer` préparent les données, features et modèles.
4. `software-engineer` et `frontend-engineer` exposent et affichent le résultat.
5. `mlops-engineer` vérifie les tests, la non-régression et le monitoring avant mise en ligne.
