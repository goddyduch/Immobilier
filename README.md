# IA-Immobilier — Estimation Immobilière Hybride, Détection de Décotes & Simulateur Locatif

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18%20%2B%20TypeScript-61dafb.svg)](https://react.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PostGIS-336791.svg)](https://postgis.net/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ed.svg)](https://www.docker.com/)

Système complet d'**Automated Valuation Model (AVM)** appliqué au marché immobilier du Grand Poitiers (`929` annonces historiques enrichies et `21 436` actes notariés DVF 2023–2025).  
L'application estime la valeur de marché d'un bien, quantifie l'incertitude, explique le prix via **SHAP**, détecte les biens potentiellement **sous-évalués (décote)** ou **surévalués**, et intègre un **simulateur d'investissement locatif** complet ainsi qu'un **back-office de monitoring (Data & Prediction Drift)**.

---

## Résultats Clés (Évaluation Hors-Échantillon)

| Approche | $R^2$ | MAE | MdAPE (Erreur médiane) | Biens à $\le 15\%$ |
|---|---:|---:|---:|---:|
| **ML seul (`VotingRegressor` HGB + Huber + ExtraTrees)** | `0.823` | `±27 236 €` | `±11.3 %` | `60.7 %` |
| **AVM Comparables seul (DVF + annonces ajustées)** | `0.743` | `±33 362 €` | `±14.7 %` | `50.8 %` |
| **Modèle Hybride (`75 % ML + 25 % AVM Comparables`)** | **`0.823`** | `±27 268 €` | `±11.5 %` | **`61.2 %`** |

> L'étude scientifique complète (protocole anti-leakage, `KFold` vs `GroupKFold`, ablation des features, décomposition erreur modèle vs bruit humain) est disponible dans [docs/research_paper.md](docs/research_paper.md).

---

## Architecture du Projet

```text
ia-immobilier/
├── main.py                      # Point d'entrée CLI (update, features, train, predict, eval, improve, api, monitor)
├── docker-compose.yml           # Stack locale Mac (db PostGIS, api FastAPI, frontend React, worker cron)
├── docker-compose.prod.yml      # Surcouche production (Caddy HTTPS + Nginx + multi-workers)
├── docker/                      # Dockerfiles (API & Worker), Caddyfile et crontab
├── db/init.sql                  # Schéma PostgreSQL / PostGIS (quarantaine, runs, monitoring)
├── src/
│   ├── ingestion/               # Collecteurs (BienIci, Jinka, MoteurImmo), géocodage BAN, déduplication & Data Contracts
│   ├── ml/                      # Feature Engineering (48 vars), OpenCLIP (5 états), AVM, VotingRegressor, SHAP, MLflow
│   └── monitoring/              # Calcul PSI (Data Drift), Prediction Drift, complétude & alertes
├── backend/                     # API REST FastAPI (< 150 ms par estimation unitaire via ModelSnapshot en RAM)
├── frontend/                    # Interface React + TypeScript + Vite + Tailwind + Recharts
├── tests/                       # Suite pytest (Data Contracts, déduplication, cash-flow, API, monotonie ML)
├── docs/                        # Article de recherche, benchmarks CSV et graphiques PNG
└── models/                      # manifest.json (poids .joblib générés localement et gitignorés)
```

---

## Choix Technologiques (Pourquoi cette stack ?)

- **Ensemble `VotingRegressor` (`HistGradientBoosting` + `GradientBoosting Huber` + `ExtraTrees`)** plutôt que Deep Learning ou modèle linéaire :
  - Sur ~`1 000` biens tabulaires, les arbres boostés surclassent largement les modèles linéaires (`R² = 0.82` vs `0.39` pour `Ridge`).
  - `HistGradientBoosting` gère nativement les valeurs manquantes (`DPE`, `année`), `Huber` protège contre les annonces aux prix aberrants, et `ExtraTrees` réduit la variance.
- **Approche Hybride (`ML + AVM Comparables`)** : combine la capacité de généralisation du ML avec l'ancrage micro-local sur les ventes notariées DVF (décroissance temporelle) et annonces voisines ajustées par équipements (`parking`, `terrasse/jardin`, `ascenseur`, `DPE`).
- **Multimodal pragmatique (`Regex + Ollama JSON + OpenCLIP`)** : extraction des signaux texte et visuels (5 catégories d'état intérieur) mise en cache lors des batchs asynchrones, sans pénaliser la latence temps réel de l'API.
- **FastAPI + React + PostgreSQL/PostGIS** : découplage propre entre le worker batch et le serveur temps réel (`ModelSnapshot` immuable en RAM + reload atomique), requêtes spatiales natives, et UX fluide.
- **Docker Compose + Caddy** : même environnement du Mac local jusqu'au déploiement VPS sans engagement (HTTPS automatique Let's Encrypt).

---

## Fonctionnalités de l'Application V1

1. **Radar des décotes** : filtrage par ville, typologie, tranche de surface et quartier, bascule **Prix brut ↔ Coût total avec travaux**, filtrage automatique des biens trop incertains (`score_confiance < 45`).
2. **Fiche détaillée au clic** : galerie photos, prix affiché, quartier, description complète, fourchette `[Q10–Q90]`, jauge de confiance, facteurs **SHAP** lisibles et 5 comparables voisins.
3. **Vérificateur d'URL & Estimation rapide** : analyse d'une annonce via son lien (ou formulaire rapide avec autocomplétion adresse **API BAN**).
4. **Simulateur d'investissement locatif** : coût total du projet, travaux `€/m²` + alerte DPE `F/G`, loyers (Bas/Moyen/Haut), charges détaillées, financement, **Cash-flow mensuel** et **Rendement brut/net**.
5. **Analyse Notaires DVF & Négociation** : comparaison prix affiché (FAI) vs prix signé notaire (DVF) et calcul d'offre cible.
6. **Back-Office Monitoring (`/monitoring`)** : complétude par source, `PSI` *Data Drift*, *Prediction Drift* et file `offres_quarantaine`.

---

## Démarrage Rapide

### Option 1 — Docker Compose (recommandé)
```bash
cp .env.example .env
docker compose up -d --build
```
- **Frontend React** : http://localhost:5173
- **API FastAPI (Swagger)** : http://localhost:8000/docs

### Option 2 — Local Python (`.venv`)
```bash
# 1. Tests unitaires & audit IA
python -m pytest -v

# 2. Pipeline batch complet
python main.py features && python main.py train && python main.py predict && python main.py monitor

# 3. API FastAPI
python main.py api --reload
```

---

## Next Steps (Feuille de Route R&D et Produit)

1. **Élargissement géographique multi-villes** :
   - Remplacement de la cible `log(prix)` par une cible résiduelle $\log(\text{prix/m}^2 / \text{DVF}_{\text{local}})$ afin de mutualiser les élasticités (DPE, surface, travaux) sur plusieurs agglomérations.
   - Jointures spatiales PostGIS généralisées (`DVF` + `IRIS` + `BAN` France entière) pour ouvrir Niort, La Rochelle et Tours.
2. **Augmentation du volume de données** :
   - Réactivation maîtrisée de `src/ingestion/moteurimmo.py` (avec filtrage strict GPS/qualité) et accumulation historique multi-trimestres (`objectif 3 000–5 000 biens`).
3. **Calibration & nouveaux tests scientifiques** :
   - Implémentation de la **Conformalized Quantile Regression (CQR)** pour resserrer les fourchettes `[Q10–Q90]` (`44.3 %` actuellement) tout en garantissant `80 %` de couverture.
   - Benchmark **CatBoost** et **LightGBM (`objective="huber"`)**.
   - **Appariement annonces expirées ↔ actes notariés DVF** pour évaluer directement sur le *prix signé* et mesurer la variance inter-agents via modèle à effets mixtes.
