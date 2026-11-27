# Rapport Stratégique, Produit & Technique — « JustePrix Immo » (Side-Project PropTech)

*Synthèse du Comité de Direction (Marc — Stratégie, Sarah — Produit, Karim — CTO/Data, Chloé — Brand/Marketing, Thomas — Business Dev) basée sur le socle technique réel (`R² = 0,823`, `MdAPE = ±11,5 %` sur `929` annonces et `21 436` actes DVF).*

---

## 1. Analyse Concurrentielle

Le marché français est fragmenté entre **3 silos** :

| Acteur | Positionnement | Tarifs publics | Forces | Angles morts (Notre opportunité) |
|---|---|---|---|---|
| **Jinka** | Agrégateur d'alertes mobiles | Freemium (~10 €/mois) | Vitesse de notification | Zéro estimation IA, zéro travaux, zéro DVF |
| **MoteurImmo** | Sourcing massif (55+ portails) | 9–39 €/mois | Exhaustivité, historique prix + DVF | Interface dense, pas de SHAP, pas de chiffrage travaux ni cash-flow |
| **Castorus** | Historique baisses de prix | 23–30 €/mois | Décote psychologique (jours en ligne) | Ignore si le prix initial était déjà surévalué |
| **LyBox** | Sourcing + Simulation fiscale | 11–59 €/mois | Simulation LMNP/SCI poussée | Estimation €/m² basique, pas d'AVM hybride ni SHAP |
| **Horiz.io** | Simulateur patrimonial & extension | 8–20 €/mois | UX extension portails, dossiers bancaires | Aucune détection de décotes IA ni comparables DVF |
| **MeilleursAgents** | Estimateur grand public | Gratuit B2C / Leads B2B | Notoriété, carte des prix | Moyenne €/m² rigide, aveugle à la dégressivité m² et aux travaux |
| **PriceHubble** | IA prédictive B2B (banques/réseaux) | Sur devis B2B | Précision & impact DPE | Boîte noire inaccessible aux particuliers et petits investisseurs |

---

## 2. Innovation & Différenciation (« Juste Prix & Décote Nette »)

Notre avantage asymétrique (zéro budget infra, 0 VLLM) repose sur **4 innovations combinées** :
1. **De la décote psychologique à la Décote Nette Intrinsèque** : croisement d'un `VotingRegressor` (`HistGB` + `Huber` + `ExtraTrees`) et d'un `AVM Comparables` ajusté par équipements (`parking ±8 000 €`, `terrasse ±5 %`, `ascenseur ±4 %`, `DPE ±3 %/classe`), intégrant le coût des travaux (grille 0–4 + DPE) et la dégressivité réelle du €/m² selon la surface.
2. **IA « Glass-Box » (Transparence totale)** : chaque estimation affiche la fourchette `[Q10–Q90]`, le **score de confiance (/100)**, les **impacts SHAP en euros** traduits en langage simple, et les **5 ventes comparables DVF / marché**.
3. **Preuve de négociation notariée (FAI vs Signé DVF)** : mesure directe de la surcote d'affichage locale (ex. à Poitiers : `+7,4 %` affiché vs notaire sur les appartements) pour chiffrer une offre prudente / conseillée / offensive.
4. **Flux décisionnel continu** : *Repérer / Coller URL → Vérifier le Juste Prix → Chiffrer travaux & négo → Simuler le cash-flow*.

---

## 3. Cibles, Personas & Use Cases

### Persona 1 — « Julien & Clara », Jeune Ménage Primo-Accédant (Tech / Rationnel)
- **Profil** : 29–34 ans, cadres/tech, achètent leur résidence principale (T3/T4, budget 220–450 k€).
- **Douleur** : Peur de surpayer dans un marché opaque ; ne font pas confiance au « prix au m² moyen » de MeilleursAgents ni au discours commercial de l'agent.
- **Use Case** : Avant de faire une offre, Julien colle le lien de l'annonce dans **Analyser une URL**. En 1 seconde, il voit que le bien affiché à `265 000 €` est estimé à `238 000 €` (`+11,3 %` de surcote) à cause du DPE E (`-9 500 €`) et de l'absence de parking (`-8 000 €`), avec 5 ventes DVF à 300 m. Il télécharge les arguments pour négocier `-20 000 €`.
- **Offre adaptée** : **Pass Recherche 3 Mois** (paiement unique sans abonnement récurrent piège).

### Persona 2 — « Karim », Investisseur Locatif Exigeant (1 à 5 biens)
- **Profil** : 35–48 ans, cherche du rendement après travaux (studios, T2, immeubles, colocations) et fuit les passoires DPE `F/G` non chiffrées.
- **Douleur** : Perd 10h/semaine à croiser MoteurImmo + DVF + Excel/Horiz.io pour vérifier si une annonce « pas chère » n'est pas un gouffre à travaux.
- **Use Case** : Le matin, il consulte le **Radar des décotes nettes** (non exhaustif, haute précision `score_confiance >= 60`, bascule *Avec travaux*). Il repère un T2 à `-14 %` sous sa valeur rénovée, clique pour vérifier SHAP + photos, puis clique sur **Simuler l'investissement** pour valider un cash-flow positif à `+110 €/mois` après crédit et charges.
- **Offre adaptée** : **Pass Investisseur (mensuel sans engagement)**.

---

## 4. Stratégie Villes : Paris vs Ville Locale (Poitiers) vs Villes Moyennes

**Arbitrage du Comité (Karim + Marc + Thomas) : Hybride Séquentiel — NE PAS attaquer Paris en V1 !**

| Critère | Ville Locale seule (Poitiers) | **Paris (V1 ?)** | **Stratégie Recommandée : Poitiers → Villes Moyennes → Paris (V2)** |
|---|---|---|---|
| **Volume de données** | Faible (`~930` annonces, `21k` DVF) | Massif (`>150k` DVF, milliers d'annonces) | Capitalise sur Poitiers déjà prêt (`R²=0,823`) puis ouvre des marchés proches |
| **Concurrence** | Quasi nulle sur l'IA explicable | Saturée (*MeilleursAgents*, *Castorus*, *PriceHubble* ultra-précis rue par rue) | Évite la guerre frontale à Paris avec zéro budget marketing |
| **Sensibilité micro-locale** | Quartiers homogènes | **Extrême** : à Paris, deux immeubles face à face varient de ±20 % (étage, cour/rue, pierre de taille, sectorisation) | À Paris, sans données très fines, une erreur de `11 %` = `±80 000 €` (perte de crédibilité) |
| **Adéquation cibles** | Fort rendement locatif (étudiants), ticket accessible | Beaucoup de primo-accédants (fort potentiel pour le Pass 3 mois !) | **V1** : Poitiers + **Tours, La Rochelle, Angers, Niort** (fort vivier primo-accédants + investisseurs). **V2** : Paris / Lyon une fois le modèle résiduel + CQR calibrés. |

---

## 5. Business Model & Grille de Pricing

| Offre | Prix | Engagement | Contenu |
|---|---|---|---|
| **Découverte (Gratuit)** | **0 €** | Aucun | **3 analyses URL / mois**, estimation hybride, fourchette `[Q10–Q90]`, top 3 facteurs SHAP. |
| **Pass Recherche Primo-Accédant** | **29 € / 3 mois** *(soit ~9,60 €/mois, paiement unique)* | **Aucun renouvellement auto** | Analyses URL **illimitées** pendant 90 jours, 5 comparables DVF complets, simulateur de négociation + arguments PDF, chiffrage travaux. |
| **Pass Investisseur (Prosumer)** | **24 € / mois** *(ou 199 €/an)* | **Mensuel sans engagement** | Tout le Pass Recherche + **Radar des décotes nettes** (haute précision, filtre travaux & DPE `F/G`), **Simulateur Cash-Flow & Rendement complet**, alertes pépites. |

*(Économie unitaire : coût serveur Hetzner/OVH ~6–8 €/mois → rentable dès le **1er abonné Investisseur** ou **1er Pass 3 mois** !)*

---

## 6. Identité Visuelle & Positionnement de Marque (Chloé)

- **Nom retenu** : **JustePrix Immo** *(ou **ClairImmo** / **JusteMètre**)* — direct, mémorisable, centré sur la promesse de transparence.
- **Tagline** : *« Arrêtez de surpayer. Voyez le vrai prix, les travaux et la marge de négo en 30 secondes. »*
- **Identité visuelle** :
  - **Palette** : Fond *Slate Dark* (`#020617` / `#0F172A` pour le côté cockpit analytique pro), **Vert Émeraude** (`#10B981` = Décote / Cash-flow positif), **Bleu Ciel** (`#38BDF8` = Confiance / Info DVF), **Ambre** (`#F59E0B` = Travaux / Prudence), **Corail** (`#F43F5E` = Surévalué).
  - **Concept Logo** : Une maison stylisée dont le toit forme une **flèche de jauge / curseur d'équilibre** pointant vers le juste prix (évocation visuelle immédiate de la justesse et de la décote).
  - **Ton éditorial** : Factuel, transparent (« Glass-Box »), pédagogique, jamais vendeur de rêve.

---

## 7. Maquette Produit (Parcours Utilisateur en 5 Écrans)

1. **Écran 1 — Radar des Décotes Nettes (`/`)** *(Pass Investisseur)* :
   - Filtres : Ville, Type, Tranche m², Quartier, Masquer incertains (`C < 45`), **Bascule Prix brut ↔ Coût total avec travaux**.
   - KPIs + Tableau/Carte des pépites triées par `decote_nette_pct`.
2. **Écran 2 — Fiche Décisionnelle au clic (Drawer)** :
   - Photos, Prix affiché, Quartier, Description complète, Fourchette `[Q10–Q90]`, Jauge Confiance `/100`, Graphique **SHAP en €**, 5 comparables DVF/marché, boutons **Simuler Cash-flow** et **Préparer Négo**.
3. **Écran 3 — Analyser une Annonce par URL (`/verifier`)** *(Produit d'appel & Pass 3 mois)* :
   - Champ URL (lookup DB ou extraction légère) + Formulaire rapide autocomplété **API BAN**.
4. **Écran 4 — Simulateur Locatif & Cash-Flow (`/simulateur`)** :
   - Projet (FAI, notaire, travaux `€/m²`, alerte DPE F/G), Loyers (Bas/Moyen/Haut), Charges détaillées, Financement → Cash-flow mensuel, Taux de couverture, Rendement brut/net.
5. **Écran 5 — Observatoire DVF & Négociation (`/marche-dvf`)** :
   - Prix affiché vs signé notaire DVF par taille de logement + Offre Prudente / Conseillée / Offensive.

---

## 8. Maquette Technique : Comment Mettre en Place le Radar Non-Exhaustif à Coût Quasi Nul ?

### A. Stratégie Data « Low-Cost & Légale » (pour le Radar non-exhaustif)
On ne scrape **pas** 1 500 sites en temps réel :
1. **Sources ciblées propres** : 2 à 3 flux structurés à fort taux de GPS/DPE (API JSON **Bien'ici** + flux **Jinka** headless) + référentiels **100 % Open Data** gratuits (`DVF`, `BAN`, `IRIS`, `DPE ADEME`, `Carte des Loyers`).
2. **Sélectivité assumée ("50 % des pépites, 95 % de précision")** : on écarte les annonces sans GPS ou sans surface fiable via nos *Data Contracts* (`offres_quarantaine`), et on n'affiche dans le Radar que `score_confiance >= 45` (badge *Opportunité forte* si `>= 60`).
3. **Légalité en SaaS commercial** : pour les annonces issues de portails tiers, afficher les **indicateurs analytiques calculés** (estimation, décote, SHAP, comparables DVF) avec un **lien sortant vers l'annonce source**, sans réhéberger commercialement les photos/textes tiers (et privilégier à terme une **Extension Chrome "Bring Your Own Link"** côté navigateur pour l'Analyse URL).

### B. Architecture Technique & Zéro Coût Infra
- **Stack** : `FastAPI` (`< 150 ms` via `ModelSnapshot` en RAM) + `React/Vite` + `PostgreSQL/PostGIS` + `Docker Compose` + `Caddy HTTPS`.
- **Zéro VLLM coûteux** : `Ollama` local (ou fallback heuristique déterministe) + `OpenCLIP` en batch asynchrone mis en cache JSON.
- **Automatisation** : `Worker Supercronic` (ingestion toutes les 3h, réentraînement hebdomadaire avec Gate de non-régression, audit features mensuel, monitoring `PSI`).

---

## 9. Changements à Réaliser par Rapport à l'Existant

| Domaine | État actuel (V1 Poitiers) | Changement à faire pour lancer commercialement | Priorité |
|---|---|---|---|
| **ML & Incertitude** | Cible `log(prix)` mono-ville ; `[Q10–Q90]` non calibré (`44,3 %` de largeur) | Passer à la **cible résiduelle** $\log(\text{pm2}/\text{DVF}_{\text{local}})$ + **Conformal Prediction (CQR)** (`80 %` garanti, largeur réduite) | **P0** |
| **Multi-villes** | Coordonnées & IRIS Poitiers codés en dur dans `features.py` | Tables nationales **PostGIS** (`DVF`, `IRIS`, `BAN`, `Loyers`) sans charger tout DVF en RAM des workers API | **P0** |
| **Analyse URL** | Lookup DB + extraction BienIci uniquement | Ajouter **Extension Chrome / Bookmarklet** (lit la page dans le navigateur de l'utilisateur → `/v1/estimate`) | **P1** |
| **Produit & Monétisation** | Aucun compte utilisateur ni paiement | Ajouter Auth légère + **Stripe Checkout** (Pass 3 mois 29 € & Abonnement 24 €/mois) + Export PDF Négo/Banque | **P1** |
| **Juridique** | Photos/descriptions affichées directement dans le Drawer | Mode SaaS public : rediriger vers l'annonce source (garder photos uniquement en usage privé/extension) + CGU | **P0** |

---

## 10. Roadmap Précise Semaine par Semaine (8 Semaines vers la V1 Rentable)

| Semaine | Objectif principal | Livrables concrets |
|---|---|---|
| **S1** | **Calibration & CQR** | Implémenter la Conformal Prediction (`CQR`) sur `[Q10–Q90]` + tester `LightGBM Huber` / `CatBoost` en `GroupKFold`. |
| **S2** | **Modèle Résiduel Multi-Villes** | Basculer sur la cible résiduelle $\log(\text{pm2}/\text{DVF})$ + dé-hardcoder Poitiers vers requêtes spatiales **PostGIS**. |
| **S3** | **Ouverture Bassin Ouest (4 villes)** | Charger DVF + IRIS + annonces pour **Tours, La Rochelle, Angers, Niort** (validation `MdAPE <= 12,5 %` par ville). |
| **S4** | **Extension Navigateur + Check URL** | Créer la mini-extension Chrome (*Analyse URL en 1 clic* sur SeLoger/Leboncoin/Bien'ici) connectée à `POST /v1/estimate`. |
| **S5** | **Branding, Landing & Export PDF** | Appliquer l'identité *JustePrix Immo*, créer la page d'accueil (promesse + démo gratuite) et l'export PDF *Dossier Négo / Banque*. |
| **S6** | **Auth, Quotas & Stripe** | Intégrer comptes utilisateurs, quota gratuit (`3/mois`), **Pass 3 mois (29 €)** et **Pass Investisseur (24 €/mois)**. |
| **S7** | **Déploiement VPS & Monitoring** | Déployer sur VPS (`Docker Compose` + `Caddy HTTPS` + sauvegardes `pg_dump`) et vérifier les alertes `PSI` / latence. |
| **S8** | **Lancement Bêta & Acquisition** | Ouvrir à 30 bêta-testeurs (primo-accédants + investisseurs locaux), publier le baromètre *« Où négocier à Poitiers/Tours/La Rochelle ? »* et itérer. |
