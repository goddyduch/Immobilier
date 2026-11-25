"""
Deduplication.py — Détection et suppression des annonces en double dans PostgreSQL & CSV miroirs.

Stratégie :
  1. Dédoublonne par `id` exact.
  2. Dédoublonne par signature métier `(prix_arrondi, surface_arrondie, code_postal, type_bien)`
     afin d'éliminer les annonces identiques présentes sur plusieurs portails (ex: Bien'ici + MoteurImmo).
  3. Pour chaque groupe de doublons, conserve la fiche la plus complète :
     - Bonus source : `bienici` > `jinka` > `moteurimmo`
     - Bonus complétude : GPS (`latitude`/`longitude`), `dpe`, `annee_construction`, `photos_urls`, `url`, longueur de `description`.

Usage :
  python Deduplication.py             # Nettoie offres_historique_brut & offres_recentes_brut
  python Deduplication.py --dry-run   # Prévisualise les doublons sans modifier la BDD
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from src.ingestion import db as DB


def _calculer_score_richesse(df: pd.DataFrame) -> pd.Series:
    """Attribue un score de complétude à chaque annonce pour garder la meilleure fiche."""
    score = pd.Series(0.0, index=df.index)

    src = df.get("source", pd.Series("", index=df.index)).astype(str).str.lower()
    score += np.where(src == "bienici", 30.0, np.where(src == "jinka", 20.0, 10.0))

    for col, pts in [
        ("latitude", 15.0),
        ("longitude", 15.0),
        ("dpe", 10.0),
        ("ges", 5.0),
        ("annee_construction", 8.0),
        ("quartier", 8.0),
        ("pieces", 4.0),
        ("etage", 3.0),
    ]:
        if col in df.columns:
            ok = df[col].notna() & (~df[col].astype(str).str.strip().isin(["", "nan", "None", "<NA>"]))
            score += np.where(ok, pts, 0.0)

    for col, pts in [("photos_urls", 10.0), ("url", 8.0)]:
        if col in df.columns:
            ok = df[col].notna() & (df[col].astype(str).str.strip().str.len() > 5)
            score += np.where(ok, pts, 0.0)

    if "description" in df.columns:
        desc_len = df["description"].fillna("").astype(str).str.len().clip(upper=1000)
        score += desc_len / 200.0

    return score


def dedupliquer_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Supprime les doublons d'id et les doublons multi-sources en conservant la fiche la plus riche."""
    if df.empty:
        return df, 0

    n_avant = len(df)
    d = df.copy()
    d["_richesse"] = _calculer_score_richesse(d)
    d = d.sort_values("_richesse", ascending=False)

    # 1. Doublons exacts d'id
    d = d.drop_duplicates(subset=["id"], keep="first")

    # 2. Doublons croisés multi-sources (prix, surface arrondie, code postal, type_bien)
    d["_prix_cle"] = pd.to_numeric(d.get("prix"), errors="coerce").round(-2)
    d["_surf_cle"] = pd.to_numeric(d.get("surface_m2"), errors="coerce").round(0)
    d["_cp_cle"] = d.get("code_postal", pd.Series("", index=d.index)).fillna("").astype(str).str.strip().str[:5]
    d["_type_cle"] = d.get("type_bien", pd.Series("", index=d.index)).fillna("").astype(str).str.strip().str.lower()

    valide = d["_prix_cle"].notna() & d["_surf_cle"].notna() & (d["_cp_cle"] != "")
    d_val = d[valide].drop_duplicates(subset=["_prix_cle", "_surf_cle", "_cp_cle", "_type_cle"], keep="first")
    d_inval = d[~valide]

    # 3. Palier flou multi-agences (même CP, type, |Δsurface| <= 2 m², |Δprix| <= 5 %, distance <= 150 m)
    if len(d_val) > 1:
        kept_idx = []
        lat_s = pd.to_numeric(d_val.get("latitude"), errors="coerce").to_numpy()
        lon_s = pd.to_numeric(d_val.get("longitude"), errors="coerce").to_numpy()
        prix_s = pd.to_numeric(d_val.get("prix"), errors="coerce").to_numpy()
        surf_s = pd.to_numeric(d_val.get("surface_m2"), errors="coerce").to_numpy()
        cp_s = d_val["_cp_cle"].to_numpy()
        type_s = d_val["_type_cle"].to_numpy()

        supprime = np.zeros(len(d_val), dtype=bool)
        for i in range(len(d_val)):
            if supprime[i]:
                continue
            kept_idx.append(d_val.index[i])
            if np.isnan(prix_s[i]) or np.isnan(surf_s[i]) or prix_s[i] <= 0:
                continue
            same_grp = (~supprime) & (np.arange(len(d_val)) > i) & (cp_s == cp_s[i]) & (type_s == type_s[i])
            if not same_grp.any():
                continue
            close_surf = np.abs(surf_s - surf_s[i]) <= 2.0
            close_prix = (np.abs(prix_s - prix_s[i]) / prix_s[i]) <= 0.05
            cand = same_grp & close_surf & close_prix
            if cand.any() and not np.isnan(lat_s[i]) and not np.isnan(lon_s[i]):
                has_gps = (~np.isnan(lat_s)) & (~np.isnan(lon_s))
                dlat = np.radians(np.where(has_gps, lat_s - lat_s[i], 0.0))
                dlon = np.radians(np.where(has_gps, lon_s - lon_s[i], 0.0))
                a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat_s[i])) * np.cos(np.radians(np.where(has_gps, lat_s, lat_s[i]))) * np.sin(dlon / 2.0) ** 2
                dist_m = 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
                cand = cand & ((~has_gps) | (dist_m <= 150.0))
            supprime[cand] = True
        d_val = d_val.loc[kept_idx]

    d_clean = pd.concat([d_val, d_inval], ignore_index=False)
    d_clean = d_clean.drop(columns=["_richesse", "_prix_cle", "_surf_cle", "_cp_cle", "_type_cle"], errors="ignore")
    d_clean = d_clean.reset_index(drop=True)

    return d_clean, n_avant - len(d_clean)


def executer_deduplication(dry_run: bool = False) -> dict:
    """Nettoie les tables brutes PostgreSQL et leurs CSV miroirs."""
    rapport = {}
    cibles = [
        (DB.TABLE_HIST_BRUT, DB.CSV_HIST_BRUT),
        (DB.TABLE_RECENT_BRUT, DB.CSV_RECENT_BRUT),
    ]
    for table, csv_path in cibles:
        df = DB.lire_table_ou_csv(table, csv_path)
        n_avant = len(df)
        df_clean, nb_suppr = dedupliquer_dataframe(df)
        rapport[table] = {"avant": n_avant, "apres": len(df_clean), "supprimes": nb_suppr}
        print(f"[Déduplication] {table:<25} : {n_avant} -> {len(df_clean)} lignes (-{nb_suppr} doublons)")
        if not dry_run and nb_suppr > 0:
            df_clean.to_csv(csv_path, index=False)
            DB.ecrire_table_sql(df_clean, table)
    return rapport


def main():
    parser = argparse.ArgumentParser(description="Déduplication de la base immobilière PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Simule sans modifier la BDD ni les CSV")
    args = parser.parse_args()
    executer_deduplication(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
