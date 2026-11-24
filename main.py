"""
Point d'entrée unique (Orchestrateur CLI) — IA Immobilier Poitiers.

Architecture à plat à la racine avec séparation stricte entre l'entraînement
et la prédiction, et regroupement de tous les CSV dans le dossier `csv/`.

Commandes disponibles :
  python main.py                  # Par défaut : synchronise les sources -> tables features -> train -> predict
  python main.py update           # Synchronise Jinka + Bien'ici -> tables brutes -> tables features -> train -> predict
  python main.py features         # ÉTAPE 1 : Calcule le Feature Engineering et sauvegarde `offres_historique_features` & `offres_recentes_features`
  python main.py train            # ÉTAPE 2 : Entraîne le modèle directement sur `offres_historique_features` et sauvegarde `modele_poitiers.joblib`
  python main.py predict          # ÉTAPE 3 : Charge `modele_poitiers.joblib` et prédit directement sur `offres_recentes_features`
  python main.py improve          # Lance les expériences MLflow (comparaison de modèles, ablation features, graphiques)
  python main.py eval             # Affiche le rapport complet d'évaluation IA + Notaires DVF + Marges de négociation
  python main.py check            # Vérifie en direct par requête HTTP si les annonces actives sont toujours en ligne
  python main.py db               # Affiche les 4 tables PostgreSQL, le décompte par statut et un aperçu des données
  python main.py scrape           # Lance les scrapers incrémentaux (--source bienici|jinka|all)
  python main.py dash             # Lance le serveur interactif Dash sur http://127.0.0.1:8050 (sans ré-entraîner)
"""

import argparse
import asyncio
import pandas as pd

import DB
import Entrainement as tr
import FeatureEngineering as fe
import Merge
import Prediction as pr


def cmd_update(args):
    """Synchronise les fichiers JSON vers PostgreSQL / csv/, génère les tables features, entraîne et prédit."""
    Merge.executer_pipeline(
        verifier_http=args.verifier_http,
        limite_verif_http=args.limite,
        reentrainer=not args.no_train,
    )


def cmd_features(_args):
    """ÉTAPE 1 (FeatureEngineering.py) — Applique le Feature Engineering et génère les tables `*_features`."""
    print("=" * 78)
    print("  ÉTAPE 1 : CONSTRUCTION DES TABLES APRÈS FEATURE ENGINEERING (FeatureEngineering.py)")
    print("=" * 78)
    fe.construire_tables_features(verbose=True)


def cmd_train(args):
    """ÉTAPE 2 (Entrainement.py) — Entraîne le modèle IA directement sur `offres_historique_features`."""
    print("=" * 78)
    print("  ÉTAPE 2 : ENTRAÎNEMENT DU MODÈLE SUR 'offres_historique_features' (Entrainement.py)")
    print("=" * 78)
    tr.entrainer_et_sauvegarder_modele(model_path=args.modele, afficher=True)


def cmd_predict(args):
    """ÉTAPE 3 (Prediction.py) — Charge `modele_poitiers.joblib` et prédit sur `offres_recentes_features`."""
    print("=" * 78)
    print("  ÉTAPE 3 : PRÉDICTION SUR 'offres_recentes_features' (Prediction.py)")
    print("=" * 78)
    df_pred = pr.predire_sur_offres_recentes(model_path=args.modele)
    col_tri = "decote_nette_pct" if "decote_nette_pct" in df_pred.columns else "ecart_pct"
    top = df_pred.sort_values(col_tri, ascending=True).head(10)
    print("\nTop 10 meilleures affaires nettes (après travaux estimés & négo) :")
    for _, r in top.iterrows():
        trav = r.get("cout_travaux_estime_euro", 0) or 0
        dec_n = r.get("decote_nette_pct", r["ecart_pct"])
        print(
            f"  • [{r['id']}] {r['type_label']} {r['surface_m2']:.0f} m² ({r['quartier_clean']}) : "
            f"Affiché {r['prix_reel']:,.0f} € + Travaux {trav:,.0f} € | "
            f"Rénové IA {r.get('valeur_estimee_renovee', r['prix_predit']):,.0f} € ({dec_n:+.1f}%) | "
            f"Cible négo : {r['prix_cible_nego']:,.0f} € (-{r['taux_nego_estime_pct']:.1f}%)"
        )


def cmd_eval(_args):
    """Affiche le rapport complet d'évaluation IA, DVF Notaires et négociation à partir de `offres_historique_features`."""
    df_hist_feat = DB.lire_table_ou_csv(DB.TABLE_HIST_FEAT, DB.CSV_HIST_FEAT)
    if df_hist_feat.empty or "dist_centre_km" not in df_hist_feat.columns:
        df_hist_feat, _ = fe.construire_tables_features(verbose=True)
    tr.evaluer_complet(df_hist_feat, afficher=True)


def cmd_improve(args):
    """Lance l'analyse des features, la comparaison de modèles et le tracking MLflow (Improvement.py)."""
    import Improvement

    Improvement.executer_improvement(ouvrir_graphe=args.open or args.ui)


def cmd_check(args):
    """Vérifie par requête HTTP directe si les annonces actives sont toujours en ligne."""
    df_hist = DB.lire_table_ou_csv(DB.TABLE_HIST_BRUT, DB.CSV_HIST_BRUT)
    df_rec = DB.lire_table_ou_csv(DB.TABLE_RECENT_BRUT, DB.CSV_RECENT_BRUT)
    if df_rec.empty:
        print("Aucune annonce active à vérifier.")
        return

    echantillon = df_rec.head(args.limite)
    print(f"[Vérification HTTP en direct sur {len(echantillon)} annonces actives...]")
    expirees = []
    for _, r in echantillon.iterrows():
        ad_id = str(r["id"])
        url = str(r.get("url", ""))
        en_ligne = DB.verifier_annonce_en_ligne_http(ad_id, url)
        statut_txt = "EN LIGNE" if en_ligne else "EXPIRÉE / VENDUE"
        print(f"  • {ad_id:<22} -> {statut_txt} ({url})")
        if not en_ligne:
            expirees.append(ad_id)

    if expirees and not df_hist.empty:
        maintenant = pd.Timestamp.now(tz="UTC").floor("s")
        mask = df_hist["id"].astype(str).isin(expirees)
        df_hist.loc[mask, "est_active"] = False
        df_hist.loc[mask, "statut"] = "expiree_http"
        df_hist.loc[mask, "date_desactivation"] = maintenant
        df_rec_maj = df_hist[df_hist["est_active"] == True].copy()  # noqa: E712
        df_hist.to_csv(DB.CSV_HIST_BRUT, index=False)
        df_rec_maj.to_csv(DB.CSV_RECENT_BRUT, index=False)
        DB.ecrire_table_sql(df_hist, DB.TABLE_HIST_BRUT)
        DB.ecrire_table_sql(df_rec_maj, DB.TABLE_RECENT_BRUT)
        print(f"-> {len(expirees)} annonces désactivées dans l'historique et retirées des offres récentes.")


def cmd_scrape(args):
    """Lance les scrapers Bien'ici et/ou Jinka en mode incrémental."""
    if args.source in ("bienici", "all"):
        import BienIci

        print(f"[Scraping Bien'ici — {args.ville} (max {args.max_annonces})]")
        brutes = BienIci.recuperer_annonces(
            ville=args.ville,
            type_operation="buy",
            max_annonces=args.max_annonces,
            nouvelles_seulement=not args.complet,
        )
        annonces = [BienIci.simplifier(a) for a in brutes]
        if annonces:
            BienIci.exporter_json(annonces, "annonces_bienici.json")
            BienIci.exporter_csv(annonces, "csv/annonces_bienici.csv")

    if args.source in ("jinka", "all"):
        import Jinka

        print("[Scraping incrémental Jinka]")
        asyncio.run(Jinka.main())

    if args.source in ("moteurimmo", "all"):
        import MoteurImmo

        print("[Export interactif MoteurImmo Poitiers -> BDD]")
        asyncio.run(MoteurImmo.exporter_moteurimmo_interactif(lancer_pipeline_ia=True))


def cmd_db(_args):
    """Affiche l'état détaillé des tables PostgreSQL (ou CSV miroirs) et un aperçu."""
    DB.afficher_etat_postgres()
    df_hist = DB.lire_table_ou_csv(DB.TABLE_HIST_BRUT, DB.CSV_HIST_BRUT)
    df_pred = DB.lire_table_ou_csv(DB.TABLE_PREDICTIONS, DB.CSV_PREDICTIONS)
    if not df_hist.empty and "statut" in df_hist.columns:
        print("\n[Répartition par statut dans 'offres_historique_brut']")
        for st, nb in df_hist["statut"].value_counts().items():
            print(f"  • {st:<18} : {nb:>4} offres")
    if not df_pred.empty:
        cols = [
            c for c in ["id", "type_label", "surface_m2", "quartier_clean", "prix_reel", "prix_predit", "ecart_pct", "prix_cible_nego"]
            if c in df_pred.columns
        ]
        print("\n[Aperçu des 5 premières lignes de 'offres_predictions']")
        print(df_pred[cols].head(5).to_string(index=False))


def cmd_dash(args):
    """Lance le Dashboard interactif Dash (lit directement les prédictions pré-calculées)."""
    import Dashboard

    Dashboard.lancer_dashboard(port=args.port, debug=not args.no_debug)


def construire_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Point d'entrée unique — IA Immobilier Poitiers"
    )
    sub = parser.add_subparsers(dest="commande", help="Sous-commande à exécuter")

    # 1. update (défaut)
    p_up = sub.add_parser("update", help="Synchronise les données brutes, entraîne sur l'historique et prédit sur le récent")
    p_up.add_argument("--verifier-http", action="store_true", help="Vérifie par HTTP si les annonces sont encore en ligne")
    p_up.add_argument("--limite", type=int, default=30, help="Nombre max d'annonces à vérifier en HTTP")
    p_up.add_argument("--no-train", action="store_true", help="Ne ré-entraîne pas le modèle (exécute uniquement la prédiction)")
    p_up.set_defaults(func=cmd_update)

    # 2. features
    p_fe = sub.add_parser("features", help="Calcule le Feature Engineering sur les tables brutes et sauvegarde les tables *_features")
    p_fe.set_defaults(func=cmd_features)

    # 3. train
    p_tr = sub.add_parser("train", help="Entraîne le modèle IA directement sur 'offres_historique_features' et sauvegarde modele_poitiers.joblib")
    p_tr.add_argument("--modele", default=tr.CHEMIN_MODELE, help="Chemin du fichier modèle .joblib")
    p_tr.set_defaults(func=cmd_train)

    # 4. predict
    p_pr = sub.add_parser("predict", help="Charge le modèle sauvegardé et prédit directement sur 'offres_recentes_features'")
    p_pr.add_argument("--modele", default=tr.CHEMIN_MODELE, help="Chemin du fichier modèle .joblib")
    p_pr.set_defaults(func=cmd_predict)

    # 5. improve (MLflow)
    p_im = sub.add_parser("improve", help="Lance la comparaison de modèles et l'analyse de features avec MLflow (Improvement.py)")
    p_im.add_argument("--open", action="store_true", help="Ouvre automatiquement le rapport graphique HTML dans le navigateur")
    p_im.add_argument("--ui", action="store_true", help="Alias de --open pour ouvrir le rapport graphique")
    p_im.set_defaults(func=cmd_improve)

    # 4. eval
    p_ev = sub.add_parser("eval", help="Affiche le rapport complet d'évaluation IA & Notaires DVF")
    p_ev.set_defaults(func=cmd_eval)

    # 6. check
    p_ch = sub.add_parser("check", help="Vérifie en direct par requête HTTP si les offres actives sont encore en ligne")
    p_ch.add_argument("--limite", type=int, default=20, help="Nombre d'annonces actives à tester")
    p_ch.set_defaults(func=cmd_check)

    # 7. db
    p_db = sub.add_parser("db", help="Affiche l'état des 4 tables PostgreSQL et un aperçu des données")
    p_db.set_defaults(func=cmd_db)

    # 6. scrape
    p_sc = sub.add_parser("scrape", help="Lance les scrapers Bien'ici, Jinka et/ou MoteurImmo")
    p_sc.add_argument("--source", choices=["bienici", "jinka", "moteurimmo", "all"], default="bienici", help="Source à scraper")
    p_sc.add_argument("--ville", default="Poitiers", help="Ville cible pour Bien'ici")
    p_sc.add_argument("--max-annonces", type=int, default=200, help="Nombre max d'annonces Bien'ici")
    p_sc.add_argument("--complet", action="store_true", help="Récupère même les annonces déjà connues")
    p_sc.set_defaults(func=cmd_scrape)

    # 7. dash
    p_da = sub.add_parser("dash", help="Lance le Dashboard interactif Dash sur le port 8050")
    p_da.add_argument("--port", type=int, default=8050, help="Port du serveur Dash")
    p_da.add_argument("--no-debug", action="store_true", help="Désactive le mode debug de Dash")
    p_da.set_defaults(func=cmd_dash)

    return parser


def main():
    parser = construire_parser()
    args = parser.parse_args()
    if not args.commande:
        # Par défaut, `python main.py` lance `update`
        args = parser.parse_args(["update"])
    args.func(args)


if __name__ == "__main__":
    main()

