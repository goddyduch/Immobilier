"""
Dashboard interactif Dash (Port 8050) — IA Immobilier Grand Poitiers.

4 onglets :
  1. Métier & Opportunités (filtres par tranche de m², bouton Détail/SHAP par offre, Hybride/AVM/ML, Travaux, Caractéristiques)
  2. Modèle & Suivi AVM (Benchmark ML vs AVM vs Hybride, suivi historique MLflow, précision par tranche de m², ablation)
  3. Features (Toutes les variables : taux de remplissage, stats, importance permutation & verdict)
  4. Notaires DVF, Négociation & Travaux (marges de négociation par m²/DPE/ancienneté/quartier, écart DVF, travaux & mobilité)

Usage :
    python main.py dash
"""

import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, ctx, dcc, html

import DB
import Entrainement as tr
import FeatureEngineering as fe
import Prediction as pr

TRANCHES_M2 = [
    ("10-20 m²", 10, 20),
    ("20-30 m²", 20, 30),
    ("30-45 m²", 30, 45),
    ("45-65 m²", 45, 65),
    ("65-90 m²", 65, 90),
    ("90-120 m²", 90, 120),
    ("120+ m²", 120, 9999),
]


def _attribuer_tranche_m2(surface_series: pd.Series) -> pd.Series:
    s = pd.to_numeric(surface_series, errors="coerce")
    res = pd.Series("Inconnu", index=s.index, dtype=object)
    for label, b_min, b_max in TRANCHES_M2:
        mask = (s >= b_min) & (s < b_max)
        res = np.where(mask, label, res)
    return pd.Series(res, index=s.index)


def _preparer_df_dashboard() -> pd.DataFrame:
    df = pr.charger_donnees_dashboard()["df_pred"]
    if df.empty:
        return df
    df = df.copy()
    if "tranche_m2" not in df.columns:
        df["tranche_m2"] = _attribuer_tranche_m2(df["surface_m2"])
    if "prix_hybride" not in df.columns:
        avm_s = pd.to_numeric(df.get("avm_comp_blended_euro"), errors="coerce")
        df["prix_hybride"] = np.round(
            np.where(avm_s.notna() & (avm_s > 0), 0.75 * df["prix_predit"] + 0.25 * avm_s, df["prix_predit"]), 0
        )
    if "ecart_hybride_pct" not in df.columns:
        df["ecart_hybride_pct"] = np.round(
            (df["prix_reel"] - df["prix_hybride"]) / df["prix_hybride"].clip(lower=1.0) * 100.0, 1
        )
    if "avm_comp_ecart_pct" not in df.columns and "avm_comp_blended_euro" in df.columns:
        denom = pd.to_numeric(df["avm_comp_blended_euro"], errors="coerce").replace(0, np.nan)
        df["avm_comp_ecart_pct"] = np.round((df["prix_reel"] - denom) / denom * 100.0, 1)
    return df


DF_PRED = _preparer_df_dashboard()
LISTE_QUARTIERS = sorted(DF_PRED["quartier_clean"].dropna().unique().tolist()) if not DF_PRED.empty else []

BG_MAIN = "#0f172a"
BG_CARD = "#1e293b"
BORDER_COL = "#334155"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"

CARD_STYLE = {
    "backgroundColor": BG_CARD,
    "border": f"1px solid {BORDER_COL}",
    "borderRadius": "10px",
    "padding": "18px",
}
TAB_STYLE = {
    "backgroundColor": BG_MAIN,
    "color": TEXT_MUTED,
    "border": f"1px solid {BORDER_COL}",
    "padding": "10px 18px",
    "fontWeight": "600",
    "fontSize": "13px",
}
TAB_SELECTED_STYLE = {
    "backgroundColor": BG_CARD,
    "color": "#38bdf8",
    "border": f"1px solid {BORDER_COL}",
    "borderTop": "3px solid #38bdf8",
    "padding": "10px 18px",
    "fontWeight": "700",
    "fontSize": "13px",
}


def fmt_euro(val) -> str:
    if val is None or pd.isna(val):
        return "N/C"
    return f"{int(round(float(val))):,} €".replace(",", " ")


def _safe_int(val, default: int = 0) -> int:
    return default if val is None or pd.isna(val) else int(round(float(val)))


def _safe_float(val, default: float = 0.0) -> float:
    return default if val is None or pd.isna(val) else float(val)


def creer_fig_courbe(df_sub: pd.DataFrame):
    fig = go.Figure()
    if df_sub.empty:
        return fig
    df_s = df_sub.sort_values("prix_predit").reset_index(drop=True)
    fig.add_trace(go.Scatter(
        x=df_s.index, y=df_s["prix_reel"], mode="markers", name="Prix Affiché",
        marker={"color": "#38bdf8", "size": 5, "opacity": 0.6},
        customdata=np.stack([df_s["quartier_clean"], df_s["type_label"], df_s["surface_m2"], df_s["prix_hybride"], df_s["taux_nego_estime_pct"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]} %{customdata[2]:.0f}m²)<br>Affiché : %{y:,.0f} €<br>Hybride : %{customdata[3]:,.0f} €<br>Négo : -%{customdata[4]:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_s.index, y=df_s["prix_predit"], mode="lines", name="ML (HistGB)",
        line={"color": "#a855f7", "width": 2},
    ))
    if "prix_hybride" in df_s.columns:
        fig.add_trace(go.Scatter(
            x=df_s.index, y=df_s["prix_hybride"], mode="lines", name="Hybride (75% ML + 25% AVM)",
            line={"color": "#10b981", "width": 2, "dash": "dot"},
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN, "size": 12}, margin={"l": 55, "r": 20, "t": 20, "b": 40},
        legend={"orientation": "h", "y": 1.08, "x": 0}, height=330,
        xaxis={"title": "Biens classés par valeur croissante", "gridcolor": "rgba(148,163,184,0.12)"},
        yaxis={"title": "Prix (€)", "gridcolor": "rgba(148,163,184,0.12)"},
    )
    return fig


def creer_fig_scatter(df_sub: pd.DataFrame):
    fig = go.Figure()
    if df_sub.empty:
        return fig
    couleurs = np.where(df_sub["ecart_hybride_pct"] <= -10, "#10b981", np.where(df_sub["ecart_hybride_pct"] >= 10, "#f43f5e", "#60a5fa"))
    fig.add_trace(go.Scatter(
        x=df_sub["prix_hybride"], y=df_sub["prix_reel"], mode="markers", name="Annonces",
        marker={"color": couleurs, "size": 6, "opacity": 0.75},
        customdata=np.stack([df_sub["quartier_clean"], df_sub["surface_m2"], df_sub["ecart_hybride_pct"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]:.0f}m²)<br>Hybride : %{x:,.0f} €<br>Affiché : %{y:,.0f} €<br>Écart Hybride : %{customdata[2]:+.1f}%<extra></extra>",
    ))
    max_val = float(max(df_sub["prix_reel"].max(), df_sub["prix_hybride"].max()))
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", name="Juste prix", line={"color": "#94a3b8", "dash": "dash"}))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN, "size": 12}, margin={"l": 55, "r": 20, "t": 20, "b": 40},
        showlegend=False, height=330,
        xaxis={"title": "Estimation Hybride (€)", "gridcolor": "rgba(148,163,184,0.12)"},
        yaxis={"title": "Prix Affiché (€)", "gridcolor": "rgba(148,163,184,0.12)"},
    )
    return fig


def creer_fig_degressivite_surface(df_sub: pd.DataFrame):
    fig = go.Figure()
    if df_sub.empty:
        return fig
    d = df_sub.copy()
    d["tranche_m2"] = _attribuer_tranche_m2(d["surface_m2"])
    ordre = [t[0] for t in TRANCHES_M2]
    g = d.groupby("tranche_m2", as_index=False).agg(
        pm2_reel=("pm2_reel", "median"),
        pm2_pred=("pm2_predit", "median"),
        pm2_dvf=("pm2_notaire_dvf", "median"),
        nb=("id", "count"),
    )
    g["ord"] = g["tranche_m2"].apply(lambda x: ordre.index(x) if x in ordre else 99)
    g = g.sort_values("ord")
    fig.add_trace(go.Bar(x=g["tranche_m2"], y=g["pm2_reel"], name="Affiché (€/m²)", marker_color="#38bdf8"))
    fig.add_trace(go.Bar(x=g["tranche_m2"], y=g["pm2_pred"], name="ML HistGB (€/m²)", marker_color="#a855f7"))
    fig.add_trace(go.Bar(x=g["tranche_m2"], y=g["pm2_dvf"], name="Notaire DVF (€/m²)", marker_color="#f59e0b"))
    fig.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN, "size": 11}, height=320, margin={"l": 50, "r": 15, "t": 25, "b": 40},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        yaxis={"title": "Prix médian (€/m²)", "gridcolor": "rgba(148,163,184,0.15)"},
    )
    return fig


def _tableau_metriques_par_tranche(df_sub: pd.DataFrame):
    if df_sub.empty:
        return html.Div("Aucune donnée.")
    d = df_sub.copy()
    d["tranche_m2"] = _attribuer_tranche_m2(d["surface_m2"])
    rows = []
    for label, _, _ in TRANCHES_M2:
        sub = d[d["tranche_m2"] == label]
        if sub.empty:
            continue
        mdape_ml = float(np.median(np.abs(sub["ecart_pct"])))
        mdape_hyb = float(np.median(np.abs(sub["ecart_hybride_pct"])))
        dec_med = float(sub["ecart_hybride_pct"].median())
        rows.append(html.Tr(style={"borderBottom": f"1px solid {BORDER_COL}", "fontSize": "12px"}, children=[
            html.Td(label, style={"padding": "7px 8px", "fontWeight": "700"}),
            html.Td(f"{len(sub)}", style={"padding": "7px 8px", "textAlign": "right"}),
            html.Td(f"{sub['pm2_reel'].median():,.0f} €/m²".replace(",", " "), style={"padding": "7px 8px", "textAlign": "right"}),
            html.Td(f"{sub['pm2_notaire_dvf'].median():,.0f} €/m²".replace(",", " "), style={"padding": "7px 8px", "textAlign": "right", "color": "#f59e0b"}),
            html.Td(f"±{mdape_ml:.1f}%", style={"padding": "7px 8px", "textAlign": "right", "color": "#a855f7"}),
            html.Td(f"±{mdape_hyb:.1f}%", style={"padding": "7px 8px", "textAlign": "right", "color": "#10b981", "fontWeight": "700"}),
            html.Td(f"{dec_med:+.1f}%", style={"padding": "7px 8px", "textAlign": "right"}),
        ]))
    return html.Table(style={"width": "100%", "borderCollapse": "collapse"}, children=[
        html.Thead(html.Tr(style={"borderBottom": f"1px solid {BORDER_COL}", "color": TEXT_MUTED, "fontSize": "11px"}, children=[
            html.Th("Catégorie surface", style={"padding": "7px 8px", "textAlign": "left"}),
            html.Th("Biens", style={"padding": "7px 8px", "textAlign": "right"}),
            html.Th("Affiché méd.", style={"padding": "7px 8px", "textAlign": "right"}),
            html.Th("DVF méd.", style={"padding": "7px 8px", "textAlign": "right"}),
            html.Th("MdAPE ML", style={"padding": "7px 8px", "textAlign": "right"}),
            html.Th("MdAPE Hybride", style={"padding": "7px 8px", "textAlign": "right"}),
            html.Th("Écart méd.", style={"padding": "7px 8px", "textAlign": "right"}),
        ])),
        html.Tbody(rows),
    ])


def construire_contenu_onglet_modele():
    """Onglet 2 : Modèle & Suivi AVM (Benchmark ML vs AVM vs Hybride, historique MLflow, tranches m², ablation)."""
    import AVMComparable

    blocs = []
    df_cmp = AVMComparable.comparer_ml_vs_avm(DF_PRED)
    if not df_cmp.empty:
        cmp_rows = [
            html.Tr(
                style={"borderBottom": f"1px solid {BORDER_COL}", "fontSize": "12px"},
                children=[
                    html.Td(r["Approche"], style={"padding": "8px", "fontWeight": "700"}),
                    html.Td(f"{int(r['Biens'])}", style={"padding": "8px", "textAlign": "right"}),
                    html.Td(f"{r['R²']:.3f}", style={"padding": "8px", "textAlign": "right", "color": "#10b981", "fontWeight": "700"}),
                    html.Td(fmt_euro(r["MAE (€)"]), style={"padding": "8px", "textAlign": "right"}),
                    html.Td(f"±{r['MdAPE (%)']:.1f}%", style={"padding": "8px", "textAlign": "right", "color": "#38bdf8", "fontWeight": "700"}),
                    html.Td(f"{r['<= 10%']:.1f}%", style={"padding": "8px", "textAlign": "right"}),
                    html.Td(f"{r['<= 15%']:.1f}%", style={"padding": "8px", "textAlign": "right"}),
                ],
            )
            for _, r in df_cmp.iterrows()
        ]
        blocs.append(html.Div(
            style=CARD_STYLE,
            children=[
                html.H2("1. Benchmark AVM : Modèle ML vs AVM Comparables vs Hybride (75/25)", style={"fontSize": "16px", "margin": "0 0 6px 0"}),
                html.P("Comparaison directe sur les offres actives : l'hybride réduit la MAE et augmente la part de biens estimés à ≤ 15 %.", style={"fontSize": "12px", "color": TEXT_MUTED, "margin": "0 0 10px 0"}),
                html.Table(
                    style={"width": "100%", "borderCollapse": "collapse"},
                    children=[
                        html.Thead(html.Tr(
                            style={"borderBottom": f"1px solid {BORDER_COL}", "color": TEXT_MUTED, "fontSize": "12px"},
                            children=[
                                html.Th("Approche", style={"padding": "8px", "textAlign": "left"}),
                                html.Th("Biens", style={"padding": "8px", "textAlign": "right"}),
                                html.Th("R²", style={"padding": "8px", "textAlign": "right"}),
                                html.Th("MAE", style={"padding": "8px", "textAlign": "right"}),
                                html.Th("MdAPE", style={"padding": "8px", "textAlign": "right"}),
                                html.Th("≤ 10%", style={"padding": "8px", "textAlign": "right"}),
                                html.Th("≤ 15%", style={"padding": "8px", "textAlign": "right"}),
                            ],
                        )),
                        html.Tbody(cmp_rows),
                    ],
                ),
            ],
        ))

    # Suivi historique MLflow (pour voir l'amélioration dans le temps)
    mlflow_path = os.path.join("csv", "mlflow_runs.csv")
    if os.path.exists(mlflow_path):
        df_runs = pd.read_csv(mlflow_path)
        if not df_runs.empty and "mae" in df_runs.columns:
            fig_runs = go.Figure()
            x_lbl = df_runs["run_name"].astype(str) if "run_name" in df_runs.columns else df_runs.index.astype(str)
            fig_runs.add_trace(go.Scatter(x=x_lbl, y=df_runs["mae"], mode="lines+markers", name="MAE (€)", line={"color": "#38bdf8", "width": 2}))
            if "r2" in df_runs.columns:
                fig_runs.add_trace(go.Scatter(x=x_lbl, y=df_runs["r2"], mode="lines+markers", name="R²", yaxis="y2", line={"color": "#10b981", "width": 2}))
            fig_runs.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": TEXT_MAIN, "size": 11}, height=300, margin={"l": 55, "r": 55, "t": 25, "b": 70},
                yaxis={"title": "MAE (€)", "gridcolor": "rgba(148,163,184,0.12)"},
                yaxis2={"title": "R²", "overlaying": "y", "side": "right"},
                legend={"orientation": "h", "y": 1.12, "x": 0},
            )
            blocs.append(html.Div(
                style=CARD_STYLE,
                children=[
                    html.H2("2. Suivi des Expériences AVM / MLflow (Évolution R² & MAE)", style={"fontSize": "16px", "margin": "0 0 6px 0"}),
                    dcc.Graph(figure=fig_runs, config={"displayModeBar": False}),
                ],
            ))

    blocs.append(html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(480px, 1fr))", "gap": "20px"},
        children=[
            html.Div(style=CARD_STYLE, children=[
                html.H2("3. Précision AVM & Prix/m² par Catégorie de Surface", style={"fontSize": "16px", "margin": "0 0 8px 0"}),
                _tableau_metriques_par_tranche(DF_PRED),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.H2("4. Effet Dégressif par Tranche de m²", style={"fontSize": "16px", "margin": "0 0 8px 0"}),
                dcc.Graph(figure=creer_fig_degressivite_surface(DF_PRED), config={"displayModeBar": False}),
            ]),
        ],
    ))

    df_abl = pd.read_csv(DB.CSV_AUDIT_ABLATION) if os.path.exists(DB.CSV_AUDIT_ABLATION) else pd.DataFrame()
    if not df_abl.empty:
        fig_abl = go.Figure()
        fig_abl.add_trace(go.Bar(
            x=df_abl["etape"], y=df_abl["mae"], name="MAE KFold (€)", marker_color="#3b82f6",
            text=[f"{int(m):,} € (R²={r:.3f})".replace(",", " ") for m, r in zip(df_abl["mae"], df_abl["r2"])],
            textposition="auto",
        ))
        fig_abl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": TEXT_MAIN, "size": 11}, height=320, margin={"l": 55, "r": 15, "t": 20, "b": 70},
            yaxis={"title": "Erreur MAE (€)", "gridcolor": "rgba(148,163,184,0.15)"},
        )
        blocs.append(html.Div(style=CARD_STYLE, children=[
            html.H2("5. Ablation Étape par Étape", style={"fontSize": "16px", "margin": "0 0 6px 0"}),
            dcc.Graph(figure=fig_abl, config={"displayModeBar": False}),
        ]))

    return html.Div(style={"display": "flex", "flexDirection": "column", "gap": "20px"}, children=blocs)


def construire_contenu_onglet_features():
    """Onglet 3 : Toutes les features affichées (remplissage, stats, importance permutation & verdict)."""
    df_aud = pd.read_csv(DB.CSV_AUDIT_FEATURES) if os.path.exists(DB.CSV_AUDIT_FEATURES) else pd.DataFrame()
    aud_map = df_aud.set_index("feature").to_dict(orient="index") if not df_aud.empty else {}

    X_cols = []
    if not DF_PRED.empty:
        try:
            X_tmp, _ = fe.construire_X_y(fe.supprimer_colonnes_prediction(DF_PRED.copy()))
            X_cols = list(X_tmp.columns)
        except Exception:
            X_cols = [c for c in DF_PRED.columns if c in fe.LABELS_FEATURES]

    extra_metier = [
        "surface_carrez_m2", "surface_utile_m2", "prix_m2_carrez", "prix_m2_utile",
        "dist_arret_bus_m", "nb_arrets_bus_400m", "dist_commerce_alim_m", "nb_commerces_500m",
        "score_commodites_500m", "loyer_m2_estime", "loyer_mensuel_estime", "rendement_brut_pct",
        "rendement_net_estime_pct", "cout_travaux_estime_euro", "commune_donnees_limitees",
    ]
    toutes = list(dict.fromkeys(X_cols + [c for c in extra_metier if c in DF_PRED.columns]))

    rows = []
    for col in toutes:
        if col not in DF_PRED.columns:
            continue
        s = DF_PRED[col]
        taux = float(s.notna().mean() * 100.0)
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() > 0:
            stat_txt = f"méd. {num.median():.2f} [{num.min():.1f} ; {num.max():.1f}]"
        else:
            stat_txt = f"{s.nunique()} modalités"
        info = aud_map.get(col, {})
        perm_txt = f"±{int(info['impact_perm_euro']):,} €".replace(",", " ") if "impact_perm_euro" in info else "—"
        gain_txt = f"{int(info['gain_drop_mae_euro']):+,} €".replace(",", " ") if "gain_drop_mae_euro" in info else "—"
        statut = info.get("statut", "Actif (Modèle)" if col in X_cols else "Indicateur Métier")
        col_st = "#10b981" if "Utile" in statut or "Actif" in statut else ("#f59e0b" if "Métier" in statut else TEXT_MUTED)
        rows.append(html.Tr(style={"borderBottom": f"1px solid {BORDER_COL}", "fontSize": "12px"}, children=[
            html.Td(col, style={"padding": "7px 8px", "fontFamily": "monospace", "color": "#38bdf8"}),
            html.Td(fe.LABELS_FEATURES.get(col, col), style={"padding": "7px 8px", "fontWeight": "600"}),
            html.Td(f"{taux:.0f}%", style={"padding": "7px 8px", "textAlign": "right"}),
            html.Td(stat_txt, style={"padding": "7px 8px", "textAlign": "right", "color": TEXT_MUTED}),
            html.Td(perm_txt, style={"padding": "7px 8px", "textAlign": "right"}),
            html.Td(gain_txt, style={"padding": "7px 8px", "textAlign": "right"}),
            html.Td(statut, style={"padding": "7px 8px", "fontWeight": "600", "color": col_st}),
        ]))

    return html.Div(style={"display": "flex", "flexDirection": "column", "gap": "20px"}, children=[
        html.Div(style=CARD_STYLE, children=[
            html.H2(f"Catalogue Complet des Features ({len(rows)} variables)", style={"fontSize": "16px", "margin": "0 0 6px 0"}),
            html.P("Inclut les variables du modèle HistGradientBoosting (dont est_en_qpv, mobilité bus, commerces 500m, loyers, embeddings photos) et les indicateurs métier.", style={"fontSize": "12px", "color": TEXT_MUTED, "margin": "0 0 12px 0"}),
            html.Div(style={"overflowX": "auto"}, children=html.Table(
                style={"width": "100%", "borderCollapse": "collapse", "textAlign": "left"},
                children=[
                    html.Thead(html.Tr(style={"borderBottom": f"1px solid {BORDER_COL}", "color": TEXT_MUTED, "fontSize": "12px"}, children=[
                        html.Th("Nom technique", style={"padding": "8px"}),
                        html.Th("Libellé métier", style={"padding": "8px"}),
                        html.Th("Remplissage", style={"padding": "8px", "textAlign": "right"}),
                        html.Th("Distribution", style={"padding": "8px", "textAlign": "right"}),
                        html.Th("Impact Perm.", style={"padding": "8px", "textAlign": "right"}),
                        html.Th("Gain Drop-Col", style={"padding": "8px", "textAlign": "right"}),
                        html.Th("Statut", style={"padding": "8px"}),
                    ])),
                    html.Tbody(rows),
                ],
            )),
        ])
    ])


def construire_contenu_onglet_travaux():
    """Onglet 4 : Notaires DVF, Négociation avancée & Travaux / Mobilité."""
    df_t = DF_PRED.copy()
    if df_t.empty:
        return html.Div("Aucune donnée.")
    df_t["tranche_m2"] = _attribuer_tranche_m2(df_t["surface_m2"])
    cout_trav = pd.to_numeric(df_t.get("cout_travaux_estime_euro", 0), errors="coerce").fillna(0)
    avec_trav = df_t[cout_trav > 0]

    kpis = html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(210px, 1fr))", "gap": "14px"},
        children=[
            html.Div(style=CARD_STYLE, children=[
                html.Div("Marge Négo Médiane Conseillée", style={"fontSize": "12px", "color": TEXT_MUTED}),
                html.Div(f"-{df_t['taux_nego_estime_pct'].median():.1f}%", style={"fontSize": "24px", "fontWeight": "700", "color": "#38bdf8", "marginTop": "4px"}),
                html.Div(f"Économie médiane : {fmt_euro(df_t['economie_nego_euro'].median())}", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div("Surcote Affiché vs Notaire DVF", style={"fontSize": "12px", "color": TEXT_MUTED}),
                html.Div(f"{df_t['ecart_vs_notaire_pct'].median():+.1f}%", style={"fontSize": "24px", "fontWeight": "700", "color": "#f59e0b", "marginTop": "4px"}),
                html.Div(f"DVF médian : {df_t['pm2_notaire_dvf'].median():,.0f} €/m²".replace(",", " "), style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div("Biens avec Travaux Détectés", style={"fontSize": "12px", "color": TEXT_MUTED}),
                html.Div(f"{len(avec_trav)} / {len(df_t)}", style={"fontSize": "24px", "fontWeight": "700", "color": "#10b981", "marginTop": "4px"}),
                html.Div(f"Budget médian : {fmt_euro(avec_trav['cout_travaux_estime_euro'].median() if len(avec_trav) else 0)}", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div("Mobilité & Commerces (Grand Poitiers)", style={"fontSize": "12px", "color": TEXT_MUTED}),
                html.Div(f"Bus {_safe_int(df_t.get('dist_arret_bus_m', pd.Series([0])).median())} m", style={"fontSize": "24px", "fontWeight": "700", "color": "#a855f7", "marginTop": "4px"}),
                html.Div(f"Médiane {_safe_int(df_t.get('nb_commerces_500m', pd.Series([0])).median())} commerces à 500 m", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
            ]),
        ],
    )

    # Négociation par catégorie de m² & par DPE
    ordre_t = [t[0] for t in TRANCHES_M2]
    g_m2 = df_t.groupby("tranche_m2", as_index=False).agg(
        nego_pct=("taux_nego_estime_pct", "mean"),
        ecart_dvf=("ecart_vs_notaire_pct", "median"),
        eco_euro=("economie_nego_euro", "median"),
        nb=("id", "count"),
    )
    g_m2["ord"] = g_m2["tranche_m2"].apply(lambda x: ordre_t.index(x) if x in ordre_t else 99)
    g_m2 = g_m2.sort_values("ord")

    fig_nego_m2 = go.Figure()
    fig_nego_m2.add_trace(go.Bar(x=g_m2["tranche_m2"], y=g_m2["nego_pct"], name="Marge Négo conseillée (%)", marker_color="#38bdf8"))
    fig_nego_m2.add_trace(go.Bar(x=g_m2["tranche_m2"], y=g_m2["ecart_dvf"], name="Surcote Affiché vs Notaire DVF (%)", marker_color="#f59e0b"))
    fig_nego_m2.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN, "size": 11}, height=320, margin={"l": 50, "r": 15, "t": 25, "b": 40},
        legend={"orientation": "h", "y": 1.12, "x": 0}, yaxis={"title": "%", "gridcolor": "rgba(148,163,184,0.15)"},
    )

    dpe_clean = df_t["dpe"].fillna("N/C").astype(str).str.upper()
    df_t["dpe_clean"] = np.where(dpe_clean.isin(list("ABCDEFG")), dpe_clean, "N/C")
    g_dpe = df_t.groupby("dpe_clean", as_index=False).agg(nego_pct=("taux_nego_estime_pct", "mean"), eco=("economie_nego_euro", "median"), nb=("id", "count"))
    g_dpe = g_dpe.sort_values("dpe_clean")
    fig_nego_dpe = go.Figure(go.Bar(
        x=g_dpe["dpe_clean"], y=g_dpe["nego_pct"], marker_color="#10b981",
        text=[f"-{v:.1f}% ({n})" for v, n in zip(g_dpe["nego_pct"], g_dpe["nb"])], textposition="auto",
    ))
    fig_nego_dpe.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN, "size": 11}, height=320, margin={"l": 50, "r": 15, "t": 25, "b": 40},
        yaxis={"title": "Marge de négociation (%)", "gridcolor": "rgba(148,163,184,0.15)"},
    )

    return html.Div(style={"display": "flex", "flexDirection": "column", "gap": "20px"}, children=[
        kpis,
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(480px, 1fr))", "gap": "20px"}, children=[
            html.Div(style=CARD_STYLE, children=[
                html.H2("Pouvoir de Négociation & Surcote DVF par Catégorie de Surface", style={"fontSize": "16px", "margin": "0 0 6px 0"}),
                dcc.Graph(figure=fig_nego_m2, config={"displayModeBar": False}),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.H2("Impact du DPE sur la Marge de Négociation Conseillée", style={"fontSize": "16px", "margin": "0 0 6px 0"}),
                dcc.Graph(figure=fig_nego_dpe, config={"displayModeBar": False}),
            ]),
        ]),
    ])


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "IA Immobilier Grand Poitiers"

app.layout = html.Div(
    style={"backgroundColor": BG_MAIN, "color": TEXT_MAIN, "minHeight": "100vh", "fontFamily": "Inter, -apple-system, sans-serif", "padding": "22px 28px"},
    children=[
        html.Div(style={"maxWidth": "1440px", "margin": "0 auto"}, children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "14px", "marginBottom": "18px"},
                children=[
                    html.Div([
                        html.H1("IA Immobilier — Grand Poitiers (86)", style={"margin": 0, "fontSize": "24px", "fontWeight": "700"}),
                        html.P("Hybride (75% ML + 25% AVM Comparables) • SHAP • Intervalles [Q10–Q90] • Mobilité, Commerces & Loyers", style={"margin": "4px 0 0 0", "color": TEXT_MUTED, "fontSize": "13px"}),
                    ]),
                    html.Div(
                        style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"},
                        children=[
                            dcc.Dropdown(
                                id="filter-type",
                                options=[{"label": "Tous types", "value": "ALL"}, {"label": "Appartements", "value": "Appartement"}, {"label": "Maisons", "value": "Maison"}],
                                value="ALL", clearable=False, style={"width": "155px", "color": "#0f172a", "fontSize": "13px"},
                            ),
                            dcc.Dropdown(
                                id="filter-tranche-m2",
                                options=[{"label": "Toutes surfaces (m²)", "value": "ALL"}] + [{"label": t[0], "value": t[0]} for t in TRANCHES_M2],
                                value="ALL", clearable=False, style={"width": "185px", "color": "#0f172a", "fontSize": "13px"},
                            ),
                            dcc.Dropdown(
                                id="filter-quartier",
                                options=[{"label": "Tous les quartiers", "value": "ALL"}] + [{"label": q, "value": q} for q in LISTE_QUARTIERS],
                                value="ALL", clearable=False, style={"width": "205px", "color": "#0f172a", "fontSize": "13px"},
                            ),
                            dcc.Dropdown(
                                id="filter-communes-min",
                                options=[{"label": "Toutes communes", "value": "ALL"}, {"label": "Exclure communes < 15 offres", "value": "MIN15"}],
                                value="ALL", clearable=False, style={"width": "215px", "color": "#0f172a", "fontSize": "13px"},
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Tabs(
                id="main-tabs", value="tab-metier",
                children=[
                    dcc.Tab(label="1. Métier & Opportunités", value="tab-metier", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "20px", "marginTop": "18px"}, children=[
                            html.Div(id="kpis-container", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "14px"}),
                            html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "20px"}, children=[
                                html.Div(style=CARD_STYLE, children=[
                                    html.H2("Prix Affiché vs ML vs Hybride (75/25)", style={"fontSize": "16px", "fontWeight": "600", "margin": "0 0 4px 0"}),
                                    dcc.Graph(id="graph-curve", config={"displayModeBar": False}),
                                ]),
                                html.Div(style=CARD_STYLE, children=[
                                    html.H2("Prix Affiché vs Estimation Hybride (Opportunités)", style={"fontSize": "16px", "fontWeight": "600", "margin": "0 0 4px 0"}),
                                    dcc.Graph(id="graph-scatter", config={"displayModeBar": False}),
                                ]),
                            ]),
                            html.Div(style=CARD_STYLE, children=[
                                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "12px", "marginBottom": "12px"}, children=[
                                    html.Div([
                                        html.H2("Opportunités Détectées (Hybride, AVM, ML, Travaux & Confiance)", style={"fontSize": "16px", "fontWeight": "600", "margin": "0 0 4px 0"}),
                                        html.P("Clique sur 'Détail / SHAP' sur n'importe quelle ligne pour afficher l'explication SHAP, les travaux, les 5 comparables et toutes les caractéristiques.", style={"fontSize": "12px", "color": TEXT_MUTED, "margin": 0}),
                                    ]),
                                    dcc.Dropdown(
                                        id="sort-table",
                                        options=[
                                            {"label": "Trier par : Plus grosse décote Hybride (%)", "value": "decote_hybride"},
                                            {"label": "Trier par : Meilleure affaire nette (après travaux)", "value": "decote_nette"},
                                            {"label": "Trier par : Plus forte décote ML (%)", "value": "decote"},
                                            {"label": "Trier par : Plus forte décote AVM Comp. (%)", "value": "decote_avm"},
                                            {"label": "Trier par : Meilleur rendement brut (%)", "value": "rendement"},
                                            {"label": "Trier par : Prix croissant", "value": "prix_asc"},
                                        ],
                                        value="decote_hybride", clearable=False, style={"width": "330px", "color": "#0f172a", "fontSize": "13px"},
                                    ),
                                ]),
                                html.Div(id="deals-table-container", style={"overflowX": "auto"}),
                            ]),
                            html.Div(id="detail-bien-card", style=CARD_STYLE, children=[
                                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "12px", "marginBottom": "14px"}, children=[
                                    html.H2("Fiche Détaillée : SHAP, Confiance, Travaux, Hybride / AVM / ML & Caractéristiques", style={"fontSize": "16px", "fontWeight": "600", "margin": 0}),
                                    dcc.Dropdown(id="detail-bien-selector", clearable=False, style={"width": "440px", "color": "#0f172a", "fontSize": "13px"}),
                                ]),
                                html.Div(id="detail-bien-panel"),
                            ]),
                        ])
                    ]),
                    dcc.Tab(label="2. Modèle & Suivi AVM", value="tab-modele", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                        html.Div(style={"marginTop": "18px"}, children=construire_contenu_onglet_modele())
                    ]),
                    dcc.Tab(label="3. Features (Toutes)", value="tab-features", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                        html.Div(style={"marginTop": "18px"}, children=construire_contenu_onglet_features())
                    ]),
                    dcc.Tab(label="4. Notaires DVF, Négociation & Travaux", value="tab-travaux", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                        html.Div(style={"marginTop": "18px"}, children=construire_contenu_onglet_travaux())
                    ]),
                ],
            ),
        ])
    ],
)


@app.callback(
    Output("kpis-container", "children"),
    Output("graph-curve", "figure"),
    Output("graph-scatter", "figure"),
    Output("deals-table-container", "children"),
    Output("detail-bien-selector", "options"),
    Output("detail-bien-selector", "value"),
    Input("filter-type", "value"),
    Input("filter-tranche-m2", "value"),
    Input("filter-quartier", "value"),
    Input("filter-communes-min", "value"),
    Input("sort-table", "value"),
)
def mettre_a_jour_dashboard(type_val: str, tranche_val: str, quartier_val: str, commune_min_val: str, sort_val: str):
    df_f = DF_PRED.copy()
    if type_val != "ALL":
        df_f = df_f[df_f["type_label"] == type_val]
    if tranche_val != "ALL":
        df_f = df_f[df_f["tranche_m2"] == tranche_val]
    if quartier_val != "ALL":
        df_f = df_f[df_f["quartier_clean"] == quartier_val]
    if commune_min_val == "MIN15" and "commune_donnees_limitees" in df_f.columns:
        df_f = df_f[pd.to_numeric(df_f["commune_donnees_limitees"], errors="coerce").fillna(0) == 0]

    m = tr.calculer_metriques_groupe(df_f)
    if not m:
        return [], go.Figure(), go.Figure(), html.Div("Aucun bien pour ce filtre."), [], None

    kpis_cards = [
        html.Div(style=CARD_STYLE, children=[
            html.Div("Biens analysés", style={"fontSize": "12px", "color": TEXT_MUTED}),
            html.Div(f"{m['nb_biens']}", style={"fontSize": "24px", "fontWeight": "700", "marginTop": "4px"}),
            html.Div(f"Prix médian : {fmt_euro(m['prix_median'])}", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.Div("Erreur Médiane (ML vs Hybride)", style={"fontSize": "12px", "color": TEXT_MUTED}),
            html.Div(f"±{float(np.median(np.abs(df_f['ecart_hybride_pct']))):.1f}%", style={"fontSize": "24px", "fontWeight": "700", "color": "#10b981", "marginTop": "4px"}),
            html.Div(f"ML seul : ±{m['erreur_mediane_pct']}% | R² : {m['r2']:.2f}", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.Div("Confiance & Rendement Brut", style={"fontSize": "12px", "color": TEXT_MUTED}),
            html.Div(f"{_safe_int(df_f.get('score_confiance', pd.Series([0])).median())}/100", style={"fontSize": "24px", "fontWeight": "700", "color": "#38bdf8", "marginTop": "4px"}),
            html.Div(f"Rendement brut méd. : {_safe_float(df_f.get('rendement_brut_pct', pd.Series([0])).median()):.1f}%", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.Div("Affiché vs Notaire DVF", style={"fontSize": "12px", "color": TEXT_MUTED}),
            html.Div(f"{m['pm2_median']:,} vs {m['pm2_notaire_median']:,} €/m²".replace(",", " "), style={"fontSize": "20px", "fontWeight": "700", "color": "#f59e0b", "marginTop": "6px"}),
            html.Div(f"Négo conseillée : -{m['nego_moyenne_pct']}%", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
        ]),
    ]

    if sort_val == "decote_hybride":
        df_t = df_f.sort_values("ecart_hybride_pct", ascending=True)
    elif sort_val == "decote_nette" and "decote_nette_pct" in df_f.columns:
        df_t = df_f.sort_values("decote_nette_pct", ascending=True)
    elif sort_val == "decote_avm" and "avm_comp_ecart_pct" in df_f.columns:
        df_t = df_f.sort_values("avm_comp_ecart_pct", ascending=True)
    elif sort_val == "rendement" and "rendement_brut_pct" in df_f.columns:
        df_t = df_f.sort_values("rendement_brut_pct", ascending=False)
    elif sort_val == "decote":
        df_t = df_f.sort_values("ecart_pct", ascending=True)
    else:
        df_t = df_f.sort_values("prix_reel", ascending=True)

    rows_html = []
    for _, r in df_t.head(30).iterrows():
        ad_id = str(r["id"])
        ec_hyb = _safe_float(r.get("ecart_hybride_pct"))
        ec_ml = _safe_float(r.get("ecart_pct"))
        ec_avm = _safe_float(r.get("avm_comp_ecart_pct"))
        cout_trav = _safe_float(r.get("cout_travaux_estime_euro"))
        conf = _safe_int(r.get("score_confiance"))
        conf_col = "#10b981" if conf >= 70 else ("#f59e0b" if conf >= 50 else "#f43f5e")
        badge_bg = "rgba(16,185,129,0.15)" if ec_hyb <= -10 else ("rgba(244,63,94,0.15)" if ec_hyb >= 10 else "rgba(148,163,184,0.15)")
        badge_col = "#10b981" if ec_hyb <= -10 else ("#f43f5e" if ec_hyb >= 10 else TEXT_MUTED)
        url_annonce = str(r.get("url", "") or "")

        rows_html.append(html.Tr(style={"borderBottom": f"1px solid {BORDER_COL}", "fontSize": "12px"}, children=[
            html.Td(style={"padding": "9px 8px"}, children=[
                html.Div(f"{r['quartier_clean']} • {r['type_label']} {r['surface_m2']:.0f} m² ({r['tranche_m2']})", style={"fontWeight": "600"}),
                html.Div(
                    f"DPE {r.get('dpe', 'N/C') if pd.notna(r.get('dpe')) else 'N/C'} • "
                    f"Bus {_safe_int(r.get('dist_arret_bus_m'))}m • Commerces {_safe_int(r.get('nb_commerces_500m'))} • "
                    f"Rdt {_safe_float(r.get('rendement_brut_pct')):.1f}%",
                    style={"color": TEXT_MUTED, "fontSize": "11px"},
                ),
            ]),
            html.Td(f"{conf}/100", style={"padding": "9px 8px", "textAlign": "center", "fontWeight": "700", "color": conf_col}),
            html.Td(fmt_euro(r["prix_reel"]), style={"padding": "9px 8px", "textAlign": "right", "fontWeight": "700"}),
            html.Td(
                style={"padding": "9px 8px", "textAlign": "right"},
                children=[
                    html.Div(fmt_euro(cout_trav) if cout_trav > 0 else "0 €", style={"color": "#f59e0b" if cout_trav > 0 else TEXT_MUTED, "fontWeight": "600"}),
                    html.Div(str(r.get("postes_travaux", ""))[:24], style={"color": TEXT_MUTED, "fontSize": "10px"}),
                ],
            ),
            html.Td(style={"padding": "9px 8px", "textAlign": "right"}, children=[
                html.Div(fmt_euro(r["prix_hybride"]), style={"color": "#10b981", "fontWeight": "700"}),
                html.Span(f"{ec_hyb:+.1f}%", style={"backgroundColor": badge_bg, "color": badge_col, "padding": "2px 6px", "borderRadius": "999px", "fontSize": "11px", "fontWeight": "700"}),
            ]),
            html.Td(style={"padding": "9px 8px", "textAlign": "right"}, children=[
                html.Div(fmt_euro(r.get("avm_comp_blended_euro")), style={"color": "#f59e0b", "fontWeight": "600"}),
                html.Div(f"{ec_avm:+.1f}% ({_safe_int(r.get('avm_comp_nb'))} comp.)", style={"color": TEXT_MUTED, "fontSize": "11px"}),
            ]),
            html.Td(style={"padding": "9px 8px", "textAlign": "right"}, children=[
                html.Div(fmt_euro(r["prix_predit"]), style={"color": "#a855f7", "fontWeight": "600"}),
                html.Div(f"{ec_ml:+.1f}%", style={"color": TEXT_MUTED, "fontSize": "11px"}),
            ]),
            html.Td(style={"padding": "9px 8px", "textAlign": "center", "whiteSpace": "nowrap"}, children=[
                html.Button(
                    "Détail / SHAP",
                    id={"type": "btn-detail-bien", "index": ad_id},
                    n_clicks=0,
                    style={
                        "backgroundColor": "rgba(56,189,248,0.15)", "color": "#38bdf8",
                        "border": "1px solid rgba(56,189,248,0.45)", "borderRadius": "6px",
                        "padding": "5px 10px", "fontSize": "11px", "fontWeight": "700", "cursor": "pointer", "marginRight": "6px",
                    },
                ),
                html.A("↗", href=url_annonce, target="_blank", rel="noopener noreferrer", style={"color": TEXT_MUTED, "textDecoration": "none", "fontWeight": "700"}) if url_annonce else "",
            ]),
        ]))

    table = html.Table(style={"width": "100%", "borderCollapse": "collapse", "textAlign": "left"}, children=[
        html.Thead(html.Tr(style={"borderBottom": f"1px solid {BORDER_COL}", "color": TEXT_MUTED, "fontSize": "12px"}, children=[
            html.Th("Bien & Caractéristiques", style={"padding": "9px 8px"}),
            html.Th("Confiance", style={"padding": "9px 8px", "textAlign": "center"}),
            html.Th("Prix Affiché", style={"padding": "9px 8px", "textAlign": "right"}),
            html.Th("Travaux Est.", style={"padding": "9px 8px", "textAlign": "right"}),
            html.Th("Hybride (75/25)", style={"padding": "9px 8px", "textAlign": "right"}),
            html.Th("AVM Comp.", style={"padding": "9px 8px", "textAlign": "right"}),
            html.Th("ML (HistGB)", style={"padding": "9px 8px", "textAlign": "right"}),
            html.Th("Actions", style={"padding": "9px 8px", "textAlign": "center"}),
        ])),
        html.Tbody(rows_html),
    ])

    detail_options = [
        {
            "label": f"{r['quartier_clean']} • {r['type_label']} {r['surface_m2']:.0f}m² • {fmt_euro(r['prix_reel'])} (Hybride {r['ecart_hybride_pct']:+.1f}%)",
            "value": str(r["id"]),
        }
        for _, r in df_t.head(30).iterrows()
    ]
    detail_default = detail_options[0]["value"] if detail_options else None

    return kpis_cards, creer_fig_courbe(df_f), creer_fig_scatter(df_f), table, detail_options, detail_default


@app.callback(
    Output("detail-bien-selector", "value", allow_duplicate=True),
    Input({"type": "btn-detail-bien", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def selectionner_bien_par_bouton(n_clicks_list):
    if not n_clicks_list or not any(n_clicks_list):
        return None
    trig = ctx.triggered_id
    if isinstance(trig, dict) and "index" in trig:
        return trig["index"]
    return None


@app.callback(
    Output("detail-bien-panel", "children"),
    Input("detail-bien-selector", "value"),
)
def afficher_detail_bien(bien_id: str | None):
    if not bien_id or DF_PRED.empty:
        return html.Div("Clique sur 'Détail / SHAP' sur une offre du tableau.", style={"color": TEXT_MUTED, "fontSize": "13px"})
    sub = DF_PRED[DF_PRED["id"].astype(str) == str(bien_id)]
    if sub.empty:
        return html.Div("Bien introuvable.", style={"color": TEXT_MUTED, "fontSize": "13px"})
    r = sub.iloc[0]

    # 1. Scores comparés : Hybride / AVM / ML / Confiance / Travaux
    cout_trav = _safe_float(r.get("cout_travaux_estime_euro"))
    trav_min = _safe_float(r.get("cout_travaux_min_euro"), cout_trav * 0.78)
    trav_max = _safe_float(r.get("cout_travaux_max_euro"), cout_trav * 1.25)
    trav_txt = f"{fmt_euro(cout_trav)} [{fmt_euro(trav_min)} – {fmt_euro(trav_max)}] • {r.get('postes_travaux', '')}" if cout_trav > 0 else "Aucun travaux majeur détecté (0 €)"

    bandeau_scores = html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px", "marginBottom": "16px"},
        children=[
            html.Div(style={"backgroundColor": BG_MAIN, "border": f"1px solid {BORDER_COL}", "borderRadius": "8px", "padding": "10px"}, children=[
                html.Div("Score Hybride (75% ML + 25% AVM)", style={"fontSize": "11px", "color": TEXT_MUTED}),
                html.Div(f"{fmt_euro(r['prix_hybride'])} ({_safe_float(r.get('ecart_hybride_pct')):+.1f}%)", style={"fontSize": "16px", "fontWeight": "700", "color": "#10b981"}),
            ]),
            html.Div(style={"backgroundColor": BG_MAIN, "border": f"1px solid {BORDER_COL}", "borderRadius": "8px", "padding": "10px"}, children=[
                html.Div("Score AVM Comparables (Travaux incl.)", style={"fontSize": "11px", "color": TEXT_MUTED}),
                html.Div(f"{fmt_euro(r.get('avm_comp_blended_euro'))} ({_safe_float(r.get('avm_comp_ecart_pct')):+.1f}%)", style={"fontSize": "16px", "fontWeight": "700", "color": "#f59e0b"}),
            ]),
            html.Div(style={"backgroundColor": BG_MAIN, "border": f"1px solid {BORDER_COL}", "borderRadius": "8px", "padding": "10px"}, children=[
                html.Div("Score ML (HistGB) & [Q10–Q90]", style={"fontSize": "11px", "color": TEXT_MUTED}),
                html.Div(f"{fmt_euro(r['prix_predit'])} ({_safe_float(r.get('ecart_pct')):+.1f}%)", style={"fontSize": "16px", "fontWeight": "700", "color": "#a855f7"}),
                html.Div(f"[{fmt_euro(r.get('prix_predit_q10'))} – {fmt_euro(r.get('prix_predit_q90'))}]", style={"fontSize": "11px", "color": TEXT_MUTED}),
            ]),
            html.Div(style={"backgroundColor": BG_MAIN, "border": f"1px solid {BORDER_COL}", "borderRadius": "8px", "padding": "10px"}, children=[
                html.Div("Score de Confiance & Travaux", style={"fontSize": "11px", "color": TEXT_MUTED}),
                html.Div(f"Confiance : {_safe_int(r.get('score_confiance'))}/100", style={"fontSize": "16px", "fontWeight": "700", "color": "#38bdf8"}),
                html.Div(trav_txt[:65], style={"fontSize": "11px", "color": "#f59e0b" if cout_trav > 0 else TEXT_MUTED}),
            ]),
        ],
    )

    # 2. Graphique SHAP
    fig_shap = go.Figure()
    try:
        shap_data = json.loads(str(r.get("shap_top_facteurs_json", "{}") or "{}"))
        facteurs = shap_data.get("facteurs", [])
        if facteurs:
            f_ord = sorted(facteurs, key=lambda x: x["impact_euro"])
            fig_shap.add_trace(go.Bar(
                y=[f["label"] for f in f_ord], x=[f["impact_euro"] for f in f_ord], orientation="h",
                marker_color=["#10b981" if f["impact_euro"] >= 0 else "#f43f5e" for f in f_ord],
                text=[f"{f['impact_euro']:+,.0f} €".replace(",", " ") for f in f_ord], textposition="outside",
            ))
    except Exception:
        pass
    fig_shap.update_layout(
        template="plotly_dark", paper_bgcolor=BG_MAIN, plot_bgcolor=BG_MAIN,
        height=270, margin=dict(l=170, r=60, t=30, b=30),
        title="Explication SHAP : pourquoi ce prix estimé ?", xaxis_title="Impact (€)",
    )

    # 3. Toutes les caractéristiques du bien
    caracs = [
        ("Type / Catégorie", f"{r['type_label']} ({r['tranche_m2']})"),
        ("Ville / Quartier", f"{r.get('ville', 'Poitiers')} ({r.get('code_postal', '')}) • {r['quartier_clean']}"),
        ("Surface (Carrez / Utile)", f"{r['surface_m2']:.1f} m² (Carrez : {r.get('surface_carrez_m2') if pd.notna(r.get('surface_carrez_m2')) else 'N/C'} | Utile : {r.get('surface_utile_m2', r['surface_m2']):.1f} m²)"),
        ("Pièces / Chambres / Étage", f"{r.get('pieces', 'N/C')} p. • {r.get('nbr_chambres', 'N/C')} ch. • Étage {r.get('etage', 'N/C')}"),
        ("DPE / GES / Année", f"DPE {r.get('dpe', 'N/C')} • GES {r.get('ges', 'N/C')} • Année {r.get('annee_construction', 'N/C')}"),
        ("Zone IRIS / QPV", f"IRIS {r.get('iris_code', 'N/C')} • QPV : {'Oui' if _safe_int(r.get('est_en_qpv')) == 1 else 'Non'}"),
        ("Mobilité & Commerces", f"Bus à {_safe_int(r.get('dist_arret_bus_m'))} m ({_safe_int(r.get('nb_arrets_bus_400m'))} arrêts/400m) • {_safe_int(r.get('nb_commerces_500m'))} commerces/500m"),
        ("Loyer & Rendement", f"{fmt_euro(r.get('loyer_mensuel_estime'))}/mois • Brut {_safe_float(r.get('rendement_brut_pct')):.2f}% • Net {_safe_float(r.get('rendement_net_estime_pct')):.2f}%"),
        ("Vision OpenCLIP (Photos)", f"Travaux : {_safe_float(r.get('clip_score_travaux'))*100:.0f}% | Rénové : {_safe_float(r.get('clip_score_renove'))*100:.0f}%" if pd.notna(r.get("clip_score_travaux")) else "En attente (disque < 605 Mo pour CLIP-ViT-B-32)"),
        ("Négo & Source", f"Offre cible : {fmt_euro(r['prix_cible_nego'])} (-{r['taux_nego_estime_pct']:.1f}%) • Source : {r.get('source', 'N/C')}"),
    ]
    carac_grid = html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))", "gap": "8px", "marginBottom": "14px"},
        children=[
            html.Div(style={"backgroundColor": BG_MAIN, "padding": "7px 10px", "borderRadius": "6px", "fontSize": "12px"}, children=[
                html.Span(f"{k} : ", style={"color": TEXT_MUTED}),
                html.B(str(v)),
            ])
            for k, v in caracs
        ],
    )

    # 4. Comparables AVM
    comp_rows = []
    try:
        for c in json.loads(str(r.get("avm_comparables_json", "[]") or "[]")):
            comp_rows.append(html.Tr(style={"borderBottom": f"1px solid {BORDER_COL}", "fontSize": "12px"}, children=[
                html.Td(c.get("source", ""), style={"padding": "6px"}),
                html.Td(c.get("titre", "")[:40], style={"padding": "6px"}),
                html.Td(f"{c.get('distance_m', 0)} m", style={"padding": "6px", "textAlign": "right"}),
                html.Td(f"{c.get('surface_m2', 0):.0f} m²", style={"padding": "6px", "textAlign": "right"}),
                html.Td(fmt_euro(c.get("prix", 0)), style={"padding": "6px", "textAlign": "right", "fontWeight": "600"}),
                html.Td(fmt_euro(c.get("travaux_euro", 0)), style={"padding": "6px", "textAlign": "right", "color": "#f59e0b"}),
                html.Td(f"{c.get('similarite_pct', 0):.0f}%", style={"padding": "6px", "textAlign": "right", "color": "#10b981", "fontWeight": "700"}),
            ]))
    except Exception:
        pass
    comp_table = html.Table(style={"width": "100%", "borderCollapse": "collapse"}, children=[
        html.Thead(html.Tr(style={"borderBottom": f"1px solid {BORDER_COL}", "color": TEXT_MUTED, "fontSize": "11px"}, children=[
            html.Th("Source", style={"padding": "6px"}), html.Th("Bien comparable", style={"padding": "6px"}),
            html.Th("Dist.", style={"padding": "6px", "textAlign": "right"}), html.Th("Surf.", style={"padding": "6px", "textAlign": "right"}),
            html.Th("Prix", style={"padding": "6px", "textAlign": "right"}), html.Th("Travaux", style={"padding": "6px", "textAlign": "right"}),
            html.Th("Simil.", style={"padding": "6px", "textAlign": "right"}),
        ])),
        html.Tbody(comp_rows if comp_rows else [html.Tr([html.Td("Aucun comparable dans le rayon.", colSpan=7, style={"padding": "8px", "color": TEXT_MUTED})])]),
    ])

    return html.Div([
        bandeau_scores,
        carac_grid,
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(460px, 1fr))", "gap": "18px"}, children=[
            html.Div([dcc.Graph(figure=fig_shap, config={"displayModeBar": False})]),
            html.Div([html.H3("5 Biens Comparables les Plus Proches (AVM + Travaux)", style={"fontSize": "14px", "margin": "0 0 8px 0"}), comp_table]),
        ]),
    ])


def lancer_dashboard(port: int = 8050, debug: bool = True):
    print(f"Lancement du Dashboard interactif sur http://127.0.0.1:{port}")
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    lancer_dashboard(port=8050, debug=True)
