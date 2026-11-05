"""
Dashboard interactif Dash (Plotly) — IA Immobilier Poitiers.

S'appuie intégralement sur FeatureEngineering.py (aucun doublon de logique métier) :
- Prédictions IA (Gradient Boosting + Validation Croisée)
- Enrichissement qualitatif LLM / Ollama (Note de l'offre, État, Travaux, Résumé)
- Ventes réelles des Notaires (DVF Poitiers) & Taux de négociation estimé

Lancement :
    python Dashboard.py
Puis ouvrir http://127.0.0.1:8050 dans le navigateur.
"""

from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go

import FeatureEngineering as fe

# 1. Chargement et évaluation complète via FeatureEngineering.py uniquement
DF_PROPRE = fe.charger_et_preparer("annonces.csv", verbose=True)
EVAL_DATA = fe.evaluer_complet(DF_PROPRE, afficher=True)

DF_PRED = EVAL_DATA["df_pred"]
IMPORTANCES = EVAL_DATA["importances"]
QUARTIERS = [q for q in EVAL_DATA["quartiers"] if q["quartier"] != "Inconnu"]
LISTE_QUARTIERS = sorted(DF_PRED["quartier_clean"].unique().tolist())

BG_MAIN = "#0f172a"
BG_CARD = "#1e293b"
BORDER_COL = "#334155"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"


def fmt_euro(val: float) -> str:
    return f"{int(round(val)):,} €".replace(",", " ")


def creer_fig_courbe(df_sub):
    """Graphique 1 : Courbe des Prix Réels vs Prédictions IA vs Prix Notaire DVF."""
    fig = go.Figure()
    if len(df_sub) == 0:
        return fig

    df_s = df_sub.sort_values("prix_predit").reset_index(drop=True)
    x_idx = list(range(1, len(df_s) + 1))

    x_lines, y_lines = [], []
    for i, row in df_s.iterrows():
        x_lines.extend([i + 1, i + 1, None])
        y_lines.extend([row["prix_predit"], row["prix_reel"], None])

    fig.add_trace(
        go.Scatter(
            x=x_lines,
            y=y_lines,
            mode="lines",
            line={"color": "rgba(148, 163, 184, 0.35)", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    couleurs = [
        "#10b981" if p <= -10 else ("#ef4444" if p >= 10 else "#94a3b8")
        for p in df_s["ecart_pct"]
    ]
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=df_s["prix_reel"],
            mode="markers",
            name="Prix affiché",
            marker={"color": couleurs, "size": 7.5},
            customdata=df_s[
                [
                    "type_label",
                    "surface_m2",
                    "quartier_clean",
                    "prix_predit",
                    "ecart_pct",
                    "prix_notaire_dvf",
                    "taux_nego_estime_pct",
                    "prix_cible_nego",
                    "llm_note_offre",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]} • %{customdata[1]:.0f} m² (%{customdata[2]})</b> • Note LLM : %{customdata[8]}/10<br>"
                "Prix affiché : <b>%{y:,.0f} €</b><br>"
                "Estimation IA : <b>%{customdata[3]:,.0f} €</b> (Écart : %{customdata[4]:+.1f}%)<br>"
                "Référence Notaire DVF : <b>%{customdata[5]:,.0f} €</b><br>"
                "Négo possible : <b>-%{customdata[6]:.1f}% → Cible %{customdata[7]:,.0f} €</b><extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=df_s["prix_predit"],
            mode="lines",
            name="Prédiction IA",
            line={"color": "#3b82f6", "width": 3},
            hovertemplate="Estimation IA : <b>%{y:,.0f} €</b><extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN, "size": 12},
        margin={"l": 55, "r": 15, "t": 15, "b": 40},
        height=320,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis={"title": "Annonces classées par valeur estimée croissante", "gridcolor": "rgba(148,163,184,0.12)"},
        yaxis={"title": "Prix (€)", "gridcolor": "rgba(148,163,184,0.15)"},
    )
    return fig


def creer_fig_scatter(df_sub):
    """Graphique 2 : Nuage Prix Affiché vs Prix Estimé + diagonale y=x."""
    fig = go.Figure()
    if len(df_sub) == 0:
        return fig

    max_p = float(max(df_sub["prix_reel"].max(), df_sub["prix_predit"].max()) * 1.05)

    fig.add_trace(
        go.Scatter(
            x=[0, max_p],
            y=[0, max_p],
            mode="lines",
            name="Équilibre (Réel = Prédit)",
            line={"color": "#3b82f6", "dash": "dash", "width": 2},
            hoverinfo="skip",
        )
    )

    couleurs = [
        "#10b981" if p <= -10 else ("#ef4444" if p >= 10 else "#6366f1")
        for p in df_sub["ecart_pct"]
    ]
    fig.add_trace(
        go.Scatter(
            x=df_sub["prix_predit"],
            y=df_sub["prix_reel"],
            mode="markers",
            name="Biens",
            marker={"color": couleurs, "size": 8.5, "opacity": 0.9},
            customdata=df_sub[
                [
                    "type_label",
                    "surface_m2",
                    "quartier_clean",
                    "ecart_pct",
                    "prix_notaire_dvf",
                    "taux_nego_estime_pct",
                    "prix_cible_nego",
                    "llm_resume",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]} • %{customdata[1]:.0f} m² (%{customdata[2]})</b><br>"
                "%{customdata[7]}<br>"
                "Prix affiché : <b>%{y:,.0f} €</b> | IA : <b>%{x:,.0f} €</b> (%{customdata[3]:+.1f}%)<br>"
                "Notaire DVF voisinage : <b>%{customdata[4]:,.0f} €</b><br>"
                "Marge négo : <b>-%{customdata[5]:.1f}% → Offre conseillée %{customdata[6]:,.0f} €</b><extra></extra>"
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN, "size": 12},
        margin={"l": 55, "r": 15, "t": 15, "b": 40},
        height=320,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis={"title": "Estimation IA (€)", "gridcolor": "rgba(148,163,184,0.15)", "range": [0, max_p]},
        yaxis={"title": "Prix Affiché (€)", "gridcolor": "rgba(148,163,184,0.15)", "range": [0, max_p]},
    )
    return fig


def creer_fig_importances():
    """Graphique 3 : Importance des variables calculée par FeatureEngineering.py."""
    imp_rev = list(reversed(IMPORTANCES))
    fig = go.Figure(
        go.Bar(
            x=[r["impact_euro"] for r in imp_rev],
            y=[r["label"] for r in imp_rev],
            orientation="h",
            marker={"color": "#3b82f6"},
            text=[f"+{fmt_euro(r['impact_euro'])}" for r in imp_rev],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Impact moyen sur l'erreur : %{x:,.0f} €<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN, "size": 12},
        margin={"l": 210, "r": 20, "t": 10, "b": 30},
        height=320,
        xaxis={"title": "Impact sur l'erreur (€)", "gridcolor": "rgba(148,163,184,0.15)"},
        yaxis={"gridcolor": "rgba(0,0,0,0)"},
    )
    return fig


def creer_fig_quartiers():
    """Graphique 4 : Prix médian au m² par Quartier (Affiché vs Prédit IA vs Signé Notaire DVF)."""
    q_rev = list(reversed(QUARTIERS[:8]))
    labels = [f"{q['quartier']} ({q['nb']})" for q in q_rev]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[q["pm2_notaire"] for q in q_rev],
            name="Vendu Notaire DVF (€/m²)",
            orientation="h",
            marker={"color": "#f59e0b"},
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[q["pm2_predit"] for q in q_rev],
            name="Prédit IA (€/m²)",
            orientation="h",
            marker={"color": "#14b8a6"},
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[q["pm2_reel"] for q in q_rev],
            name="Affiché Annonce (€/m²)",
            orientation="h",
            marker={"color": "#6366f1"},
        )
    )
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN, "size": 12},
        margin={"l": 190, "r": 20, "t": 10, "b": 30},
        height=320,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis={"title": "Prix médian (€/m²)", "gridcolor": "rgba(148,163,184,0.15)"},
    )
    return fig


app = Dash(__name__)
app.title = "Dashboard IA Immobilier — Poitiers"

CARD_STYLE = {
    "backgroundColor": BG_CARD,
    "border": f"1px solid {BORDER_COL}",
    "borderRadius": "12px",
    "padding": "18px",
}

app.layout = html.Div(
    style={
        "backgroundColor": BG_MAIN,
        "color": TEXT_MAIN,
        "minHeight": "100vh",
        "padding": "24px",
        "fontFamily": "system-ui, -apple-system, sans-serif",
    },
    children=[
        html.Div(
            style={"maxWidth": "1360px", "margin": "0 auto", "display": "flex", "flexDirection": "column", "gap": "20px"},
            children=[
                # En-tête & Filtres
                html.Div(
                    style={
                        **CARD_STYLE,
                        "display": "flex",
                        "flexWrap": "wrap",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "gap": "16px",
                    },
                    children=[
                        html.Div([
                            html.H1(
                                "Dashboard IA Immobilier, Notaires DVF & Négociation — Poitiers",
                                style={"fontSize": "20px", "fontWeight": "700", "margin": "0 0 4px 0"},
                            ),
                            html.P(
                                "Propulsé par FeatureEngineering.py (ML + Enrichissement LLM/Ollama + Ventes Notariées DVF)",
                                style={"fontSize": "13px", "color": TEXT_MUTED, "margin": 0},
                            ),
                        ]),
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "12px", "flexWrap": "wrap"},
                            children=[
                                html.Span("Type de bien :", style={"fontSize": "12px", "color": TEXT_MUTED}),
                                dcc.Dropdown(
                                    id="filter-type",
                                    options=[
                                        {"label": "Tous les biens", "value": "ALL"},
                                        {"label": "Appartements", "value": "Appartement"},
                                        {"label": "Maisons", "value": "Maison"},
                                    ],
                                    value="ALL",
                                    clearable=False,
                                    style={"width": "170px", "color": "#0f172a", "fontSize": "13px"},
                                ),
                                html.Span("Quartier :", style={"fontSize": "12px", "color": TEXT_MUTED}),
                                dcc.Dropdown(
                                    id="filter-quartier",
                                    options=[{"label": "Tous les quartiers", "value": "ALL"}]
                                    + [{"label": q, "value": q} for q in LISTE_QUARTIERS],
                                    value="ALL",
                                    clearable=False,
                                    style={"width": "230px", "color": "#0f172a", "fontSize": "13px"},
                                ),
                            ],
                        ),
                    ],
                ),

                # KPIs
                html.Div(
                    id="kpis-container",
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "14px"},
                ),

                # Panneau d'évaluation approfondie
                html.Div(id="eval-detail-container", style=CARD_STYLE),

                # Ligne 1 : Courbe & Nuage
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "20px"},
                    children=[
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.H2("Courbe des Prix Réels vs Prédictions IA", style={"fontSize": "16px", "fontWeight": "600", "margin": "0 0 4px 0"}),
                                html.P("Annonces classées par valeur estimée croissante (Survolez pour voir le prix Notaire DVF et la marge de négo)", style={"fontSize": "12px", "color": TEXT_MUTED, "margin": "0 0 8px 0"}),
                                dcc.Graph(id="graph-curve", config={"displayModeBar": False}),
                            ],
                        ),
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.H2("Prix Affiché vs Prix Estimé (Détection d'opportunités)", style={"fontSize": "16px", "fontWeight": "600", "margin": "0 0 4px 0"}),
                                html.P("Vert = sous-coté (< -10% sous l'estimation IA) • Rouge = sur-coté (> +10%)", style={"fontSize": "12px", "color": TEXT_MUTED, "margin": "0 0 8px 0"}),
                                dcc.Graph(id="graph-scatter", config={"displayModeBar": False}),
                            ],
                        ),
                    ],
                ),

                # Ligne 2 : Importances & Quartiers (Affiché vs IA vs Notaire DVF)
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))", "gap": "20px"},
                    children=[
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.H2("Variables qui influencent le plus le prix (incl. DVF & LLM)", style={"fontSize": "16px", "fontWeight": "600", "margin": "0 0 4px 0"}),
                                html.P("Impact moyen sur l'erreur de prédiction (€) lorsqu'on brouille la variable", style={"fontSize": "12px", "color": TEXT_MUTED, "margin": "0 0 8px 0"}),
                                dcc.Graph(id="graph-importances", figure=creer_fig_importances(), config={"displayModeBar": False}),
                            ],
                        ),
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.H2("Prix/m² par Quartier : Affiché vs IA vs Ventes Notaires (DVF)", style={"fontSize": "16px", "fontWeight": "600", "margin": "0 0 4px 0"}),
                                html.P("Comparaison entre les prix demandés en annonce et les actes de vente notariés réels", style={"fontSize": "12px", "color": TEXT_MUTED, "margin": "0 0 8px 0"}),
                                dcc.Graph(id="graph-quartiers", figure=creer_fig_quartiers(), config={"displayModeBar": False}),
                            ],
                        ),
                    ],
                ),

                # Ligne 3 : Tableau des opportunités + Note LLM + Prix Notaire + Taux de Négo + Lien
                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "12px", "marginBottom": "14px"},
                            children=[
                                html.Div([
                                    html.H2("Top Opportunités Détectées, Analyse LLM, Prix Notaire & Marge de Négociation", style={"fontSize": "16px", "fontWeight": "600", "margin": "0 0 4px 0"}),
                                    html.P("Synthèse par bien : Prix affiché, Estimation IA, Valeur Notaire DVF voisinage, Offre après négociation et Lien direct", style={"fontSize": "12px", "color": TEXT_MUTED, "margin": 0}),
                                ]),
                                dcc.Dropdown(
                                    id="sort-table",
                                    options=[
                                        {"label": "Trier par : Plus forte décote vs IA (%)", "value": "decote"},
                                        {"label": "Trier par : Meilleure note d'offre LLM (/10)", "value": "note_llm"},
                                        {"label": "Trier par : Plus forte marge de négociation (%)", "value": "nego"},
                                        {"label": "Trier par : Prix croissant", "value": "prix_asc"},
                                    ],
                                    value="decote",
                                    clearable=False,
                                    style={"width": "290px", "color": "#0f172a", "fontSize": "13px"},
                                ),
                            ],
                        ),
                        html.Div(id="deals-table-container", style={"overflowX": "auto"}),
                    ],
                ),
            ],
        )
    ],
)


@app.callback(
    Output("kpis-container", "children"),
    Output("eval-detail-container", "children"),
    Output("graph-curve", "figure"),
    Output("graph-scatter", "figure"),
    Output("deals-table-container", "children"),
    Input("filter-type", "value"),
    Input("filter-quartier", "value"),
    Input("sort-table", "value"),
)
def mettre_a_jour_dashboard(type_val: str, quartier_val: str, sort_val: str):
    df_f = DF_PRED.copy()
    if type_val != "ALL":
        df_f = df_f[df_f["type_label"] == type_val]
    if quartier_val != "ALL":
        df_f = df_f[df_f["quartier_clean"] == quartier_val]

    m = fe.calculer_metriques_groupe(df_f)
    if not m:
        return [], html.Div("Aucun bien pour ce filtre."), go.Figure(), go.Figure(), html.Div()

    kpis_cards = [
        html.Div(style=CARD_STYLE, children=[
            html.Div("Biens analysés", style={"fontSize": "12px", "color": TEXT_MUTED}),
            html.Div(f"{m['nb_biens']}", style={"fontSize": "24px", "fontWeight": "700", "marginTop": "4px"}),
            html.Div(f"Prix médian : {fmt_euro(m['prix_median'])}", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.Div("Score R² & Gain IA", style={"fontSize": "12px", "color": TEXT_MUTED}),
            html.Div(f"{m['r2']:.2f}", style={"fontSize": "24px", "fontWeight": "700", "color": "#10b981", "marginTop": "4px"}),
            html.Div(f"+{m['gain_vs_ref_pct']}% vs estimation naïve €/m²", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.Div("Erreur Médiane IA", style={"fontSize": "12px", "color": TEXT_MUTED}),
            html.Div(f"±{m['erreur_mediane_pct']}%", style={"fontSize": "24px", "fontWeight": "700", "color": "#3b82f6", "marginTop": "4px"}),
            html.Div(f"±{fmt_euro(m['med_ae'])} (MAE : {fmt_euro(m['mae'])})", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.Div("Affiché vs Notaire (DVF)", style={"fontSize": "12px", "color": TEXT_MUTED}),
            html.Div(f"{m['pm2_median']:,} vs {m['pm2_notaire_median']:,} €/m²".replace(",", " "), style={"fontSize": "20px", "fontWeight": "700", "color": "#f59e0b", "marginTop": "6px"}),
            html.Div(f"Écart affiché vs signé : {m['ecart_notaire_median_pct']:+.1f}%", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.Div("Marge Négociation Moyenne", style={"fontSize": "12px", "color": TEXT_MUTED}),
            html.Div(f"-{m['nego_moyenne_pct']}%", style={"fontSize": "24px", "fontWeight": "700", "color": "#38bdf8", "marginTop": "4px"}),
            html.Div(f"Économie médiane : -{fmt_euro(m['economie_nego_mediane'])}", style={"fontSize": "12px", "color": TEXT_MUTED, "marginTop": "4px"}),
        ]),
    ]

    paliers = [
        ("Écart ≤ 5 % (Quasi parfait)", m["within_5pct"], "#10b981"),
        ("Écart ≤ 10 % (Très précis)", m["within_10pct"], "#3b82f6"),
        ("Écart ≤ 15 % (Standard marché)", m["within_15pct"], "#6366f1"),
        ("Écart ≤ 20 % (Acceptable)", m["within_20pct"], "#8b5cf6"),
        ("Écart > 25 % (Décotes / Atypiques)", m["outliers_25pct"], "#f59e0b"),
    ]
    eval_panel = html.Div([
        html.Div(
            style={"marginBottom": "12px"},
            children=[
                html.H3("Diagnostic de fiabilité concret : à quel point l'IA tombe juste ?", style={"fontSize": "15px", "fontWeight": "600", "margin": "0 0 2px 0"}),
                html.P(
                    f"Sur {m['nb_biens']} biens, 50% sont estimés à moins de ±{m['erreur_mediane_pct']}% (±{fmt_euro(m['med_ae'])}). "
                    f"Les prix affichés sont en médiane à {m['ecart_notaire_median_pct']:+.1f}% des ventes notariées DVF proches.",
                    style={"fontSize": "12px", "color": TEXT_MUTED, "margin": 0},
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px"},
            children=[
                html.Div(
                    style={"backgroundColor": BG_MAIN, "border": f"1px solid {BORDER_COL}", "borderRadius": "8px", "padding": "10px 12px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "fontSize": "12px", "marginBottom": "6px"},
                            children=[
                                html.Span(label, style={"color": TEXT_MUTED}),
                                html.B(f"{pct:.1f}%", style={"color": col}),
                            ],
                        ),
                        html.Div(
                            style={"width": "100%", "backgroundColor": "#1e293b", "height": "6px", "borderRadius": "4px", "overflow": "hidden"},
                            children=[html.Div(style={"width": f"{min(100, pct)}%", "backgroundColor": col, "height": "6px"})],
                        ),
                    ],
                )
                for label, pct, col in paliers
            ],
        ),
    ])

    if sort_val == "decote":
        df_t = df_f.sort_values("ecart_pct", ascending=True)
    elif sort_val == "note_llm":
        df_t = df_f.sort_values(["llm_note_offre", "ecart_pct"], ascending=[False, True])
    elif sort_val == "nego":
        df_t = df_f.sort_values("taux_nego_estime_pct", ascending=False)
    else:
        df_t = df_f.sort_values("prix_reel", ascending=True)

    rows_html = []
    for _, r in df_t.head(20).iterrows():
        pct = float(r["ecart_pct"])
        ecart_e = float(r["ecart_euro"])
        sign = "+" if pct > 0 else ""
        badge_bg = "rgba(16,185,129,0.15)" if pct <= -10 else ("rgba(244,63,94,0.15)" if pct >= 10 else "rgba(148,163,184,0.15)")
        badge_col = "#10b981" if pct <= -10 else ("#f43f5e" if pct >= 10 else TEXT_MUTED)
        url_annonce = str(r.get("url", "") or "")
        pieces_txt = f" • {int(r['pieces'])} p." if r.get("pieces") and str(r["pieces"]) != "<NA>" else ""

        rows_html.append(
            html.Tr(
                style={"borderBottom": f"1px solid {BORDER_COL}", "fontSize": "12px"},
                children=[
                    html.Td(
                        style={"padding": "10px 10px"},
                        children=[
                            html.Div(f"{r['quartier_clean']} ({r['type_label']} • {r['surface_m2']:.0f} m²{pieces_txt})", style={"fontWeight": "600"}),
                            html.Div(str(r.get("llm_resume", r["titre_propre"]))[:80], style={"color": TEXT_MUTED, "fontSize": "12px"}),
                        ],
                    ),
                    html.Td(
                        f"{r.get('llm_note_offre', 5.0):.1f}/10",
                        style={"padding": "10px 10px", "fontWeight": "700", "color": "#a855f7"},
                    ),
                    html.Td(str(r["dpe"]) if str(r.get("dpe", "")) != "nan" else "N/C", style={"padding": "10px 10px", "fontWeight": "600"}),
                    html.Td(fmt_euro(r["prix_reel"]), style={"padding": "10px 10px", "textAlign": "right", "fontWeight": "600"}),
                    html.Td(fmt_euro(r["prix_predit"]), style={"padding": "10px 10px", "textAlign": "right", "color": "#3b82f6", "fontWeight": "600"}),
                    html.Td(fmt_euro(r["prix_notaire_dvf"]), style={"padding": "10px 10px", "textAlign": "right", "color": "#f59e0b", "fontWeight": "600"}),
                    html.Td(
                        style={"padding": "10px 10px", "textAlign": "right"},
                        children=html.Span(
                            f"{sign}{pct:.1f}%",
                            style={
                                "backgroundColor": badge_bg,
                                "color": badge_col,
                                "padding": "3px 8px",
                                "borderRadius": "999px",
                                "fontWeight": "600",
                            },
                        ),
                    ),
                    html.Td(
                        style={"padding": "10px 10px", "textAlign": "right"},
                        children=[
                            html.Div(f"-{r['taux_nego_estime_pct']:.1f}%", style={"color": "#38bdf8", "fontWeight": "700"}),
                            html.Div(f"→ {fmt_euro(r['prix_cible_nego'])}", style={"color": TEXT_MUTED, "fontSize": "12px"}),
                        ],
                    ),
                    html.Td(
                        style={"padding": "10px 10px", "textAlign": "center"},
                        children=html.A(
                            "Voir l'annonce ↗",
                            href=url_annonce,
                            target="_blank",
                            rel="noopener noreferrer",
                            style={
                                "color": "#38bdf8",
                                "textDecoration": "none",
                                "fontWeight": "600",
                                "border": "1px solid rgba(56,189,248,0.4)",
                                "padding": "4px 10px",
                                "borderRadius": "6px",
                                "display": "inline-block",
                            },
                        )
                        if url_annonce
                        else html.Span("—", style={"color": TEXT_MUTED}),
                    ),
                ],
            )
        )

    table = html.Table(
        style={"width": "100%", "borderCollapse": "collapse", "textAlign": "left"},
        children=[
            html.Thead(
                html.Tr(
                    style={"borderBottom": f"1px solid {BORDER_COL}", "color": TEXT_MUTED, "fontSize": "12px"},
                    children=[
                        html.Th("Bien / Synthèse LLM", style={"padding": "10px 10px"}),
                        html.Th("Note LLM", style={"padding": "10px 10px"}),
                        html.Th("DPE", style={"padding": "10px 10px"}),
                        html.Th("Prix Affiché", style={"padding": "10px 10px", "textAlign": "right"}),
                        html.Th("Estimation IA", style={"padding": "10px 10px", "textAlign": "right"}),
                        html.Th("Notaire DVF", style={"padding": "10px 10px", "textAlign": "right"}),
                        html.Th("Écart IA", style={"padding": "10px 10px", "textAlign": "right"}),
                        html.Th("Négo & Cible", style={"padding": "10px 10px", "textAlign": "right"}),
                        html.Th("Lien Annonce", style={"padding": "10px 10px", "textAlign": "center"}),
                    ],
                )
            ),
            html.Tbody(rows_html),
        ],
    )

    return (
        kpis_cards,
        eval_panel,
        creer_fig_courbe(df_f),
        creer_fig_scatter(df_f),
        table,
    )


if __name__ == "__main__":
    print("\nLancement du Dashboard Dash sur http://127.0.0.1:8050 ...")
    app.run(debug=True, port=8050)
