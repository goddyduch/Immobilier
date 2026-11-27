"""Schémas Pydantic complets pour tous les endpoints API V1."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    db_available: bool
    nb_offres_predictions: int
    nb_dvf_reference: int


class ShapFactor(BaseModel):
    feature: str
    label: str
    impact_euro: float
    impact_pct: float = 0.0
    description: str = ""


class ComparableItem(BaseModel):
    source: str
    titre: str
    distance_m: int
    surface_m2: float
    prix: float
    pm2: float
    pm2_ajuste: float | None = None
    poids_pct: float | None = None
    date_transaction: str = ""
    age_mois: float = 0.0
    quartier: str = ""
    pieces: int | None = None
    dpe: str | None = None
    a_exterieur: bool = False
    a_parking: bool = False
    a_ascenseur: bool = False
    travaux_euro: float = 0.0
    similarite_pct: float
    url: str = ""


class OpportunityItem(BaseModel):
    id: str
    titre: str | None = None
    ville: str | None = None
    quartier: str | None = None
    type_bien: str | None = None
    type_label: str | None = None
    surface_m2: float
    pieces: float | None = None
    dpe: str | None = None
    prix_affiche: float
    prix_estime_ml: float
    prix_hybride: float
    prix_q10: float
    prix_q90: float
    intervalle_largeur_pct: float
    valeur_estimee_renovee: float
    cout_travaux_estime: float
    niveau_travaux_cat: int = 0
    ecart_brut_pct: float
    decote_nette_pct: float
    score_confiance: int
    verdict: str
    loyer_mensuel_estime: float | None = None
    rendement_brut_pct: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    url: str | None = None
    photos_urls: list[str] = Field(default_factory=list)
    source: str | None = None


class OpportunityDetail(OpportunityItem):
    description: str = ""
    ges: str | None = None
    etage: float | None = None
    annee_construction: float | None = None
    prix_notaire_dvf: float | None = None
    taux_nego_estime_pct: float | None = None
    prix_cible_nego: float | None = None
    loyer_m2_estime: float | None = None
    loyer_bas: float | None = None
    loyer_haut: float | None = None
    avm_comp_blended_euro: float | None = None
    avm_comp_ecart_pct: float | None = None
    clip_etat_score: float | None = None
    clip_score_travaux: float | None = None
    shap_facteurs: list[ShapFactor] = Field(default_factory=list)
    shap_resume: str = ""
    comparables: list[ComparableItem] = Field(default_factory=list)


class OpportunitiesKpis(BaseModel):
    total_biens: int
    nb_opportunites_fortes: int
    nb_sous_evalues: int
    decote_mediane_pct: float | None = None
    confiance_mediane: float | None = None
    rendement_brut_median_pct: float | None = None


class OpportunitiesFacets(BaseModel):
    villes: list[str]
    quartiers: list[str]
    tranches_m2: list[str]


class OpportunitiesResponse(BaseModel):
    kpis: OpportunitiesKpis
    facets: OpportunitiesFacets
    total_filtered: int
    items: list[OpportunityItem]
    fraicheur_villes: dict[str, Any] = Field(default_factory=dict)


class EstimateRequest(BaseModel):
    type_bien: Literal["flat", "house"] = "flat"
    surface_m2: float = Field(..., gt=9, le=1000)
    pieces: int = Field(default=3, ge=1, le=30)
    nbr_chambres: int | None = Field(default=None, ge=0, le=20)
    ville: str = ""
    code_postal: str = ""
    quartier: str = ""
    latitude: float | None = None
    longitude: float | None = None
    etage: float | None = None
    dpe: Literal["A", "B", "C", "D", "E", "F", "G"] | None = "D"
    ges: Literal["A", "B", "C", "D", "E", "F", "G"] | None = None
    annee_construction: int | None = Field(default=None, ge=1700, le=2030)
    prix_affiche: float | None = Field(default=None, gt=10000)
    niveau_travaux_manuel: int | None = Field(default=None, ge=0, le=4)
    description: str = ""
    has_ascenseur: bool = False
    has_terrasse: bool = False
    has_jardin: bool = False
    has_balcon: bool = False
    has_parking: bool = False
    has_cave: bool = False


class EstimateResponse(BaseModel):
    prix_ml: float
    prix_q10: float
    prix_q90: float
    intervalle_largeur_pct: float
    prix_avm: float | None = None
    prix_hybride: float
    valeur_estimee_renovee: float
    cout_travaux_estime: float
    niveau_travaux_cat: int
    ecart_brut_pct: float | None = None
    decote_nette_pct: float | None = None
    score_confiance: int
    verdict: str
    taux_nego_estime_pct: float
    prix_cible_nego: float
    loyer_mensuel_estime: float
    loyer_bas: float
    loyer_haut: float
    rendement_brut_pct: float | None = None
    ville_couverte: bool = True
    avertissement_couverture: str | None = None
    shap_facteurs: list[ShapFactor]
    comparables: list[ComparableItem]
    latence_ms: float


class CheckUrlRequest(BaseModel):
    url: str = Field(..., min_length=8)


class CheckUrlResponse(BaseModel):
    status: Literal["found_in_db", "analyzed_live", "blocked_or_unavailable"]
    message: str
    en_ligne: bool | None = None
    detail: OpportunityDetail | None = None


class SimulateInvestmentRequest(BaseModel):
    prix_achat_fai: float = Field(..., gt=0)
    mobilier_electromenager: float = Field(default=0.0, ge=0)
    frais_notaire_euro: float | None = Field(default=None, ge=0)
    frais_notaire_pct: float = Field(default=8.0, ge=0, le=15)
    frais_agence_euro: float = Field(default=0.0, ge=0)
    autres_frais_achat: float = Field(default=0.0, ge=0)
    surface_m2: float = Field(default=50.0, gt=0)
    travaux_ratio_m2: float | None = Field(default=None, ge=0)
    travaux_total_euro: float = Field(default=0.0, ge=0)
    dpe_avant: str = "NC"
    dpe_apres: str = "NC"
    loyer_hc_mensuel: float = Field(..., ge=0)
    charges_locataire_mensuel: float = Field(default=0.0, ge=0)
    gestion_agence_pct: float = Field(default=0.0, ge=0, le=20)
    frais_relocation_annuel: float = Field(default=0.0, ge=0)
    gli_pct: float = Field(default=0.0, ge=0, le=10)
    assurance_pno_annuel: float = Field(default=200.0, ge=0)
    taxe_fonciere_annuel: float = Field(default=600.0, ge=0)
    vacance_mois_an: float = Field(default=0.5, ge=0, le=12)
    charges_copro_proprio_annuel: float = Field(default=300.0, ge=0)
    autres_charges_annuel: float = Field(default=0.0, ge=0)
    frais_comptable_annuel: float = Field(default=0.0, ge=0)
    cfe_annuel: float = Field(default=0.0, ge=0)
    apport_euro: float = Field(default=0.0, ge=0)
    duree_credit_ans: int = Field(default=20, ge=1, le=35)
    taux_credit_pct: float = Field(default=3.5, ge=0, le=15)
    taux_assurance_pct: float = Field(default=0.30, ge=0, le=5)
    frais_bancaires_euro: float = Field(default=0.0, ge=0)
    frais_cautionnement_euro: float = Field(default=0.0, ge=0)


class SimulateInvestmentResponse(BaseModel):
    frais_notaire_euro: float
    travaux_total_euro: float
    total_projet_euro: float
    emprunt_total_euro: float
    total_loyers_mensuels_euro: float
    loyer_effectif_mensuel_euro: float
    total_charges_mensuelles_euro: float
    mensualite_credit_euro: float
    total_sorties_mensuelles_euro: float
    cash_flow_mensuel_euro: float
    taux_couverture_pct: float | None = None
    rendement_brut_pct: float
    rendement_net_pct: float
    alerte_dpe: str | None = None


class DvfTrancheStat(BaseModel):
    tranche_m2: str
    nb_biens: int
    pm2_affiche_median: float
    pm2_notaire_median: float
    ecart_affiche_vs_dvf_pct: float
    taux_nego_conseille_pct: float


class DvfAnalysisResponse(BaseModel):
    nb_offres: int
    nb_dvf_total: int
    pm2_affiche_median: float
    pm2_notaire_median: float
    ecart_global_pct: float
    taux_nego_median_pct: float
    economie_mediane_euro: float
    par_tranche: list[DvfTrancheStat]
    par_dpe: list[dict[str, Any]]
    villes_disponibles: list[str] = Field(default_factory=list)


class EvaluationResponse(BaseModel):
    model_version: str
    date_evaluation: str
    approche_retenue: str
    global_metrics: dict[str, Any]
    comparatif_modeles: dict[str, Any]
    par_ville: list[dict[str, Any]]
    par_categorie_taille: list[dict[str, Any]]
    avant_apres: dict[str, Any]


class NegotiationRequest(BaseModel):
    prix_affiche: float = Field(..., gt=0)
    surface_m2: float = Field(..., gt=9)
    dpe: str = "D"
    travaux_euro: float = Field(default=0.0, ge=0)
    prix_notaire_ref_m2: float | None = None


class NegotiationResponse(BaseModel):
    prix_ref_dvf_total: float | None = None
    offre_prudente: float
    offre_conseillee: float
    offre_offensive: float
    marge_conseillee_pct: float
    economie_estimee_euro: float
    arguments: list[str]
