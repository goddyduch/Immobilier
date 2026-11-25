import pandas as pd

from src.ingestion import deduplication as Deduplication
from src.ml import features as fe
from src.ingestion import merge as Merge
from src.monitoring import drift as monitoring


def test_data_contracts_quarantaine():
    df = pd.DataFrame([
        {"id": "ok1", "prix": 180000, "surface_m2": 75, "code_postal": "86000", "description": "Bel<br>appartement"},
        {"id": "ko_prix", "prix": 2000, "surface_m2": 50, "code_postal": "86000", "description": "Erreur"},
        {"id": "ko_pm2", "prix": 900000, "surface_m2": 20, "code_postal": "86000", "description": "Aberrant"},
        {"id": "ko_cp", "prix": 150000, "surface_m2": 60, "code_postal": "ABC", "description": "CP faux"},
    ])
    valides, rejets = Merge.nettoyer_et_valider_data_contracts(df)
    assert list(valides["id"]) == ["ok1"]
    assert "<br>" not in valides.iloc[0]["description"]
    assert set(rejets["id"]) == {"ko_prix", "ko_pm2", "ko_cp"}


def test_deduplication_exacte_et_floue():
    df = pd.DataFrame([
        {"id": "bi_1", "source": "bienici", "prix": 200000, "surface_m2": 80.0, "code_postal": "86000", "type_bien": "flat", "latitude": 46.580, "longitude": 0.340, "dpe": "C"},
        {"id": "jk_1", "source": "jinka", "prix": 204000, "surface_m2": 81.0, "code_postal": "86000", "type_bien": "flat", "latitude": 46.5805, "longitude": 0.3405},
        {"id": "bi_2", "source": "bienici", "prix": 300000, "surface_m2": 120.0, "code_postal": "86000", "type_bien": "house", "latitude": 46.590, "longitude": 0.350},
    ])
    clean, suppr = Deduplication.dedupliquer_dataframe(df)
    assert suppr == 1
    assert set(clean["id"]) == {"bi_1", "bi_2"}


def test_grille_travaux_4_categories():
    df = pd.DataFrame([
        {"type_bien": "flat", "surface_m2": 50, "dpe": "B", "description": "Appartement refait à neuf, aucun travaux.", "llm_travaux_score": 0, "llm_etat_score": 5, "clip_score_travaux": 0.1, "clip_score_renove": 0.8},
        {"type_bien": "flat", "surface_m2": 50, "dpe": "D", "description": "À rafraîchir, peinture à prévoir.", "llm_travaux_score": 1, "llm_etat_score": 3, "clip_score_travaux": 0.3},
        {"type_bien": "house", "surface_m2": 50, "dpe": "E", "description": "Maison entièrement à rénover, rénovation complète.", "llm_travaux_score": 3, "llm_etat_score": 2, "clip_score_travaux": 0.5},
        {"type_bien": "house", "surface_m2": 50, "dpe": "G", "description": "Gros œuvre, charpente à refaire, rénovation lourde.", "llm_travaux_score": 3, "llm_etat_score": 1, "clip_score_travaux": 0.8},
    ])
    res = fe.estimer_cout_travaux(df)
    assert list(res["niveau_travaux_cat"]) == [0, 1, 3, 4]
    assert res["cout_travaux_estime_euro"].iloc[0] == 0.0
    assert res["cout_travaux_estime_euro"].iloc[1] == 20000.0  # 50 * 400
    assert res["cout_travaux_estime_euro"].iloc[3] > res["cout_travaux_estime_euro"].iloc[2]


def test_psi_stabilite():
    ref = pd.Series(range(100))
    cur_ok = pd.Series(range(100))
    cur_drift = pd.Series(range(150, 250))
    assert monitoring.calculer_psi(ref, cur_ok) == 0.0
    assert monitoring.calculer_psi(ref, cur_drift) > 0.25
