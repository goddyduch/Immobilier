"""Moteur de simulation d'investissement locatif & cash-flow (calcul pur déterministe)."""

from __future__ import annotations

from backend.schemas.all import SimulateInvestmentRequest, SimulateInvestmentResponse


def simuler_investissement(req: SimulateInvestmentRequest) -> SimulateInvestmentResponse:
    frais_notaire = (
        req.frais_notaire_euro
        if req.frais_notaire_euro is not None
        else round(req.prix_achat_fai * (req.frais_notaire_pct / 100.0), 2)
    )
    travaux = (
        round(req.travaux_ratio_m2 * req.surface_m2, 2)
        if req.travaux_ratio_m2 is not None
        else round(req.travaux_total_euro, 2)
    )
    total_projet = round(
        req.prix_achat_fai
        + req.mobilier_electromenager
        + frais_notaire
        + req.frais_agence_euro
        + req.autres_frais_achat
        + travaux
        + req.frais_bancaires_euro
        + req.frais_cautionnement_euro,
        2,
    )
    emprunt_total = round(max(0.0, total_projet - req.apport_euro), 2)

    total_loyers_mensuels = round(req.loyer_hc_mensuel + req.charges_locataire_mensuel, 2)
    taux_occupation = max(0.0, 1.0 - req.vacance_mois_an / 12.0)
    loyer_effectif_hc = req.loyer_hc_mensuel * taux_occupation
    loyer_effectif_total = round(loyer_effectif_hc + req.charges_locataire_mensuel * taux_occupation, 2)

    charges_fixes_mensuelles = (
        req.frais_relocation_annuel
        + req.assurance_pno_annuel
        + req.taxe_fonciere_annuel
        + req.charges_copro_proprio_annuel
        + req.autres_charges_annuel
        + req.frais_comptable_annuel
        + req.cfe_annuel
    ) / 12.0
    charges_var_mensuelles = loyer_effectif_hc * ((req.gestion_agence_pct + req.gli_pct) / 100.0)
    total_charges_mensuelles = round(charges_fixes_mensuelles + charges_var_mensuelles, 2)

    # Mensualité crédit amortissable + assurance emprunteur sur capital initial
    if emprunt_total <= 0:
        mensualite_credit = 0.0
    else:
        n_mois = req.duree_credit_ans * 12
        t_m = (req.taux_credit_pct / 100.0) / 12.0
        if t_m > 0:
            mens_hors_assu = emprunt_total * t_m / (1.0 - (1.0 + t_m) ** (-n_mois))
        else:
            mens_hors_assu = emprunt_total / n_mois
        assu_mens = emprunt_total * ((req.taux_assurance_pct / 100.0) / 12.0)
        mensualite_credit = round(mens_hors_assu + assu_mens, 2)

    total_sorties = round(total_charges_mensuelles + mensualite_credit, 2)
    cash_flow = round(loyer_effectif_total - total_sorties, 2)
    taux_couverture = (
        round((loyer_effectif_total / mensualite_credit) * 100.0, 1)
        if mensualite_credit > 0
        else None
    )

    rendement_brut = round((req.loyer_hc_mensuel * 12.0) / max(total_projet, 1.0) * 100.0, 2)
    rendement_net = round(
        ((loyer_effectif_hc - total_charges_mensuelles) * 12.0) / max(total_projet, 1.0) * 100.0,
        2,
    )

    dpe_final = (req.dpe_apres if req.dpe_apres not in ("NC", "", None) else req.dpe_avant).upper()
    alerte_dpe = None
    if dpe_final == "G":
        alerte_dpe = "DPE G : logement interdit à la location (loi Climat & Résilience). Travaux d'isolation indispensables."
    elif dpe_final == "F":
        alerte_dpe = "DPE F : interdiction de location prévue en 2028. Anticiper la rénovation énergétique."

    return SimulateInvestmentResponse(
        frais_notaire_euro=frais_notaire,
        travaux_total_euro=travaux,
        total_projet_euro=total_projet,
        emprunt_total_euro=emprunt_total,
        total_loyers_mensuels_euro=total_loyers_mensuels,
        loyer_effectif_mensuel_euro=loyer_effectif_total,
        total_charges_mensuelles_euro=total_charges_mensuelles,
        mensualite_credit_euro=mensualite_credit,
        total_sorties_mensuelles_euro=total_sorties,
        cash_flow_mensuel_euro=cash_flow,
        taux_couverture_pct=taux_couverture,
        rendement_brut_pct=rendement_brut,
        rendement_net_pct=rendement_net,
        alerte_dpe=alerte_dpe,
    )
