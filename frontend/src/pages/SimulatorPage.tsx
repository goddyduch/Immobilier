import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { api } from "../api/client";

const fmt = (v?: number | null) =>
  v === undefined || v === null ? "—" : `${Math.round(v).toLocaleString("fr-FR")} €`;

export default function SimulatorPage() {
  const { state } = useLocation() as { state: any };

  const [form, setForm] = useState({
    prix_achat_fai: state?.prix_achat_fai || 47000,
    mobilier_electromenager: 0,
    frais_notaire_pct: 8.5,
    frais_agence_euro: 0,
    autres_frais_achat: 0,
    surface_m2: state?.surface_m2 || 25,
    travaux_total_euro: state?.travaux_total_euro ?? 5000,
    dpe_avant: state?.dpe_avant || "NC",
    dpe_apres: "NC",
    loyer_hc_mensuel: state?.loyer_hc_mensuel || 450,
    charges_locataire_mensuel: 0,
    gestion_agence_pct: 0,
    frais_relocation_annuel: 0,
    gli_pct: 0,
    assurance_pno_annuel: 200,
    taxe_fonciere_annuel: 431,
    vacance_mois_an: 0.5,
    charges_copro_proprio_annuel: 291,
    autres_charges_annuel: 0,
    frais_comptable_annuel: 200,
    cfe_annuel: 200,
    apport_euro: 20000,
    duree_credit_ans: 10,
    taux_credit_pct: 1.5,
    taux_assurance_pct: 0,
    frais_bancaires_euro: 0,
    frais_cautionnement_euro: 0,
  });

  const [res, setRes] = useState<any>(null);

  useEffect(() => {
    api.simulateInvestment(form).then(setRes).catch(() => {});
  }, [form]);

  const setNum = (k: string, val: string) => setForm((f) => ({ ...f, [k]: Number(val) || 0 }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Simulateur d'investissement locatif & Cash-flow</h1>
        {state?.titre && <p className="text-sm text-sky-400 mt-1">Pré-rempli depuis : {state.titre}</p>}
      </div>

      {/* Synthèse KPIs */}
      {res && (
        <div className="space-y-3">
          {res.alerte_dpe && (
            <div className="p-3 rounded-lg bg-amber-500/20 border border-amber-500/40 text-amber-300 text-sm font-medium">
              {res.alerte_dpe}
            </div>
          )}
          <div className="grid grid-cols-2 sm:grid-cols-6 gap-3">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
              <div className="text-xs text-slate-400">Total projet</div>
              <div className="text-lg font-bold">{fmt(res.total_projet_euro)}</div>
              <div className="text-xs text-slate-500">Emprunt : {fmt(res.emprunt_total_euro)}</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
              <div className="text-xs text-slate-400">Loyers mensuels</div>
              <div className="text-lg font-bold text-emerald-400">{fmt(res.total_loyers_mensuels_euro)}</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
              <div className="text-xs text-slate-400">Charges + Crédit</div>
              <div className="text-lg font-bold text-rose-400">{fmt(res.total_sorties_mensuelles_euro)}</div>
              <div className="text-xs text-slate-500">Crédit : {fmt(res.mensualite_credit_euro)}</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
              <div className="text-xs text-slate-400">Cash-flow mensuel</div>
              <div className={`text-xl font-bold ${res.cash_flow_mensuel_euro >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                {res.cash_flow_mensuel_euro >= 0 ? "+" : ""}{res.cash_flow_mensuel_euro} €
              </div>
              <div className="text-xs text-slate-500">Couverture : {res.taux_couverture_pct ?? "—"}%</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
              <div className="text-xs text-slate-400">Rendement brut</div>
              <div className="text-xl font-bold text-sky-400">{res.rendement_brut_pct}%</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
              <div className="text-xs text-slate-400">Rendement net</div>
              <div className="text-xl font-bold text-emerald-400">{res.rendement_net_pct}%</div>
            </div>
          </div>
        </div>
      )}

      {/* Formulaire 4 blocs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
          <h2 className="font-semibold text-sky-300">1. Achat & Travaux</h2>
          <div className="grid grid-cols-2 gap-3">
            {[
              ["prix_achat_fai", "Prix d'achat FAI (€)"],
              ["frais_notaire_pct", "Frais de notaire (%)"],
              ["mobilier_electromenager", "Mobilier (€)"],
              ["travaux_total_euro", "Montant travaux (€)"],
              ["frais_agence_euro", "Frais d'agence (€)"],
              ["autres_frais_achat", "Autres frais (€)"],
            ].map(([k, lbl]) => (
              <div key={k}>
                <label className="text-xs text-slate-400">{lbl}</label>
                <input type="number" step="any" value={(form as any)[k]} onChange={(e) => setNum(k, e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 mt-1" />
              </div>
            ))}
            <div>
              <label className="text-xs text-slate-400">DPE avant</label>
              <select value={form.dpe_avant} onChange={(e) => setForm({ ...form, dpe_avant: e.target.value })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 mt-1">
                {["NC", "A", "B", "C", "D", "E", "F", "G"].map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-slate-400">DPE après travaux</label>
              <select value={form.dpe_apres} onChange={(e) => setForm({ ...form, dpe_apres: e.target.value })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 mt-1">
                {["NC", "A", "B", "C", "D", "E", "F", "G"].map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
          <h2 className="font-semibold text-emerald-300">2. Loyers & Financement</h2>
          {state?.loyer_bas && (
            <div className="flex gap-2 text-xs">
              <button type="button" onClick={() => setForm({ ...form, loyer_hc_mensuel: state.loyer_bas })} className="px-2 py-1 rounded bg-slate-800">Bas : {state.loyer_bas} €</button>
              <button type="button" onClick={() => setForm({ ...form, loyer_hc_mensuel: state.loyer_hc_mensuel })} className="px-2 py-1 rounded bg-slate-800">Moyen : {state.loyer_hc_mensuel} €</button>
              <button type="button" onClick={() => setForm({ ...form, loyer_hc_mensuel: state.loyer_haut })} className="px-2 py-1 rounded bg-slate-800">Haut : {state.loyer_haut} €</button>
            </div>
          )}
          <div className="grid grid-cols-2 gap-3">
            {[
              ["loyer_hc_mensuel", "Loyer HC (€/mois)"],
              ["charges_locataire_mensuel", "Charges locataire (€/mois)"],
              ["apport_euro", "Apport (€)"],
              ["duree_credit_ans", "Durée crédit (ans)"],
              ["taux_credit_pct", "Taux crédit (%)"],
              ["taux_assurance_pct", "Taux assurance (%)"],
            ].map(([k, lbl]) => (
              <div key={k}>
                <label className="text-xs text-slate-400">{lbl}</label>
                <input type="number" step="any" value={(form as any)[k]} onChange={(e) => setNum(k, e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 mt-1" />
              </div>
            ))}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3 md:col-span-2">
          <h2 className="font-semibold text-amber-300">3. Charges d'exploitation</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              ["taxe_fonciere_annuel", "Taxe foncière (€/an)"],
              ["charges_copro_proprio_annuel", "Copro proprio (€/an)"],
              ["assurance_pno_annuel", "Assurance PNO (€/an)"],
              ["vacance_mois_an", "Vacance (mois/an)"],
              ["gestion_agence_pct", "Gestion agence (%)"],
              ["gli_pct", "GLI (%)"],
              ["frais_comptable_annuel", "Comptable (€/an)"],
              ["cfe_annuel", "CFE (€/an)"],
            ].map(([k, lbl]) => (
              <div key={k}>
                <label className="text-xs text-slate-400">{lbl}</label>
                <input type="number" step="any" value={(form as any)[k]} onChange={(e) => setNum(k, e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 mt-1" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
