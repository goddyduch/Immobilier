import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { Bar, BarChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";

export default function DvfPage() {
  const { state } = useLocation() as { state: any };
  const [dvf, setDvf] = useState<any>(null);
  const [ville, setVille] = useState("Toutes");
  const [typeBien, setTypeBien] = useState("Tous");

  const [negoForm, setNegoForm] = useState({
    prix_affiche: state?.prix_affiche || 180000,
    surface_m2: state?.surface_m2 || 70,
    dpe: state?.dpe || "D",
    travaux_euro: state?.travaux_euro || 0,
    prix_notaire_ref_m2: 2220,
  });
  const [negoRes, setNegoRes] = useState<any>(null);

  useEffect(() => {
    api.getDvfAnalysis({ ville, type_bien: typeBien }).then((res) => {
      setDvf(res);
      if (res?.pm2_notaire_median > 0) {
        setNegoForm((prev) => ({ ...prev, prix_notaire_ref_m2: Math.round(res.pm2_notaire_median) }));
      }
    }).catch(() => {});
  }, [ville, typeBien]);

  useEffect(() => {
    api.simulateNegotiation(negoForm).then(setNegoRes).catch(() => {});
  }, [negoForm]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Analyse Notaires DVF & Simulateur de Négociation</h1>
          <p className="text-sm text-slate-500">Prix affiché (FAI) vs prix signé net vendeur (DVF) par ville, taille et DPE.</p>
        </div>
        <div className="flex gap-2 text-sm">
          <select value={ville} onChange={(e) => setVille(e.target.value)} className="bg-white border border-slate-300 rounded-lg px-3 py-2 text-slate-800 shadow-sm">
            <option value="Toutes">Toutes villes</option>
            {(dvf?.villes_disponibles || []).map((v: string) => <option key={v} value={v}>{v}</option>)}
          </select>
          <select value={typeBien} onChange={(e) => setTypeBien(e.target.value)} className="bg-white border border-slate-300 rounded-lg px-3 py-2 text-slate-800 shadow-sm">
            <option value="Tous">Tous types</option>
            <option value="flat">Appartement</option>
            <option value="house">Maison</option>
          </select>
        </div>
      </div>

      {dvf && dvf.nb_offres === 0 && (
        <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-sm">
          Pas encore assez d'offres actives indexées pour ce filtre ({ville} • {typeBien}). Le référentiel s'enrichira automatiquement au prochain cycle de collecte.
        </div>
      )}

      {dvf && dvf.nb_offres > 0 && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <div className="text-xs text-slate-500">Affiché médian</div>
              <div className="text-2xl font-bold text-slate-900">{dvf.pm2_affiche_median} €/m²</div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <div className="text-xs text-slate-500">Notaire DVF médian</div>
              <div className="text-2xl font-bold text-amber-600">{dvf.pm2_notaire_median} €/m²</div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <div className="text-xs text-slate-500">Écart Affiché vs DVF</div>
              <div className="text-2xl font-bold text-sky-700">{dvf.ecart_global_pct > 0 ? `+${dvf.ecart_global_pct}` : dvf.ecart_global_pct}%</div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <div className="text-xs text-slate-500">Marge négo médiane</div>
              <div className="text-2xl font-bold text-emerald-600">-{dvf.taux_nego_median_pct}%</div>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
            <h2 className="font-semibold text-sm text-slate-900 mb-3">Prix/m² Affiché vs Notaire DVF par tranche de surface</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dvf.par_tranche}>
                  <XAxis dataKey="tranche_m2" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="pm2_affiche_median" name="Affiché €/m²" fill="#0284c7" />
                  <Bar dataKey="pm2_notaire_median" name="Notaire DVF €/m²" fill="#d97706" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* Simulateur négociation */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4 shadow-sm">
        <h2 className="font-semibold text-base text-slate-900">Simulateur d'offre de négociation</h2>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-sm">
          <div>
            <label className="text-xs text-slate-500">Prix affiché (€)</label>
            <input type="number" value={negoForm.prix_affiche} onChange={(e) => setNegoForm({ ...negoForm, prix_affiche: Number(e.target.value) })} className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 mt-1 text-slate-900" />
          </div>
          <div>
            <label className="text-xs text-slate-400">Surface (m²)</label>
            <input type="number" value={negoForm.surface_m2} onChange={(e) => setNegoForm({ ...negoForm, surface_m2: Number(e.target.value) })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1" />
          </div>
          <div>
            <label className="text-xs text-slate-400">DPE</label>
            <select value={negoForm.dpe} onChange={(e) => setNegoForm({ ...negoForm, dpe: e.target.value })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1">
              {["A", "B", "C", "D", "E", "F", "G"].map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-400">Travaux estimés (€)</label>
            <input type="number" value={negoForm.travaux_euro} onChange={(e) => setNegoForm({ ...negoForm, travaux_euro: Number(e.target.value) })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1" />
          </div>
          <div>
            <label className="text-xs text-slate-400">Réf. Notaire (€/m²)</label>
            <input type="number" value={negoForm.prix_notaire_ref_m2} onChange={(e) => setNegoForm({ ...negoForm, prix_notaire_ref_m2: Number(e.target.value) })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1" />
          </div>
        </div>

        {negoRes && (
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 pt-2">
            <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
              <div className="text-xs text-slate-400">Offre prudente</div>
              <div className="text-lg font-bold text-sky-300">{negoRes.offre_prudente.toLocaleString("fr-FR")} €</div>
            </div>
            <div className="bg-slate-950 border border-emerald-500/40 rounded-lg p-3">
              <div className="text-xs text-slate-400">Offre conseillée (-{negoRes.marge_conseillee_pct}%)</div>
              <div className="text-lg font-bold text-emerald-400">{negoRes.offre_conseillee.toLocaleString("fr-FR")} €</div>
              <div className="text-xs text-emerald-300">Économie : {negoRes.economie_estimee_euro.toLocaleString("fr-FR")} €</div>
            </div>
            <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
              <div className="text-xs text-slate-400">Offre offensive</div>
              <div className="text-lg font-bold text-amber-400">{negoRes.offre_offensive.toLocaleString("fr-FR")} €</div>
            </div>
            <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-300 space-y-1">
              <div className="font-semibold text-slate-200">Arguments clés :</div>
              {negoRes.arguments.map((a: string, i: number) => <div key={i}>• {a}</div>)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
