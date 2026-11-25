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
    api.getDvfAnalysis({ ville, type_bien: typeBien }).then(setDvf).catch(() => {});
  }, [ville, typeBien]);

  useEffect(() => {
    api.simulateNegotiation(negoForm).then(setNegoRes).catch(() => {});
  }, [negoForm]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Analyse Notaires DVF & Simulateur de Négociation</h1>
          <p className="text-sm text-slate-400">Prix affiché (FAI) vs prix signé net vendeur (DVF) par taille de logement et DPE.</p>
        </div>
        <div className="flex gap-2 text-sm">
          <select value={typeBien} onChange={(e) => setTypeBien(e.target.value)} className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2">
            <option value="Tous">Tous types</option>
            <option value="flat">Appartement</option>
            <option value="house">Maison</option>
          </select>
        </div>
      </div>

      {dvf && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400">Affiché médian</div>
              <div className="text-2xl font-bold">{dvf.pm2_affiche_median} €/m²</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400">Notaire DVF médian</div>
              <div className="text-2xl font-bold text-amber-400">{dvf.pm2_notaire_median} €/m²</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400">Écart Affiché vs DVF</div>
              <div className="text-2xl font-bold text-sky-400">{dvf.ecart_global_pct > 0 ? `+${dvf.ecart_global_pct}` : dvf.ecart_global_pct}%</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400">Marge négo médiane</div>
              <div className="text-2xl font-bold text-emerald-400">-{dvf.taux_nego_median_pct}%</div>
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h2 className="font-semibold text-sm mb-3">Prix/m² Affiché vs Notaire DVF par tranche de surface</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dvf.par_tranche}>
                  <XAxis dataKey="tranche_m2" stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="pm2_affiche_median" name="Affiché €/m²" fill="#38bdf8" />
                  <Bar dataKey="pm2_notaire_median" name="Notaire DVF €/m²" fill="#f59e0b" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* Simulateur négociation */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <h2 className="font-semibold text-base">Simulateur d'offre de négociation</h2>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-sm">
          <div>
            <label className="text-xs text-slate-400">Prix affiché (€)</label>
            <input type="number" value={negoForm.prix_affiche} onChange={(e) => setNegoForm({ ...negoForm, prix_affiche: Number(e.target.value) })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1" />
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
