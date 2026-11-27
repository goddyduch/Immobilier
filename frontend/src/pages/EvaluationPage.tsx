import { useEffect, useState } from "react";
import { ChevronDown, ChevronUp, Info } from "lucide-react";
import { api } from "../api/client";

const fmtEuro = (v?: number | null) =>
  v === undefined || v === null ? "—" : `${Math.round(v).toLocaleString("fr-FR")} €`;

const fmtPct = (v?: number | null) =>
  v === undefined || v === null ? "—" : `${v.toFixed(1)} %`;

const DEFINITIONS_METRIQUES: Record<string, string> = {
  R2: "Part de la variation des prix expliquée par le modèle (plus proche de 1,00 = meilleur ajustement).",
  Chi2: "Test d'adéquation comparant la distribution réelle des prix à celle prédite par tranches.",
  RMSE: "Racine de l'erreur quadratique moyenne (€) : pénalise fortement les grosses erreurs.",
  MAE: "Erreur absolue moyenne en euros sur l'ensemble des biens.",
  MAPE: "Erreur absolue moyenne en pourcentage (%).",
  MdAPE: "Médiane des erreurs absolues en pourcentage (%) : moins sensible aux valeurs extrêmes.",
  Within20: "Part des estimations situées dans une marge d'erreur de ±20 % autour du prix réel.",
};

function MetricHelp({ label, helpKey }: { label: string; helpKey: string }) {
  return (
    <span className="inline-flex items-center gap-1" title={DEFINITIONS_METRIQUES[helpKey] || ""}>
      <span>{label}</span>
      <Info className="w-3.5 h-3.5 text-slate-400 cursor-help" />
    </span>
  );
}

export default function EvaluationPage() {
  const [data, setData] = useState<any>(null);
  const [villeCode, setVilleCode] = useState<string>("global");
  const [showTech, setShowTech] = useState<boolean>(false);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    api.getEvaluation().then(setData).catch((e) => setErr(e.message));
  }, []);

  if (err) return <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700">{err}</div>;
  if (!data) return <div className="text-slate-500 py-8">Calcul / chargement de l'évaluation multi-villes...</div>;

  const parVille: any[] = data.par_ville || [];
  const selectedMetrics =
    villeCode === "global"
      ? data.global_metrics || {}
      : parVille.find((v) => v.code_territoire === villeCode) || data.global_metrics || {};

  const duel = data.comparatif_modeles || {};
  const mDirect = duel.hybride_log_prix || {};
  const mRes = duel.hybride_residus || {};

  return (
    <div className="space-y-6">
      {/* En-tête + sélecteur Global / Ville */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Évaluation du Modèle Justeprix</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Modèle : <span className="font-mono text-slate-700">{data.model_version}</span> • Approche retenue :{" "}
            <span className="font-semibold text-sky-700">
              {data.approche_retenue === "hybride_residus" ? "Hybride Résidus + AVM" : "Hybride log(prix) + AVM"}
            </span>
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium text-slate-600">Périmètre :</label>
          <select
            value={villeCode}
            onChange={(e) => setVilleCode(e.target.value)}
            className="bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm font-medium text-slate-800"
          >
            <option value="global">Global (12 villes-centres)</option>
            {parVille.map((v) => (
              <option key={v.code_territoire} value={v.code_territoire}>
                {v.nom} ({v.nb_biens || 0} biens — {v.maturite})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Niveau 1 : Vue Métier claire */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <div className="text-xs font-medium text-slate-500">
            <MetricHelp label="Dans ±20 %" helpKey="Within20" />
          </div>
          <div className="text-2xl font-extrabold text-emerald-600 mt-1">{fmtPct(selectedMetrics.within_20pct)}</div>
          <div className="text-[11px] text-slate-400 mt-1">Seuil cible : ≥ 80 %</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <div className="text-xs font-medium text-slate-500">
            <MetricHelp label="MdAPE (Erreur médiane %)" helpKey="MdAPE" />
          </div>
          <div className="text-2xl font-extrabold text-sky-700 mt-1">{fmtPct(selectedMetrics.mdape)}</div>
          <div className="text-[11px] text-slate-400 mt-1">Médiane € : {fmtEuro(selectedMetrics.med_ae)}</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <div className="text-xs font-medium text-slate-500">
            <MetricHelp label="MAPE (Erreur moyenne %)" helpKey="MAPE" />
          </div>
          <div className="text-2xl font-bold text-slate-900 mt-1">{fmtPct(selectedMetrics.mape)}</div>
          <div className="text-[11px] text-slate-400 mt-1">Moyenne € : {fmtEuro(selectedMetrics.mae)}</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <div className="text-xs font-medium text-slate-500">
            <MetricHelp label="R² (Ajustement)" helpKey="R2" />
          </div>
          <div className="text-2xl font-bold text-slate-900 mt-1">{selectedMetrics.r2 ?? "—"}</div>
          <div className="text-[11px] text-slate-400 mt-1">Échantillon : {selectedMetrics.nb_biens ?? 0} biens</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <div className="text-xs font-medium text-slate-500">Fiabilité fourchette (Q10–Q90)</div>
          <div className="text-2xl font-bold text-indigo-600 mt-1">{fmtPct(selectedMetrics.couverture_q10_q90_pct)}</div>
          <div className="text-[11px] text-slate-400 mt-1">Cible théorique : ~80 %</div>
        </div>
      </div>

      {/* Duel Hybride log(prix) vs Hybride Résidus */}
      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
        <h2 className="text-base font-bold text-slate-900 mb-3">
          Comparatif Hybride : ML sur log(prix) + AVM vs ML sur Résidus + AVM
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase text-slate-500 border-b border-slate-200">
              <tr>
                <th className="py-2 pr-4">Modèle hybride</th>
                <th className="py-2 px-3 text-right">R²</th>
                <th className="py-2 px-3 text-right">RMSE (€)</th>
                <th className="py-2 px-3 text-right">MAE (€)</th>
                <th className="py-2 px-3 text-right">MAPE (%)</th>
                <th className="py-2 px-3 text-right">MdAPE (%)</th>
                <th className="py-2 pl-3 text-right">Dans ±20 %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              <tr className={duel.gagnant === "hybride_log_prix" ? "bg-emerald-50/60 font-semibold" : ""}>
                <td className="py-2.5 pr-4">Hybride 1 : ML sur log(prix) + AVM</td>
                <td className="py-2.5 px-3 text-right">{mDirect.r2 ?? "—"}</td>
                <td className="py-2.5 px-3 text-right">{fmtEuro(mDirect.rmse)}</td>
                <td className="py-2.5 px-3 text-right">{fmtEuro(mDirect.mae)}</td>
                <td className="py-2.5 px-3 text-right">{fmtPct(mDirect.mape)}</td>
                <td className="py-2.5 px-3 text-right">{fmtPct(mDirect.mdape)}</td>
                <td className="py-2.5 pl-3 text-right text-emerald-700">{fmtPct(mDirect.within_20pct)}</td>
              </tr>
              <tr className={duel.gagnant === "hybride_residus" ? "bg-emerald-50/60 font-semibold" : ""}>
                <td className="py-2.5 pr-4">Hybride 2 : ML sur résidus + AVM</td>
                <td className="py-2.5 px-3 text-right">{mRes.r2 ?? "—"}</td>
                <td className="py-2.5 px-3 text-right">{fmtEuro(mRes.rmse)}</td>
                <td className="py-2.5 px-3 text-right">{fmtEuro(mRes.mae)}</td>
                <td className="py-2.5 px-3 text-right">{fmtPct(mRes.mape)}</td>
                <td className="py-2.5 px-3 text-right">{fmtPct(mRes.mdape)}</td>
                <td className="py-2.5 pl-3 text-right text-emerald-700">{fmtPct(mRes.within_20pct)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Tableau par catégorie de taille (< 20 m² à >= 130 m²) */}
      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
        <h2 className="text-base font-bold text-slate-900 mb-3">Performance par catégorie de taille (Global)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase text-slate-500 border-b border-slate-200">
              <tr>
                <th className="py-2 pr-4">Tranche de surface</th>
                <th className="py-2 px-3 text-right">Biens</th>
                <th className="py-2 px-3 text-right">R²</th>
                <th className="py-2 px-3 text-right">MAE (€)</th>
                <th className="py-2 px-3 text-right">RMSE (€)</th>
                <th className="py-2 px-3 text-right">MAPE (%)</th>
                <th className="py-2 px-3 text-right">MdAPE (%)</th>
                <th className="py-2 pl-3 text-right">Dans ±20 %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {(data.par_categorie_taille || []).map((row: any) => (
                <tr key={row.categorie} className="hover:bg-slate-50">
                  <td className="py-2.5 pr-4 font-semibold text-slate-800">{row.categorie}</td>
                  <td className="py-2.5 px-3 text-right text-slate-600">{row.nb_biens ?? 0}</td>
                  <td className="py-2.5 px-3 text-right">{row.r2 ?? "—"}</td>
                  <td className="py-2.5 px-3 text-right">{fmtEuro(row.mae)}</td>
                  <td className="py-2.5 px-3 text-right">{fmtEuro(row.rmse)}</td>
                  <td className="py-2.5 px-3 text-right">{fmtPct(row.mape)}</td>
                  <td className="py-2.5 px-3 text-right font-semibold text-sky-700">{fmtPct(row.mdape)}</td>
                  <td className="py-2.5 pl-3 text-right font-semibold text-emerald-700">{fmtPct(row.within_20pct)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Tableau par ville (12 villes-centres) */}
      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
        <h2 className="text-base font-bold text-slate-900 mb-3">Détail par ville-centre (12 villes)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase text-slate-500 border-b border-slate-200">
              <tr>
                <th className="py-2 pr-4">Ville-centre</th>
                <th className="py-2 px-3">Maturité</th>
                <th className="py-2 px-3 text-right">Biens</th>
                <th className="py-2 px-3 text-right">R²</th>
                <th className="py-2 px-3 text-right">MAE (€)</th>
                <th className="py-2 px-3 text-right">MAPE (%)</th>
                <th className="py-2 px-3 text-right">MdAPE (%)</th>
                <th className="py-2 pl-3 text-right">Dans ±20 %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {parVille.map((v: any) => (
                <tr key={v.code_territoire} className="hover:bg-slate-50">
                  <td className="py-2.5 pr-4 font-semibold text-slate-800">{v.nom}</td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
                        v.maturite === "Fiable"
                          ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                          : "bg-amber-50 text-amber-700 border border-amber-200"
                      }`}
                    >
                      {v.maturite}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right">{v.nb_biens ?? 0}</td>
                  <td className="py-2.5 px-3 text-right">{v.r2 ?? "—"}</td>
                  <td className="py-2.5 px-3 text-right">{fmtEuro(v.mae)}</td>
                  <td className="py-2.5 px-3 text-right">{fmtPct(v.mape)}</td>
                  <td className="py-2.5 px-3 text-right font-semibold text-sky-700">{fmtPct(v.mdape)}</td>
                  <td className="py-2.5 pl-3 text-right font-semibold text-emerald-700">{fmtPct(v.within_20pct)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Niveau 2 : Accordéon Détails Techniques Data Science */}
      <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
        <button
          onClick={() => setShowTech((s) => !s)}
          className="w-full px-5 py-4 flex items-center justify-between text-left hover:bg-slate-50 transition"
        >
          <span className="font-semibold text-slate-800">
            Détails techniques Data Science (χ², RMSE, Gain vs Naïf, Avant/Après)
          </span>
          {showTech ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
        </button>
        {showTech && (
          <div className="px-5 pb-5 pt-2 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="text-xs text-slate-500">
                <MetricHelp label="Test χ² (Adéquation)" helpKey="Chi2" />
              </div>
              <div className="text-lg font-bold text-slate-900 mt-1">
                {selectedMetrics.chi2_stat ?? "—"} (p={selectedMetrics.chi2_pvalue ?? "—"})
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="text-xs text-slate-500">
                <MetricHelp label="RMSE (€)" helpKey="RMSE" />
              </div>
              <div className="text-lg font-bold text-slate-900 mt-1">{fmtEuro(selectedMetrics.rmse)}</div>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="text-xs text-slate-500">Gain vs prix/m² naïf</div>
              <div className="text-lg font-bold text-emerald-700 mt-1">{fmtPct(selectedMetrics.gain_vs_naif_pct)}</div>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="text-xs text-slate-500">Précision ≤ 10 %</div>
              <div className="text-lg font-bold text-slate-900 mt-1">{fmtPct(selectedMetrics.within_10pct)}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
