import { useEffect, useState } from "react";
import { api } from "../api/client";
import PropertyDrawer, { VerdictBadge } from "../components/PropertyDrawer";

const fmt = (v?: number | null) =>
  v === undefined || v === null ? "—" : `${Math.round(v).toLocaleString("fr-FR")} €`;

export default function RadarPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");
  const [selected, setSelected] = useState<any>(null);

  const [ville, setVille] = useState("Toutes");
  const [typeBien, setTypeBien] = useState("Tous");
  const [tranche, setTranche] = useState("Toutes");
  const [quartier, setQuartier] = useState("Tous");
  const [mode, setMode] = useState<"avec_travaux" | "brut">("avec_travaux");
  const [minConf, setMinConf] = useState(45);
  const [sortBy, setSortBy] = useState("auto");

  useEffect(() => {
    setLoading(true);
    api
      .getOpportunities({
        ville,
        type_bien: typeBien,
        tranche_m2: tranche,
        quartier,
        mode,
        min_confiance: minConf,
        sort_by: sortBy,
        limit: 100,
      })
      .then((res) => {
        setData(res);
        setErr("");
      })
      .catch((e) => setErr(e.message))
      .finally(() => setLoading(false));
  }, [ville, typeBien, tranche, quartier, mode, minConf, sortBy]);

  const openDetail = async (id: string) => {
    try {
      const det = await api.getOpportunityDetail(id);
      setSelected(det);
    } catch (e: any) {
      setErr(e.message);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Radar des plus fortes décotes</h1>
          <p className="text-sm text-slate-400">Clique sur une annonce pour voir les photos, la description, SHAP et les comparables.</p>
        </div>
        <div className="flex bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
          <button
            onClick={() => setMode("avec_travaux")}
            className={`px-3 py-1.5 rounded-md font-semibold ${mode === "avec_travaux" ? "bg-emerald-600 text-white" : "text-slate-400"}`}
          >
            Avec travaux (Coût total)
          </button>
          <button
            onClick={() => setMode("brut")}
            className={`px-3 py-1.5 rounded-md font-semibold ${mode === "brut" ? "bg-sky-600 text-white" : "text-slate-400"}`}
          >
            Prix brut (Sans travaux)
          </button>
        </div>
      </div>

      {/* Filtres */}
      <div className="grid grid-cols-2 sm:grid-cols-6 gap-3 bg-slate-900 border border-slate-800 rounded-xl p-4 text-sm">
        <select value={ville} onChange={(e) => setVille(e.target.value)} className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2">
          <option value="Toutes">Toutes villes</option>
          {(data?.facets?.villes || []).map((v: string) => <option key={v} value={v}>{v}</option>)}
        </select>
        <select value={typeBien} onChange={(e) => setTypeBien(e.target.value)} className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2">
          <option value="Tous">Tous types</option>
          <option value="flat">Appartement</option>
          <option value="house">Maison</option>
        </select>
        <select value={tranche} onChange={(e) => setTranche(e.target.value)} className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2">
          <option value="Toutes">Toutes surfaces</option>
          {(data?.facets?.tranches_m2 || []).map((t: string) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select value={quartier} onChange={(e) => setQuartier(e.target.value)} className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2">
          <option value="Tous">Tous quartiers</option>
          {(data?.facets?.quartiers || []).map((q: string) => <option key={q} value={q}>{q}</option>)}
        </select>
        <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2">
          <option value="auto">Tri : Plus forte décote</option>
          <option value="rendement">Tri : Rendement brut</option>
          <option value="confiance">Tri : Confiance</option>
          <option value="prix_asc">Tri : Prix croissant</option>
        </select>
        <label className="flex items-center gap-2 text-xs text-slate-300 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2">
          <input type="checkbox" checked={minConf >= 45} onChange={(e) => setMinConf(e.target.checked ? 45 : 0)} />
          Masquer incertains (&lt;45)
        </label>
      </div>

      {/* KPIs */}
      {data?.kpis && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="text-xs text-slate-400">Biens filtrés</div>
            <div className="text-2xl font-bold">{data.kpis.total_biens}</div>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="text-xs text-slate-400">Opportunités fortes</div>
            <div className="text-2xl font-bold text-emerald-400">{data.kpis.nb_opportunites_fortes}</div>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="text-xs text-slate-400">Sous-évalués total</div>
            <div className="text-2xl font-bold text-teal-400">{data.kpis.nb_sous_evalues}</div>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="text-xs text-slate-400">Confiance médiane</div>
            <div className="text-2xl font-bold text-sky-400">{data.kpis.confiance_mediane ?? "—"}/100</div>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="text-xs text-slate-400">Rendement brut méd.</div>
            <div className="text-2xl font-bold text-amber-400">{data.kpis.rendement_brut_median_pct ?? "—"}%</div>
          </div>
        </div>
      )}

      {err && <div className="p-3 rounded-lg bg-rose-500/20 border border-rose-500/40 text-rose-300 text-sm">{err}</div>}

      {/* Tableau */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-950 text-slate-400 text-xs border-b border-slate-800">
              <tr>
                <th className="px-4 py-3">Bien & Quartier</th>
                <th className="px-4 py-3 text-right">Surf.</th>
                <th className="px-4 py-3 text-right">Prix affiché</th>
                <th className="px-4 py-3 text-right">Travaux est.</th>
                <th className="px-4 py-3 text-right">Estimation IA</th>
                <th className="px-4 py-3 text-right">Écart</th>
                <th className="px-4 py-3 text-right">Confiance</th>
                <th className="px-4 py-3">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={8} className="px-4 py-8 text-center text-slate-400">Chargement des offres...</td></tr>
              ) : (data?.items || []).map((it: any) => {
                const ecart = mode === "avec_travaux" ? it.decote_nette_pct : it.ecart_brut_pct;
                return (
                  <tr
                    key={it.id}
                    onClick={() => openDetail(it.id)}
                    className="border-b border-slate-800/60 hover:bg-slate-800/50 cursor-pointer transition"
                  >
                    <td className="px-4 py-3">
                      <div className="font-semibold text-slate-100">{it.type_label || it.type_bien} • {it.ville}</div>
                      <div className="text-xs text-slate-400">{it.quartier || "Quartier NC"} • DPE {it.dpe || "NC"}</div>
                    </td>
                    <td className="px-4 py-3 text-right">{it.surface_m2} m²</td>
                    <td className="px-4 py-3 text-right font-semibold">{fmt(it.prix_affiche)}</td>
                    <td className="px-4 py-3 text-right text-amber-400">{fmt(it.cout_travaux_estime)}</td>
                    <td className="px-4 py-3 text-right text-sky-300 font-semibold">{fmt(mode === "avec_travaux" ? it.valeur_estimee_renovee : it.prix_hybride)}</td>
                    <td className={`px-4 py-3 text-right font-bold ${ecart <= -10 ? "text-emerald-400" : ecart >= 7 ? "text-rose-400" : "text-slate-300"}`}>
                      {ecart > 0 ? `+${ecart}` : ecart}%
                    </td>
                    <td className="px-4 py-3 text-right">{it.score_confiance}/100</td>
                    <td className="px-4 py-3"><VerdictBadge verdict={it.verdict} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <PropertyDrawer detail={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
