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
  const [offset, setOffset] = useState(0);
  const limit = 50;

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
        limit,
        offset,
      })
      .then((res) => {
        setData(res);
        setErr("");
      })
      .catch((e) => setErr(e.message))
      .finally(() => setLoading(false));
  }, [ville, typeBien, tranche, quartier, mode, minConf, sortBy, offset]);

  const handleVilleChange = (v: string) => {
    setVille(v);
    setQuartier("Tous");
    setOffset(0);
  };

  const openDetail = async (id: string) => {
    try {
      const det = await api.getOpportunityDetail(id);
      setSelected(det);
    } catch (e: any) {
      setErr(e.message);
    }
  };

  const fraicheurMap = data?.fraicheur_villes || {};
  const villeKey = Object.keys(fraicheurMap).find(
    (k) => fraicheurMap[k]?.nom?.toLowerCase() === ville.toLowerCase()
  );
  const villeFraicheur = villeKey ? fraicheurMap[villeKey] : null;
  const totalFiltered = data?.total_filtered || 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Radar des opportunités immobilières</h1>
          <p className="text-sm text-slate-500">Cliquez sur un bien pour ouvrir la fiche complète (SHAP %, comparables DVF, simulation).</p>
        </div>
        <div className="flex bg-white border border-slate-200 rounded-lg p-1 text-xs shadow-sm">
          <button
            onClick={() => { setMode("avec_travaux"); setOffset(0); }}
            className={`px-3 py-1.5 rounded-md font-semibold transition ${mode === "avec_travaux" ? "bg-emerald-600 text-white" : "text-slate-600 hover:text-slate-900"}`}
          >
            Avec travaux (Coût total)
          </button>
          <button
            onClick={() => { setMode("brut"); setOffset(0); }}
            className={`px-3 py-1.5 rounded-md font-semibold transition ${mode === "brut" ? "bg-sky-600 text-white" : "text-slate-600 hover:text-slate-900"}`}
          >
            Prix brut (Sans travaux)
          </button>
        </div>
      </div>

      {/* Bandeau fraîcheur / cooldown */}
      {villeFraicheur && (
        <div className={`px-4 py-2.5 rounded-xl border text-xs flex items-center justify-between ${villeFraicheur.en_cooldown ? "bg-amber-50 border-amber-200 text-amber-800" : "bg-sky-50 border-sky-200 text-sky-800"}`}>
          <span>
            <strong>{villeFraicheur.nom}</strong> — Dernière collecte :{" "}
            {villeFraicheur.last_success_at ? new Date(villeFraicheur.last_success_at).toLocaleString("fr-FR") : "initialisation en cours"}
          </span>
          {villeFraicheur.en_cooldown && <span className="font-semibold">Pause anti-blocage active (reprise sous 3h)</span>}
        </div>
      )}

      {/* Filtres */}
      <div className="grid grid-cols-2 sm:grid-cols-6 gap-3 bg-white border border-slate-200 rounded-xl p-4 text-sm shadow-sm">
        <select value={ville} onChange={(e) => handleVilleChange(e.target.value)} className="bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-800">
          <option value="Toutes">Toutes villes</option>
          {(data?.facets?.villes || []).map((v: string) => <option key={v} value={v}>{v}</option>)}
        </select>
        <select value={typeBien} onChange={(e) => { setTypeBien(e.target.value); setOffset(0); }} className="bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-800">
          <option value="Tous">Tous types</option>
          <option value="flat">Appartement</option>
          <option value="house">Maison</option>
        </select>
        <select value={tranche} onChange={(e) => { setTranche(e.target.value); setOffset(0); }} className="bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-800">
          <option value="Toutes">Toutes surfaces</option>
          {(data?.facets?.tranches_m2 || []).map((t: string) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select value={quartier} onChange={(e) => { setQuartier(e.target.value); setOffset(0); }} className="bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-800">
          <option value="Tous">Tous quartiers</option>
          {(data?.facets?.quartiers || []).map((q: string) => <option key={q} value={q}>{q}</option>)}
        </select>
        <select value={sortBy} onChange={(e) => { setSortBy(e.target.value); setOffset(0); }} className="bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-800">
          <option value="auto">Tri : Plus forte décote</option>
          <option value="rendement">Tri : Rendement brut</option>
          <option value="confiance">Tri : Confiance</option>
          <option value="prix_asc">Tri : Prix croissant</option>
        </select>
        <label className="flex items-center gap-2 text-xs text-slate-700 bg-slate-50 border border-slate-300 rounded-lg px-3 py-2">
          <input type="checkbox" checked={minConf >= 45} onChange={(e) => { setMinConf(e.target.checked ? 45 : 0); setOffset(0); }} />
          Masquer incertains (&lt;45)
        </label>
      </div>

      {/* KPIs */}
      {data?.kpis && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <div className="text-xs text-slate-500">Biens filtrés</div>
            <div className="text-2xl font-bold text-slate-900">{data.kpis.total_biens}</div>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <div className="text-xs text-slate-500">Opportunités fortes</div>
            <div className="text-2xl font-bold text-emerald-600">{data.kpis.nb_opportunites_fortes}</div>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <div className="text-xs text-slate-500">Sous-évalués total</div>
            <div className="text-2xl font-bold text-teal-600">{data.kpis.nb_sous_evalues}</div>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <div className="text-xs text-slate-500">Confiance médiane</div>
            <div className="text-2xl font-bold text-sky-600">{data.kpis.confiance_mediane ?? "—"}/100</div>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <div className="text-xs text-slate-500">Rendement brut méd.</div>
            <div className="text-2xl font-bold text-amber-600">{data.kpis.rendement_brut_median_pct ?? "—"}%</div>
          </div>
        </div>
      )}

      {err && <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-sm">{err}</div>}

      {/* Tableau */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 text-slate-500 text-xs uppercase border-b border-slate-200">
              <tr>
                <th className="px-4 py-3">Bien & Quartier</th>
                <th className="px-4 py-3 text-right">Surf.</th>
                <th className="px-4 py-3 text-right">Prix affiché</th>
                <th className="px-4 py-3 text-right">Travaux est.</th>
                <th className="px-4 py-3 text-right">Justeprix IA</th>
                <th className="px-4 py-3 text-right">Écart</th>
                <th className="px-4 py-3 text-right">Confiance</th>
                <th className="px-4 py-3">Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr><td colSpan={8} className="px-4 py-8 text-center text-slate-400">Chargement des offres...</td></tr>
              ) : (data?.items || []).map((it: any) => {
                const ecart = mode === "avec_travaux" ? it.decote_nette_pct : it.ecart_brut_pct;
                return (
                  <tr
                    key={it.id}
                    onClick={() => openDetail(it.id)}
                    className="hover:bg-sky-50/50 cursor-pointer transition"
                  >
                    <td className="px-4 py-3">
                      <div className="font-semibold text-slate-900">{it.type_label || it.type_bien} • {it.ville}</div>
                      <div className="text-xs text-slate-500">{it.quartier || "Quartier NC"} • DPE {it.dpe || "NC"}</div>
                    </td>
                    <td className="px-4 py-3 text-right text-slate-700">{it.surface_m2} m²</td>
                    <td className="px-4 py-3 text-right font-semibold text-slate-900">{fmt(it.prix_affiche)}</td>
                    <td className="px-4 py-3 text-right text-amber-700 font-medium">{fmt(it.cout_travaux_estime)}</td>
                    <td className="px-4 py-3 text-right text-sky-700 font-semibold">{fmt(mode === "avec_travaux" ? it.valeur_estimee_renovee : it.prix_hybride)}</td>
                    <td className={`px-4 py-3 text-right font-bold ${ecart < 0 ? "text-emerald-600" : ecart >= 7 ? "text-rose-600" : "text-slate-600"}`}>
                      {ecart < 0 ? `+${Math.abs(ecart).toFixed(1)}% décote` : ecart > 0 ? `+${ecart}% surcote` : "0%"}
                    </td>
                    <td className="px-4 py-3 text-right text-slate-700">{it.score_confiance}/100</td>
                    <td className="px-4 py-3"><VerdictBadge verdict={it.verdict} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {totalFiltered > limit && (
          <div className="px-4 py-3 border-t border-slate-200 bg-slate-50 flex items-center justify-between text-xs text-slate-600">
            <span>Affichage {offset + 1}–{Math.min(offset + limit, totalFiltered)} sur {totalFiltered} biens</span>
            <div className="flex gap-2">
              <button
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - limit))}
                className="px-3 py-1 rounded border border-slate-300 bg-white disabled:opacity-40"
              >
                Précédent
              </button>
              <button
                disabled={offset + limit >= totalFiltered}
                onClick={() => setOffset(offset + limit)}
                className="px-3 py-1 rounded border border-slate-300 bg-white disabled:opacity-40"
              >
                Suivant
              </button>
            </div>
          </div>
        )}
      </div>

      <PropertyDrawer detail={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
