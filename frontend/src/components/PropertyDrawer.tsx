import { useNavigate } from "react-router-dom";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Calculator, ExternalLink, MapPin, Scale, X } from "lucide-react";

const fmt = (v?: number | null) =>
  v === undefined || v === null ? "—" : `${Math.round(v).toLocaleString("fr-FR")} €`;

export function VerdictBadge({ verdict }: { verdict: string }) {
  const styles: Record<string, string> = {
    "Opportunité forte": "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
    "Sous-évalué": "bg-teal-500/20 text-teal-300 border-teal-500/40",
    "Aligné": "bg-sky-500/20 text-sky-300 border-sky-500/40",
    "Surévalué": "bg-rose-500/20 text-rose-300 border-rose-500/40",
    "Trop incertain": "bg-amber-500/20 text-amber-300 border-amber-500/40",
  };
  return (
    <span className={`text-xs px-2.5 py-0.5 rounded-full border font-semibold ${styles[verdict] || "bg-slate-800 text-slate-300 border-slate-700"}`}>
      {verdict}
    </span>
  );
}

export default function PropertyDrawer({ detail, onClose }: { detail: any; onClose: () => void }) {
  const navigate = useNavigate();
  if (!detail) return null;

  const shapData = (detail.shap_facteurs || []).map((f: any) => ({
    name: f.label,
    impact: Math.round(f.impact_euro),
  }));

  const goSimulateur = () => {
    navigate("/simulateur", {
      state: {
        prix_achat_fai: detail.prix_cible_nego || detail.prix_affiche,
        surface_m2: detail.surface_m2,
        travaux_total_euro: detail.cout_travaux_estime || 0,
        loyer_hc_mensuel: detail.loyer_mensuel_estime || 550,
        loyer_bas: detail.loyer_bas,
        loyer_haut: detail.loyer_haut,
        dpe_avant: detail.dpe || "NC",
        titre: detail.titre,
      },
    });
  };

  const goNego = () => {
    navigate("/marche-dvf", {
      state: {
        prix_affiche: detail.prix_affiche,
        surface_m2: detail.surface_m2,
        dpe: detail.dpe || "D",
        travaux_euro: detail.cout_travaux_estime || 0,
      },
    });
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 flex justify-end" onClick={onClose}>
      <div
        className="w-full max-w-3xl bg-slate-900 border-l border-slate-800 h-full overflow-y-auto p-6 space-y-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-800 pb-4">
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <VerdictBadge verdict={detail.verdict} />
              <span className="text-xs text-slate-400 flex items-center gap-1">
                <MapPin className="w-3.5 h-3.5 text-sky-400" />
                {detail.ville} {detail.quartier ? `• ${detail.quartier}` : ""}
              </span>
            </div>
            <h2 className="text-xl font-bold mt-2">{detail.titre || `${detail.type_label || "Bien"} ${detail.surface_m2} m²`}</h2>
            <p className="text-sm text-slate-400 mt-1">
              {detail.surface_m2} m² • {detail.pieces || "?"} pièces • DPE {detail.dpe || "NC"}
              {detail.etage !== null && detail.etage !== undefined ? ` • Étage ${detail.etage}` : ""}
              {detail.annee_construction ? ` • Année ${detail.annee_construction}` : ""}
            </p>
          </div>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-slate-800 text-slate-400">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Galerie photos */}
        {detail.photos_urls && detail.photos_urls.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {detail.photos_urls.slice(0, 6).map((url: string, idx: number) => (
              <img
                key={idx}
                src={url}
                alt={`Photo ${idx + 1}`}
                className="w-full h-36 object-cover rounded-lg border border-slate-800 bg-slate-950"
                onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
              />
            ))}
          </div>
        )}

        {/* KPIs estimation */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
            <div className="text-xs text-slate-400">Prix affiché (FAI)</div>
            <div className="text-lg font-bold text-slate-100">{fmt(detail.prix_affiche)}</div>
          </div>
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
            <div className="text-xs text-slate-400">Estimation Hybride</div>
            <div className="text-lg font-bold text-emerald-400">{fmt(detail.prix_hybride)}</div>
            <div className="text-xs text-slate-400">{detail.ecart_brut_pct > 0 ? `+${detail.ecart_brut_pct}` : detail.ecart_brut_pct}%</div>
          </div>
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
            <div className="text-xs text-slate-400">Fourchette [Q10–Q90]</div>
            <div className="text-sm font-semibold text-sky-300 mt-0.5">{fmt(detail.prix_q10)} – {fmt(detail.prix_q90)}</div>
            <div className="text-xs text-slate-500">Largeur {detail.intervalle_largeur_pct}%</div>
          </div>
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
            <div className="text-xs text-slate-400">Confiance & Travaux</div>
            <div className="text-lg font-bold text-sky-400">{detail.score_confiance}/100</div>
            <div className="text-xs text-amber-400">Travaux : {fmt(detail.cout_travaux_estime)}</div>
          </div>
        </div>

        {/* Boutons d'action */}
        <div className="flex flex-wrap gap-3">
          <button onClick={goSimulateur} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-sm font-semibold">
            <Calculator className="w-4 h-4" /> Simuler l'investissement locatif
          </button>
          <button onClick={goNego} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-sm font-semibold">
            <Scale className="w-4 h-4" /> Préparer la négociation
          </button>
          {detail.url && (
            <a href={detail.url} target="_blank" rel="noreferrer" className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm">
              <ExternalLink className="w-4 h-4" /> Voir l'annonce
            </a>
          )}
        </div>

        {/* Graphique SHAP */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
          <h3 className="font-semibold text-sm mb-3">Explication du prix (SHAP — impacts en €)</h3>
          {shapData.length > 0 ? (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={shapData} layout="vertical" margin={{ left: 20, right: 30 }}>
                  <XAxis type="number" stroke="#94a3b8" fontSize={11} />
                  <YAxis type="category" dataKey="name" width={180} stroke="#cbd5e1" fontSize={11} />
                  <Tooltip formatter={(val: any) => `${Number(val) > 0 ? "+" : ""}${Number(val).toLocaleString("fr-FR")} €`} />
                  <Bar dataKey="impact" radius={[4, 4, 4, 4]}>
                    {shapData.map((entry: any, i: number) => (
                      <Cell key={i} fill={entry.impact >= 0 ? "#10b981" : "#f43f5e"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-xs text-slate-400">Facteurs SHAP non disponibles pour cette annonce.</p>
          )}
        </div>

        {/* Description intégrale */}
        {detail.description && (
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
            <h3 className="font-semibold text-sm mb-2">Description de l'annonce</h3>
            <p className="text-sm text-slate-300 whitespace-pre-line leading-relaxed">{detail.description}</p>
          </div>
        )}

        {/* Comparables */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
          <h3 className="font-semibold text-sm mb-3">Comparables voisins (DVF & Marché)</h3>
          {detail.comparables && detail.comparables.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="py-2">Source</th>
                    <th className="py-2">Bien</th>
                    <th className="py-2 text-right">Dist.</th>
                    <th className="py-2 text-right">Surf.</th>
                    <th className="py-2 text-right">Prix</th>
                    <th className="py-2 text-right">Sim.</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.comparables.map((c: any, idx: number) => (
                    <tr key={idx} className="border-b border-slate-800/60">
                      <td className="py-2 text-sky-400">{c.source}</td>
                      <td className="py-2">{c.titre}</td>
                      <td className="py-2 text-right">{c.distance_m} m</td>
                      <td className="py-2 text-right">{c.surface_m2} m²</td>
                      <td className="py-2 text-right">{fmt(c.prix)}</td>
                      <td className="py-2 text-right">{c.similarite_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-xs text-slate-400">Aucun comparable voisin dans le rayon retenu.</p>
          )}
        </div>
      </div>
    </div>
  );
}
