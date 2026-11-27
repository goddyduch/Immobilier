import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Calculator, ExternalLink, Info, MapPin, Scale, TrendingDown, TrendingUp, X } from "lucide-react";

const fmt = (v?: number | null) =>
  v === undefined || v === null ? "—" : `${Math.round(v).toLocaleString("fr-FR")} €`;

export function VerdictBadge({ verdict }: { verdict: string }) {
  const styles: Record<string, string> = {
    "Opportunité forte": "bg-emerald-50 text-emerald-700 border-emerald-200",
    "Sous-évalué": "bg-teal-50 text-teal-700 border-teal-200",
    "Aligné": "bg-sky-50 text-sky-700 border-sky-200",
    "Surévalué": "bg-rose-50 text-rose-700 border-rose-200",
    "Trop incertain": "bg-amber-50 text-amber-700 border-amber-200",
  };
  return (
    <span className={`text-xs px-2.5 py-0.5 rounded-full border font-semibold ${styles[verdict] || "bg-slate-100 text-slate-700 border-slate-200"}`}>
      {verdict}
    </span>
  );
}

export default function PropertyDrawer({ detail, onClose }: { detail: any; onClose: () => void }) {
  const navigate = useNavigate();
  const [showAllShap, setShowAllShap] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!detail) return null;

  const basePrix = detail.prix_estime_ml || detail.prix_hybride || detail.prix_affiche || 1;
  const allShap = (detail.shap_facteurs || []).map((f: any) => {
    const impEuro = Math.round(f.impact_euro ?? 0);
    const impPct =
      f.impact_pct !== undefined && f.impact_pct !== null && (f.impact_pct !== 0 || impEuro === 0)
        ? Number(f.impact_pct)
        : Number(((impEuro / basePrix) * 100).toFixed(1));
    return {
      name: f.label,
      impact_pct: impPct,
      impact_euro: impEuro,
      description: f.description || "",
    };
  });
  const shapVisibles = showAllShap ? allShap : allShap.slice(0, 8);
  const shapPos = shapVisibles.filter((f: any) => f.impact_pct >= 0);
  const shapNeg = shapVisibles.filter((f: any) => f.impact_pct < 0);

  // Calcul décote en positif (ex: +18.5 % de décote si prix affiché < estimation)
  const ecartBrutPct = Number(detail.ecart_brut_pct ?? 0);
  const estSousEvalue = ecartBrutPct < 0;
  const decoteBrutePosPct = Number((-ecartBrutPct).toFixed(1));
  const travauxEst = Number(detail.cout_travaux_estime || 0);
  const topComparables = (detail.comparables || []).slice(0, 5);

  // Barre de positionnement par rapport aux biens similaires
  const prixSimilaires = topComparables
    .map((c: any) => (c.pm2_ajuste ? c.pm2_ajuste * Number(detail.surface_m2 || 1) : Number(c.prix || 0)))
    .filter((v: number) => v > 0);
  const allRefs = [
    ...prixSimilaires,
    Number(detail.prix_q10 || 0),
    Number(detail.prix_q90 || 0),
    Number(detail.prix_affiche || 0),
  ].filter((v) => v > 0);
  const barMin = allRefs.length ? Math.min(...allRefs) * 0.96 : 1;
  const barMax = allRefs.length ? Math.max(...allRefs) * 1.04 : 100;
  const posPct = (v: number) => Math.max(2, Math.min(98, ((v - barMin) / Math.max(barMax - barMin, 1)) * 100));

  const goSimulateur = () => {
    navigate("/simulateur", {
      state: {
        prix_achat_fai: detail.prix_cible_nego || detail.prix_affiche,
        surface_m2: detail.surface_m2,
        travaux_total_euro: travauxEst,
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
        travaux_euro: travauxEst,
      },
    });
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-5xl bg-white border border-slate-200 rounded-2xl shadow-2xl max-h-[90vh] overflow-y-auto p-6 space-y-6 text-slate-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-200 pb-4">
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <VerdictBadge verdict={detail.verdict} />
              <span className="text-xs text-slate-500 flex items-center gap-1">
                <MapPin className="w-3.5 h-3.5 text-sky-600" />
                {detail.ville} {detail.quartier ? `• ${detail.quartier}` : ""}
              </span>
            </div>
            <h2 className="text-xl font-bold mt-2 text-slate-900">
              {detail.titre || `${detail.type_label || "Bien"} ${detail.surface_m2} m²`}
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              {detail.surface_m2} m² • {detail.pieces || "?"} pièces • DPE {detail.dpe || "NC"}
              {detail.etage !== null && detail.etage !== undefined ? ` • Étage ${detail.etage}` : ""}
              {detail.annee_construction ? ` • Année ${detail.annee_construction}` : ""}
            </p>
          </div>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-slate-100 text-slate-500" title="Fermer (Échap)">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Galerie photos */}
        {detail.photos_urls && detail.photos_urls.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {detail.photos_urls.slice(0, 6).map((url: string, idx: number) => (
              <img
                key={idx}
                src={url}
                alt={`Photo ${idx + 1}`}
                className="w-full h-40 object-cover rounded-xl border border-slate-200 bg-slate-100"
                onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
              />
            ))}
          </div>
        )}

        {/* KPIs estimation (décote affichée en positif) */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5">
            <div className="text-xs font-medium text-slate-500">Prix affiché (FAI)</div>
            <div className="text-lg font-bold text-slate-900 mt-0.5">{fmt(detail.prix_affiche)}</div>
          </div>
          <div className={`border rounded-xl p-3.5 ${estSousEvalue ? "bg-emerald-50/70 border-emerald-200" : "bg-rose-50/70 border-rose-200"}`}>
            <div className={`text-xs font-medium ${estSousEvalue ? "text-emerald-700" : "text-rose-700"}`}>
              Justeprix estimé
            </div>
            <div className={`text-lg font-extrabold mt-0.5 ${estSousEvalue ? "text-emerald-700" : "text-rose-700"}`}>
              {fmt(detail.prix_hybride)}
            </div>
            <div className={`text-xs font-bold ${estSousEvalue ? "text-emerald-700" : "text-rose-700"}`}>
              {estSousEvalue
                ? `+${decoteBrutePosPct} % de décote`
                : `+${Math.abs(ecartBrutPct).toFixed(1)} % au-dessus du marché`}
            </div>
          </div>
          <div className="bg-sky-50/70 border border-sky-200 rounded-xl p-3.5">
            <div className="text-xs font-medium text-sky-700">Fourchette resserrée</div>
            <div className="text-sm font-bold text-sky-900 mt-1">{fmt(detail.prix_q10)} – {fmt(detail.prix_q90)}</div>
            <div className="text-xs text-slate-500">Largeur {detail.intervalle_largeur_pct}%</div>
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5">
            <div className="text-xs font-medium text-slate-500">Confiance & Travaux</div>
            <div className="text-lg font-bold text-sky-700 mt-0.5">{detail.score_confiance}/100</div>
            <div className="text-xs text-amber-700 font-medium">Travaux est. : {fmt(travauxEst)}</div>
          </div>
        </div>

        {/* Barre de positionnement par rapport aux biens similaires */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-sm text-slate-900">Positionnement du prix face aux biens similaires</h3>
            <span className="text-xs text-slate-500">
              Fourchette marché : {fmt(detail.prix_q10)} – {fmt(detail.prix_q90)}
            </span>
          </div>
          <div className="relative h-8 bg-slate-200 rounded-full overflow-visible">
            {/* Zone fourchette [Q10-Q90] */}
            <div
              className="absolute top-1 bottom-1 bg-sky-200 border border-sky-400 rounded-full"
              style={{
                left: `${posPct(Number(detail.prix_q10 || 0))}%`,
                width: `${Math.max(4, posPct(Number(detail.prix_q90 || 0)) - posPct(Number(detail.prix_q10 || 0)))}%`,
              }}
            />
            {/* Repère Justeprix */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-sky-700"
              style={{ left: `${posPct(Number(detail.prix_hybride || 0))}%` }}
              title={`Justeprix : ${fmt(detail.prix_hybride)}`}
            />
            {/* Curseur Prix affiché */}
            <div
              className={`absolute -top-1 w-4 h-10 rounded-full border-2 border-white shadow ${estSousEvalue ? "bg-emerald-600" : "bg-rose-600"}`}
              style={{ left: `calc(${posPct(Number(detail.prix_affiche || 0))}% - 8px)` }}
              title={`Ce bien (affiché) : ${fmt(detail.prix_affiche)}`}
            />
          </div>
          <div className="flex justify-between text-[11px] text-slate-500">
            <span>Plancher observé : {fmt(barMin)}</span>
            <span className="font-semibold text-sky-700">Justeprix : {fmt(detail.prix_hybride)}</span>
            <span>Plafond observé : {fmt(barMax)}</span>
          </div>
        </div>

        {/* Boutons d'action */}
        <div className="flex flex-wrap gap-3">
          <button onClick={goSimulateur} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold shadow-sm">
            <Calculator className="w-4 h-4" /> Simuler l'investissement locatif
          </button>
          <button onClick={goNego} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-sm font-semibold shadow-sm">
            <Scale className="w-4 h-4" /> Préparer la négociation
          </button>
          {detail.url && (
            <a href={detail.url} target="_blank" rel="noreferrer" className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-800 text-sm font-medium">
              <ExternalLink className="w-4 h-4" /> Voir l'annonce d'origine
            </a>
          )}
        </div>

        {/* SHAP en 2 colonnes claires (+ / -) */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <h3 className="font-bold text-sm text-slate-900">
                Ce qui explique la valeur estimée de {fmt(detail.prix_estime_ml)}
              </h3>
              <p className="text-xs text-slate-500">
                Écarts en % par rapport à un logement standard de la ville (survolez * pour comprendre chaque critère).
              </p>
            </div>
            {allShap.length > 8 && (
              <button onClick={() => setShowAllShap((s) => !s)} className="text-xs font-semibold text-sky-700 hover:underline">
                {showAllShap ? "Afficher moins" : `Voir les ${allShap.length} critères (+${allShap.length - 8})`}
              </button>
            )}
          </div>
          {shapVisibles.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Colonne (+) */}
              <div className="space-y-1.5 bg-white border border-emerald-200 rounded-xl p-3">
                <div className="text-xs font-bold text-emerald-700 flex items-center gap-1.5 mb-2">
                  <TrendingUp className="w-4 h-4" /> Ce qui tire le prix vers le haut (+)
                </div>
                {shapPos.length === 0 ? (
                  <p className="text-xs text-slate-400">Aucun facteur positif majeur.</p>
                ) : shapPos.map((f: any, idx: number) => (
                  <div key={idx} className="flex items-center justify-between text-xs border-b border-slate-100 last:border-0 py-1.5">
                    <span className="flex items-center gap-1 text-slate-700 font-medium">
                      <span>{f.name}</span>
                      {f.description && (
                        <span title={f.description} className="text-sky-600 cursor-help inline-flex items-center">
                          *<Info className="w-3 h-3 ml-0.5" />
                        </span>
                      )}
                    </span>
                    <span className="font-bold text-emerald-700 shrink-0 ml-2">
                      +{f.impact_pct}% (+{f.impact_euro.toLocaleString("fr-FR")} €)
                    </span>
                  </div>
                ))}
              </div>
              {/* Colonne (-) */}
              <div className="space-y-1.5 bg-white border border-rose-200 rounded-xl p-3">
                <div className="text-xs font-bold text-rose-700 flex items-center gap-1.5 mb-2">
                  <TrendingDown className="w-4 h-4" /> Ce qui explique un prix plus bas (-)
                </div>
                {shapNeg.length === 0 ? (
                  <p className="text-xs text-slate-400">Aucun facteur pénalisant majeur.</p>
                ) : shapNeg.map((f: any, idx: number) => (
                  <div key={idx} className="flex items-center justify-between text-xs border-b border-slate-100 last:border-0 py-1.5">
                    <span className="flex items-center gap-1 text-slate-700 font-medium">
                      <span>{f.name}</span>
                      {f.description && (
                        <span title={f.description} className="text-sky-600 cursor-help inline-flex items-center">
                          *<Info className="w-3 h-3 ml-0.5" />
                        </span>
                      )}
                    </span>
                    <span className="font-bold text-rose-700 shrink-0 ml-2">
                      {f.impact_pct}% ({f.impact_euro.toLocaleString("fr-FR")} €)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-500">Facteurs SHAP non disponibles pour cette annonce.</p>
          )}
        </div>

        {/* Carte de localisation précise */}
        {detail.latitude && detail.longitude && (
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm text-slate-900 flex items-center gap-1.5">
                <MapPin className="w-4 h-4 text-sky-600" />
                Localisation exacte ({Number(detail.latitude).toFixed(5)}, {Number(detail.longitude).toFixed(5)})
              </h3>
              <a
                href={`https://www.openstreetmap.org/?mlat=${detail.latitude}&mlon=${detail.longitude}#map=16/${detail.latitude}/${detail.longitude}`}
                target="_blank"
                rel="noreferrer"
                className="text-xs font-semibold text-sky-700 hover:underline flex items-center gap-1"
              >
                <ExternalLink className="w-3.5 h-3.5" /> Ouvrir la carte
              </a>
            </div>
            <iframe
              title="Localisation du bien"
              className="w-full h-56 rounded-lg border border-slate-200"
              loading="lazy"
              src={`https://www.openstreetmap.org/export/embed.html?bbox=${Number(detail.longitude) - 0.008}%2C${Number(detail.latitude) - 0.005}%2C${Number(detail.longitude) + 0.008}%2C${Number(detail.latitude) + 0.005}&layer=mapnik&marker=${detail.latitude}%2C${detail.longitude}`}
            />
          </div>
        )}

        {/* Comparables enrichis (max 5) */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
          <h3 className="font-bold text-sm text-slate-900 mb-3">
            Comparables voisins de taille similaire ({topComparables.length})
          </h3>
          {topComparables.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="text-slate-500 border-b border-slate-200 uppercase">
                  <tr>
                    <th className="py-2 pr-2">Source</th>
                    <th className="py-2 px-2">Bien & Caractéristiques</th>
                    <th className="py-2 px-2">Date</th>
                    <th className="py-2 px-2 text-right">Dist.</th>
                    <th className="py-2 px-2 text-right">Surf.</th>
                    <th className="py-2 px-2 text-right">Prix (€/m²)</th>
                    <th className="py-2 px-2 text-right">€/m² ajusté</th>
                    <th className="py-2 pl-2 text-right">Poids / Sim.</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200/70">
                  {topComparables.map((c: any, idx: number) => (
                    <tr key={idx} className="hover:bg-white">
                      <td className="py-2 pr-2">
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${c.source.includes("DVF") ? "bg-amber-100 text-amber-800" : "bg-sky-100 text-sky-800"}`}>
                          {c.source}
                        </span>
                      </td>
                      <td className="py-2 px-2">
                        <div className="font-medium text-slate-800 flex items-center gap-1.5 flex-wrap">
                          {c.url ? (
                            <a
                              href={c.url}
                              target="_blank"
                              rel="noreferrer"
                              className="text-sky-700 hover:underline inline-flex items-center gap-1"
                            >
                              <span>{c.titre}</span>
                              <ExternalLink className="w-3 h-3 shrink-0" />
                            </a>
                          ) : (
                            <span>{c.titre}</span>
                          )}
                        </div>
                        <div className="text-[11px] text-slate-500 flex flex-wrap gap-1.5 mt-0.5">
                          {c.quartier && <span>{c.quartier}</span>}
                          {c.pieces && <span>• T{c.pieces}</span>}
                          {c.dpe && <span>• DPE {c.dpe}</span>}
                          {c.a_parking && <span className="text-emerald-700">• Parking</span>}
                          {c.a_exterieur && <span className="text-emerald-700">• Extérieur</span>}
                          {c.a_ascenseur && <span className="text-emerald-700">• Ascenseur</span>}
                        </div>
                      </td>
                      <td className="py-2 px-2 text-slate-600">{c.date_transaction || "—"}</td>
                      <td className="py-2 px-2 text-right">{c.distance_m} m</td>
                      <td className="py-2 px-2 text-right">{c.surface_m2} m²</td>
                      <td className="py-2 px-2 text-right font-semibold">{fmt(c.prix)} ({Math.round(c.pm2)} €/m²)</td>
                      <td className="py-2 px-2 text-right font-semibold text-sky-700">
                        {c.pm2_ajuste ? `${Math.round(c.pm2_ajuste)} €/m²` : "—"}
                      </td>
                      <td className="py-2 pl-2 text-right">
                        <span className="font-semibold">{c.poids_pct ? `${c.poids_pct}%` : "—"}</span>
                        <span className="text-slate-400 ml-1">({c.similarite_pct}%)</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-xs text-slate-500">Aucun comparable voisin dans le rayon retenu.</p>
          )}
        </div>

        {/* Description intégrale */}
        {detail.description && (
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <h3 className="font-bold text-sm text-slate-900 mb-2">Description de l'annonce</h3>
            <p className="text-sm text-slate-700 whitespace-pre-line leading-relaxed">{detail.description}</p>
          </div>
        )}
      </div>
    </div>
  );
}
