import { useEffect, useState } from "react";
import { api } from "../api/client";

export default function MonitoringPage() {
  const [snap, setSnap] = useState<any>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.getMonitoringOverview().then(setSnap).catch((e) => setErr(e.message));
  }, []);

  if (err) return <div className="text-rose-400">{err}</div>;
  if (!snap) return <div className="text-slate-400">Chargement du monitoring...</div>;

  const badgeColor =
    snap.statut_global === "ok"
      ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
      : snap.statut_global === "warning"
      ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
      : "bg-rose-500/20 text-rose-300 border-rose-500/40";

  const psiEntries = Object.entries(snap.data_drift_psi || {});

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Back-Office Monitoring : Drift, Données & Modèle</h1>
          <p className="text-sm text-slate-400">Version modèle : {snap.model_version}</p>
        </div>
        <span className={`px-3 py-1 rounded-full border text-sm font-bold uppercase ${badgeColor}`}>
          Statut : {snap.statut_global}
        </span>
      </div>

      {snap.alertes?.length > 0 && (
        <div className="space-y-2">
          {snap.alertes.map((a: any, i: number) => (
            <div key={i} className={`p-3 rounded-lg border text-sm ${a.niveau === "critical" ? "bg-rose-500/20 border-rose-500/40 text-rose-300" : "bg-amber-500/20 border-amber-500/40 text-amber-300"}`}>
              [{a.niveau.toUpperCase()}] {a.message}
            </div>
          ))}
        </div>
      )}

      {/* KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">Historique (train)</div>
          <div className="text-2xl font-bold">{snap.nb_historique}</div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">Offres actives</div>
          <div className="text-2xl font-bold text-sky-400">{snap.nb_actives}</div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">Quarantaine (rejets)</div>
          <div className="text-2xl font-bold text-amber-400">{snap.nb_quarantaine}</div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">Opportunités fortes</div>
          <div className="text-2xl font-bold text-emerald-400">{snap.prediction_drift?.taux_opportunites_fortes_pct ?? "—"}%</div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-400">Largeur Q10-Q90 méd.</div>
          <div className="text-2xl font-bold">{snap.prediction_drift?.largeur_intervalle_median_pct ?? "—"}%</div>
        </div>
      </div>

      {/* Complétude & PSI */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2">
          <h2 className="font-semibold text-sm">Complétude des champs (%)</h2>
          {Object.entries(snap.completude || {})
            .filter(([k]) => k !== "par_source")
            .map(([k, v]: any) => (
              <div key={k} className="flex justify-between text-sm border-b border-slate-800/60 py-1.5">
                <span className="font-mono text-sky-300">{k}</span>
                <span className={`font-semibold ${v >= 85 ? "text-emerald-400" : v >= 70 ? "text-amber-400" : "text-rose-400"}`}>{v}%</span>
              </div>
            ))}
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2">
          <h2 className="font-semibold text-sm">Data Drift (PSI — &lt;0.10 OK, &gt;=0.25 Critique)</h2>
          {psiEntries.map(([k, v]: any) => (
            <div key={k} className="flex justify-between text-sm border-b border-slate-800/60 py-1.5">
              <span className="font-mono text-slate-300">{k}</span>
              <span className={`font-semibold ${v < 0.1 ? "text-emerald-400" : v < 0.25 ? "text-amber-400" : "text-rose-400"}`}>{v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Quarantaine */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
        <h2 className="font-semibold text-sm mb-2">Dernières annonces mises en quarantaine</h2>
        {snap.quarantaine_recente?.length > 0 ? (
          <div className="overflow-x-auto text-xs">
            <table className="w-full text-left">
              <thead className="text-slate-400 border-b border-slate-800">
                <tr><th className="py-2">ID</th><th className="py-2">Source</th><th className="py-2">Motif</th><th className="py-2">Date</th></tr>
              </thead>
              <tbody>
                {snap.quarantaine_recente.map((q: any, i: number) => (
                  <tr key={i} className="border-b border-slate-800/50">
                    <td className="py-1.5 font-mono">{q.id}</td>
                    <td className="py-1.5">{q.source}</td>
                    <td className="py-1.5 text-amber-400">{q.motif_rejet}</td>
                    <td className="py-1.5 text-slate-400">{String(q.detected_at).slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-xs text-slate-400">Aucune annonce rejetée en quarantaine.</p>
        )}
      </div>
    </div>
  );
}
