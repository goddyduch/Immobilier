import { useState } from "react";
import { api } from "../api/client";
import PropertyDrawer from "../components/PropertyDrawer";

export default function CheckUrlPage() {
  const [url, setUrl] = useState("");
  const [resUrl, setResUrl] = useState<any>(null);
  const [loadingUrl, setLoadingUrl] = useState(false);

  const [form, setForm] = useState({
    type_bien: "flat",
    surface_m2: 60,
    pieces: 3,
    ville: "Poitiers",
    code_postal: "86000",
    quartier: "Centre-ville",
    latitude: "" as string | number,
    longitude: "" as string | number,
    dpe: "D",
    annee_construction: "" as string | number,
    prix_affiche: "" as string | number,
    niveau_travaux_manuel: 0,
    description: "",
    has_parking: false,
    has_terrasse: false,
    has_jardin: false,
    has_ascenseur: false,
  });
  const [adresseQuery, setAdresseQuery] = useState("");
  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [estRes, setEstRes] = useState<any>(null);
  const [loadingEst, setLoadingEst] = useState(false);
  const [err, setErr] = useState("");
  const [drawerDetail, setDrawerDetail] = useState<any>(null);

  const handleCheckUrl = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoadingUrl(true);
    setErr("");
    setResUrl(null);
    try {
      const r = await api.checkUrl(url);
      setResUrl(r);
      if (r.detail) setDrawerDetail(r.detail);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setLoadingUrl(false);
    }
  };

  const searchBan = async (q: string) => {
    setAdresseQuery(q);
    if (q.length < 4) { setSuggestions([]); return; }
    try {
      const r = await fetch(`https://api-adresse.data.gouv.fr/search/?q=${encodeURIComponent(q)}&limit=5`);
      const j = await r.json();
      setSuggestions(j.features || []);
    } catch {
      setSuggestions([]);
    }
  };

  const pickBan = (feat: any) => {
    const [lon, lat] = feat.geometry.coordinates;
    const p = feat.properties;
    setForm((f) => ({
      ...f,
      ville: p.city || f.ville,
      code_postal: p.postcode || f.code_postal,
      quartier: p.district || p.city || f.quartier,
      latitude: lat,
      longitude: lon,
    }));
    setAdresseQuery(p.label);
    setSuggestions([]);
  };

  const handleEstimate = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoadingEst(true);
    setErr("");
    try {
      const payload: any = {
        ...form,
        surface_m2: Number(form.surface_m2),
        pieces: Number(form.pieces),
        latitude: form.latitude !== "" ? Number(form.latitude) : null,
        longitude: form.longitude !== "" ? Number(form.longitude) : null,
        annee_construction: form.annee_construction !== "" ? Number(form.annee_construction) : null,
        prix_affiche: form.prix_affiche !== "" ? Number(form.prix_affiche) : null,
        niveau_travaux_manuel: Number(form.niveau_travaux_manuel),
      };
      const r = await api.estimate(payload);
      setEstRes(r);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setLoadingEst(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Vérificateur d'annonce par URL & Estimation rapide</h1>
        <p className="text-sm text-slate-400">Colle un lien d'annonce pour vérifier si elle est en base, ou estime un bien manuellement via l'API BAN.</p>
      </div>

      {/* Bloc URL */}
      <form onSubmit={handleCheckUrl} className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <label className="block text-sm font-semibold">1. Vérifier une annonce par lien (URL)</label>
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="url"
            required
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://www.bienici.com/annonce/..."
            className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm"
          />
          <button disabled={loadingUrl} className="px-5 py-2.5 rounded-lg bg-sky-600 hover:bg-sky-500 font-semibold text-sm disabled:opacity-50">
            {loadingUrl ? "Analyse..." : "Analyser le lien"}
          </button>
        </div>
        {resUrl && (
          <div className={`p-3 rounded-lg text-sm border ${resUrl.detail ? "bg-emerald-500/10 border-emerald-500/40 text-emerald-300" : "bg-amber-500/10 border-amber-500/40 text-amber-300"}`}>
            {resUrl.message}
          </div>
        )}
      </form>

      {/* Bloc Formulaire Rapide */}
      <form onSubmit={handleEstimate} className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <h2 className="text-base font-semibold">2. Formulaire d'estimation rapide (avec autocomplétion adresse BAN)</h2>
        <div className="relative">
          <input
            type="text"
            value={adresseQuery}
            onChange={(e) => searchBan(e.target.value)}
            placeholder="Rechercher une adresse à Poitiers / Grand Poitiers..."
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm"
          />
          {suggestions.length > 0 && (
            <div className="absolute z-20 w-full mt-1 bg-slate-950 border border-slate-700 rounded-lg shadow-xl">
              {suggestions.map((s: any, i: number) => (
                <button
                  type="button"
                  key={i}
                  onClick={() => pickBan(s)}
                  className="w-full text-left px-4 py-2 text-sm hover:bg-slate-800 border-b border-slate-800/50 last:border-0"
                >
                  {s.properties.label}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
          <div>
            <label className="text-xs text-slate-400">Type</label>
            <select value={form.type_bien} onChange={(e) => setForm({ ...form, type_bien: e.target.value })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1">
              <option value="flat">Appartement</option>
              <option value="house">Maison</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-400">Surface (m²)</label>
            <input type="number" required value={form.surface_m2} onChange={(e) => setForm({ ...form, surface_m2: Number(e.target.value) })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1" />
          </div>
          <div>
            <label className="text-xs text-slate-400">Pièces</label>
            <input type="number" value={form.pieces} onChange={(e) => setForm({ ...form, pieces: Number(e.target.value) })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1" />
          </div>
          <div>
            <label className="text-xs text-slate-400">DPE</label>
            <select value={form.dpe} onChange={(e) => setForm({ ...form, dpe: e.target.value })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1">
              {["A", "B", "C", "D", "E", "F", "G"].map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-400">Prix affiché (€, optionnel)</label>
            <input type="number" value={form.prix_affiche} onChange={(e) => setForm({ ...form, prix_affiche: e.target.value })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1" />
          </div>
          <div>
            <label className="text-xs text-slate-400">Année construction</label>
            <input type="number" value={form.annee_construction} onChange={(e) => setForm({ ...form, annee_construction: e.target.value })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1" />
          </div>
          <div className="col-span-2">
            <label className="text-xs text-slate-400">Niveau de travaux</label>
            <select value={form.niveau_travaux_manuel} onChange={(e) => setForm({ ...form, niveau_travaux_manuel: Number(e.target.value) })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 mt-1">
              <option value={0}>0 — Aucun travaux (0 €/m²)</option>
              <option value={1}>1 — Travaux légers (400 €/m²)</option>
              <option value={2}>2 — Rénovation (750 €/m²)</option>
              <option value={3}>3 — Remise à neuf (1 200 €/m²)</option>
              <option value={4}>4 — Rénovation lourde (2 000 €/m²)</option>
            </select>
          </div>
        </div>

        <div className="flex flex-wrap gap-4 text-xs text-slate-300">
          {[
            ["has_parking", "Parking / Garage"],
            ["has_terrasse", "Terrasse"],
            ["has_jardin", "Jardin"],
            ["has_ascenseur", "Ascenseur"],
          ].map(([k, lbl]) => (
            <label key={k} className="flex items-center gap-1.5">
              <input type="checkbox" checked={(form as any)[k]} onChange={(e) => setForm({ ...form, [k]: e.target.checked })} />
              {lbl}
            </label>
          ))}
        </div>

        <button disabled={loadingEst} className="px-6 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 font-semibold text-sm disabled:opacity-50">
          {loadingEst ? "Calcul en cours..." : "Estimer le bien"}
        </button>

        {err && <p className="text-sm text-rose-400">{err}</p>}
      </form>

      {estRes && (
        <div className="bg-slate-900 border border-emerald-500/40 rounded-xl p-5 grid grid-cols-2 sm:grid-cols-5 gap-4">
          <div>
            <div className="text-xs text-slate-400">Estimation Hybride</div>
            <div className="text-2xl font-bold text-emerald-400">{Math.round(estRes.prix_hybride).toLocaleString("fr-FR")} €</div>
            <div className="text-xs text-slate-400">Latence {estRes.latence_ms} ms</div>
          </div>
          <div>
            <div className="text-xs text-slate-400">Fourchette [Q10–Q90]</div>
            <div className="text-sm font-semibold text-sky-300 mt-1">
              {Math.round(estRes.prix_q10).toLocaleString("fr-FR")} € – {Math.round(estRes.prix_q90).toLocaleString("fr-FR")} €
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-400">Verdict & Confiance</div>
            <div className="text-lg font-bold">{estRes.verdict}</div>
            <div className="text-xs text-sky-400">{estRes.score_confiance}/100</div>
          </div>
          <div>
            <div className="text-xs text-slate-400">Travaux estimés</div>
            <div className="text-lg font-bold text-amber-400">{Math.round(estRes.cout_travaux_estime).toLocaleString("fr-FR")} €</div>
          </div>
          <div>
            <div className="text-xs text-slate-400">Loyer estimé</div>
            <div className="text-lg font-bold">{Math.round(estRes.loyer_mensuel_estime)} €/mois</div>
            <div className="text-xs text-slate-400">[{estRes.loyer_bas}–{estRes.loyer_haut} €]</div>
          </div>
        </div>
      )}

      <PropertyDrawer detail={drawerDetail} onClose={() => setDrawerDetail(null)} />
    </div>
  );
}
