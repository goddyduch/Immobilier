import { NavLink, Route, Routes } from "react-router-dom";
import { Activity, BarChart3, Calculator, Home, Link2, Scale } from "lucide-react";
import RadarPage from "./pages/RadarPage";
import CheckUrlPage from "./pages/CheckUrlPage";
import SimulatorPage from "./pages/SimulatorPage";
import DvfPage from "./pages/DvfPage";
import MonitoringPage from "./pages/MonitoringPage";
import EvaluationPage from "./pages/EvaluationPage";

const NAV_ITEMS = [
  { to: "/", label: "1. Radar Décotes", icon: Home },
  { to: "/verifier", label: "2. Vérifier URL / Estimer", icon: Link2 },
  { to: "/simulateur", label: "3. Simulateur Locatif", icon: Calculator },
  { to: "/marche-dvf", label: "4. Notaires DVF & Négo", icon: Scale },
  { to: "/monitoring", label: "5. Monitoring & Drift", icon: Activity },
  { to: "/evaluation", label: "6. Évaluation Modèle", icon: BarChart3 },
];

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      <header className="border-b border-slate-200 bg-white/95 backdrop-blur sticky top-0 z-30 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <span className="h-3 w-3 rounded-full bg-emerald-500 ring-4 ring-emerald-500/20" />
            <span className="font-extrabold text-xl tracking-tight text-slate-900">Justeprix</span>
            <span className="text-[11px] px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200 font-semibold">
              12 Villes-Centres
            </span>
          </div>
          <nav className="flex flex-wrap gap-1.5">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition ${
                    isActive
                      ? "bg-sky-600 text-white shadow-sm"
                      : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-6">
        <Routes>
          <Route path="/" element={<RadarPage />} />
          <Route path="/verifier" element={<CheckUrlPage />} />
          <Route path="/simulateur" element={<SimulatorPage />} />
          <Route path="/marche-dvf" element={<DvfPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
        </Routes>
      </main>
    </div>
  );
}
