import { NavLink, Route, Routes } from "react-router-dom";
import { Activity, Calculator, Home, Link2, Scale } from "lucide-react";
import RadarPage from "./pages/RadarPage";
import CheckUrlPage from "./pages/CheckUrlPage";
import SimulatorPage from "./pages/SimulatorPage";
import DvfPage from "./pages/DvfPage";
import MonitoringPage from "./pages/MonitoringPage";

const NAV_ITEMS = [
  { to: "/", label: "1. Radar Décotes", icon: Home },
  { to: "/verifier", label: "2. Vérifier URL / Estimer", icon: Link2 },
  { to: "/simulateur", label: "3. Simulateur Locatif", icon: Calculator },
  { to: "/marche-dvf", label: "4. Notaires DVF & Négo", icon: Scale },
  { to: "/monitoring", label: "5. Monitoring & Drift", icon: Activity },
];

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-full bg-emerald-400" />
            <span className="font-bold text-lg tracking-tight">IA-Immobilier</span>
            <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-sky-400 font-medium">V1</span>
          </div>
          <nav className="flex flex-wrap gap-2">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition ${
                    isActive
                      ? "bg-sky-500/20 text-sky-300 border border-sky-500/40"
                      : "text-slate-400 hover:text-slate-100 hover:bg-slate-800"
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
        </Routes>
      </main>
    </div>
  );
}
