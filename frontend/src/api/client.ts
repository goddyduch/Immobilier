const API_BASE = import.meta.env.VITE_API_BASE || "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data?.error?.message || `Erreur API (${res.status})`);
  }
  return data as T;
}

export const api = {
  getOpportunities: (params: Record<string, string | number>) => {
    const qs = new URLSearchParams(
      Object.entries(params)
        .filter(([, v]) => v !== "" && v !== undefined && v !== null)
        .map(([k, v]) => [k, String(v)])
    ).toString();
    return request<any>(`/v1/opportunities?${qs}`);
  },
  getOpportunityDetail: (id: string) => request<any>(`/v1/opportunities/${encodeURIComponent(id)}`),
  estimate: (payload: any) =>
    request<any>("/v1/estimate", { method: "POST", body: JSON.stringify(payload) }),
  checkUrl: (url: string) =>
    request<any>("/v1/check-url", { method: "POST", body: JSON.stringify({ url }) }),
  simulateInvestment: (payload: any) =>
    request<any>("/v1/simulate-investment", { method: "POST", body: JSON.stringify(payload) }),
  getDvfAnalysis: (params: { ville?: string; type_bien?: string }) => {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v && v !== "Toutes" && v !== "Tous") as [string, string][]
    ).toString();
    return request<any>(`/v1/dvf-analysis${qs ? `?${qs}` : ""}`);
  },
  simulateNegotiation: (payload: any) =>
    request<any>("/v1/dvf-negotiation", { method: "POST", body: JSON.stringify(payload) }),
  getMonitoringOverview: () => request<any>("/v1/monitoring/overview"),
  getEvaluation: () => request<any>("/v1/evaluation"),
};
