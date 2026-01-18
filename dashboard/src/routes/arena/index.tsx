/**
 * Shadow Arena Dashboard
 *
 * Uses routeLoader$ for SSR data fetching - optimal for edge deployment.
 * Data is preloaded during speculative navigation.
 */

import { component$, useSignal, $ } from "@builder.io/qwik";
import { type DocumentHead, routeLoader$, Link } from "@builder.io/qwik-city";

// Route loader - runs on server, cached at edge
export const useArenaStats = routeLoader$(async ({ env }) => {
  const gatewayEndpoint = env.get("GATEWAY_ENDPOINT") || "http://localhost:8080";

  try {
    // In production, fetch from gateway
    // const response = await fetch(`${gatewayEndpoint}/api/arena/stats`);
    // return await response.json();

    // Mock data for development
    return {
      totalTrajectories: 12847,
      certificatesIssued: 11203,
      verificationRate: 87.2,
      dpoTripletsGenerated: 3421,
      activeVerifiers: 4,
      shadowLoopVariations: 3,
      cacheHitRate: 73.5,
      averageLatencyMs: 245,
    };
  } catch {
    return {
      totalTrajectories: 0,
      certificatesIssued: 0,
      verificationRate: 0,
      dpoTripletsGenerated: 0,
      activeVerifiers: 0,
      shadowLoopVariations: 0,
      cacheHitRate: 0,
      averageLatencyMs: 0,
      error: "Gateway unavailable",
    };
  }
});

export const useRecentTrajectories = routeLoader$(async () => {
  // Mock data - replace with gateway fetch
  return [
    { id: "traj-001", status: "verified", model: "llama-3.1-70b", timestamp: "2 min ago", score: 0.94 },
    { id: "traj-002", status: "pending", model: "llama-3.1-70b", timestamp: "5 min ago", score: null },
    { id: "traj-003", status: "failed", model: "llama-3.1-8b", timestamp: "8 min ago", score: 0.31 },
    { id: "traj-004", status: "verified", model: "llama-3.1-70b", timestamp: "12 min ago", score: 0.89 },
    { id: "traj-005", status: "verified", model: "mistral-7b", timestamp: "15 min ago", score: 0.76 },
  ];
});

export default component$(() => {
  const stats = useArenaStats();
  const trajectories = useRecentTrajectories();
  const refreshing = useSignal(false);

  // Lazy-loaded refresh handler
  const onRefresh$ = $(() => {
    refreshing.value = true;
    // In real app, would invalidate cache and refetch
    setTimeout(() => {
      refreshing.value = false;
    }, 500);
  });

  return (
    <>
      <div class="section-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h2>Shadow Arena</h2>
            <p>Real-time verification and trajectory comparison</p>
          </div>
          <button
            class="btn btn-ghost"
            onClick$={onRefresh$}
            disabled={refreshing.value}
            style={{ padding: "0.5rem 1rem", fontSize: "0.75rem" }}
          >
            {refreshing.value ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div class="stats-grid">
        <StatCard
          label="Total Trajectories"
          value={stats.value.totalTrajectories.toLocaleString()}
          change="+124 today"
          positive
        />
        <StatCard
          label="Certificates Issued"
          value={stats.value.certificatesIssued.toLocaleString()}
          change="+98 today"
          positive
        />
        <StatCard
          label="Verification Rate"
          value={`${stats.value.verificationRate}%`}
          change="+2.1% this week"
          positive
        />
        <StatCard
          label="DPO Triplets"
          value={stats.value.dpoTripletsGenerated.toLocaleString()}
          change="+45 today"
          positive
        />
      </div>

      {/* Recent Trajectories */}
      <div class="card">
        <div class="card-header">
          <h3 class="card-title">Recent Trajectories</h3>
          <Link
            href="/arena/trajectories"
            class="btn btn-ghost"
            style={{ padding: "0.5rem 1rem", fontSize: "0.6875rem" }}
          >
            View All
          </Link>
        </div>

        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Trajectory ID</th>
                <th>Status</th>
                <th>Model</th>
                <th>Score</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {trajectories.value.map((t) => (
                <tr key={t.id}>
                  <td><code style={{ fontSize: "0.75rem" }}>{t.id}</code></td>
                  <td>
                    <span
                      class={`badge badge-${
                        t.status === "verified" ? "success" : t.status === "pending" ? "warning" : "error"
                      }`}
                    >
                      {t.status}
                    </span>
                  </td>
                  <td>{t.model}</td>
                  <td style={{ fontVariantNumeric: "tabular-nums" }}>
                    {t.score !== null ? t.score.toFixed(2) : "—"}
                  </td>
                  <td style={{ color: "var(--text-secondary)" }}>{t.timestamp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bottom Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginTop: "1.5rem" }}>
        {/* Verifier Status */}
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Verifier Status</h3>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            <VerifierStatus name="Leash (Cedar)" status="online" />
            <VerifierStatus name="OPA (Rego)" status="online" />
            <VerifierStatus name="Firecracker" status="online" />
            <VerifierStatus name="SGLang Router" status="online" />
          </div>
        </div>

        {/* Shadow Loop Config */}
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Shadow Loop Config</h3>
          </div>
          <div class="code-block">
{`variations: ${stats.value.shadowLoopVariations}
temperatures: [0.1, 0.7, 1.0]
max_tokens: 4096
timeout_ms: 30000
cache: radix_attention
cache_hit_rate: ${stats.value.cacheHitRate}%`}
          </div>
        </div>
      </div>

      {/* Latency Indicator */}
      <div class="card" style={{ marginTop: "1rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span class="mono-label">Average Verification Latency</span>
          <span class="mono" style={{ fontSize: "1.25rem", fontWeight: 700 }}>
            {stats.value.averageLatencyMs}ms
          </span>
        </div>
      </div>
    </>
  );
});

/**
 * Stat Card Component
 */
const StatCard = component$<{
  label: string;
  value: string;
  change: string;
  positive?: boolean;
}>(({ label, value, change, positive = true }) => {
  return (
    <div class="stat-card">
      <div class="stat-label">{label}</div>
      <div class="stat-value">{value}</div>
      <div class={`stat-change ${positive ? "positive" : "negative"}`}>{change}</div>
    </div>
  );
});

/**
 * Verifier Status Row
 */
const VerifierStatus = component$<{
  name: string;
  status: "online" | "offline" | "degraded";
}>(({ name, status }) => {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: "0.875rem" }}>{name}</span>
      <span
        class={`badge badge-${status === "online" ? "success" : status === "degraded" ? "warning" : "error"}`}
      >
        {status}
      </span>
    </div>
  );
});

export const head: DocumentHead = {
  title: "Shadow Arena — RLMArena",
  meta: [
    {
      name: "description",
      content: "Real-time verification dashboard for AI agent trajectories.",
    },
  ],
};
