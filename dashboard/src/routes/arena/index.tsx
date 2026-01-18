import { component$, useSignal, useVisibleTask$ } from "@builder.io/qwik";
import type { DocumentHead } from "@builder.io/qwik-city";

// Mock data - will be replaced with API calls
const mockStats = {
  totalTrajectories: 12847,
  certificatesIssued: 11203,
  verificationRate: 87.2,
  dpoTripletsGenerated: 3421,
};

const mockRecentTrajectories = [
  { id: "traj-001", status: "verified", model: "llama-3.1-70b", timestamp: "2 min ago", score: 0.94 },
  { id: "traj-002", status: "pending", model: "llama-3.1-70b", timestamp: "5 min ago", score: null },
  { id: "traj-003", status: "failed", model: "llama-3.1-8b", timestamp: "8 min ago", score: 0.31 },
  { id: "traj-004", status: "verified", model: "llama-3.1-70b", timestamp: "12 min ago", score: 0.89 },
  { id: "traj-005", status: "verified", model: "mistral-7b", timestamp: "15 min ago", score: 0.76 },
];

export default component$(() => {
  const stats = useSignal(mockStats);
  const trajectories = useSignal(mockRecentTrajectories);
  const loading = useSignal(true);

  // eslint-disable-next-line qwik/no-use-visible-task
  useVisibleTask$(() => {
    // TODO: Replace with actual API call
    // fetch('/api/arena/stats').then(r => r.json()).then(data => stats.value = data);
    setTimeout(() => {
      loading.value = false;
    }, 500);
  });

  return (
    <>
      <div class="section-header">
        <h2>Shadow Arena Overview</h2>
        <p>Real-time verification and trajectory comparison</p>
      </div>

      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-label">Total Trajectories</div>
          <div class="stat-value">{stats.value.totalTrajectories.toLocaleString()}</div>
          <div class="stat-change positive">+124 today</div>
        </div>

        <div class="stat-card">
          <div class="stat-label">Certificates Issued</div>
          <div class="stat-value">{stats.value.certificatesIssued.toLocaleString()}</div>
          <div class="stat-change positive">+98 today</div>
        </div>

        <div class="stat-card">
          <div class="stat-label">Verification Rate</div>
          <div class="stat-value">{stats.value.verificationRate}%</div>
          <div class="stat-change positive">+2.1% this week</div>
        </div>

        <div class="stat-card">
          <div class="stat-label">DPO Triplets</div>
          <div class="stat-value">{stats.value.dpoTripletsGenerated.toLocaleString()}</div>
          <div class="stat-change positive">+45 today</div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h3 class="card-title">Recent Trajectories</h3>
          <a href="/arena/trajectories" class="btn btn-secondary" style={{ padding: "0.5rem 1rem", fontSize: "0.75rem" }}>
            View All
          </a>
        </div>

        {loading.value ? (
          <div class="empty-state">
            <p>Loading trajectories...</p>
          </div>
        ) : (
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
                    <td><code>{t.id}</code></td>
                    <td>
                      <span class={`badge badge-${t.status === "verified" ? "success" : t.status === "pending" ? "warning" : "error"}`}>
                        {t.status}
                      </span>
                    </td>
                    <td>{t.model}</td>
                    <td>{t.score !== null ? t.score.toFixed(2) : "-"}</td>
                    <td style={{ color: "var(--text-secondary)" }}>{t.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginTop: "1.5rem" }}>
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Verifier Status</h3>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>Leash (Cedar)</span>
              <span class="badge badge-success">Online</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>OPA (Rego)</span>
              <span class="badge badge-success">Online</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>Firecracker</span>
              <span class="badge badge-success">Online</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>SGLang Router</span>
              <span class="badge badge-success">Online</span>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Shadow Loop Config</h3>
          </div>
          <div class="code-block">
{`variations: 3
temperatures: [0.1, 0.7, 1.0]
max_tokens: 4096
timeout_ms: 30000
cache: radix_attention`}
          </div>
        </div>
      </div>
    </>
  );
});

export const head: DocumentHead = {
  title: "Shadow Arena - RLMArena",
};
