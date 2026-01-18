import { component$, useSignal, useVisibleTask$ } from "@builder.io/qwik";
import type { DocumentHead } from "@builder.io/qwik-city";

interface Trajectory {
  id: string;
  status: string;
  model: string;
  createdAt: string;
  score: number | null;
  certificateId: string | null;
  messageCount: number;
  failureReason?: string;
}

export default component$(() => {
  const trajectories = useSignal<Trajectory[]>([]);
  const loading = useSignal(true);
  const filter = useSignal("all");

  // eslint-disable-next-line qwik/no-use-visible-task
  useVisibleTask$(async () => {
    try {
      const response = await fetch("/api/arena/trajectories");
      const data = await response.json();
      trajectories.value = data.trajectories;
    } catch {
      console.error("Failed to fetch trajectories");
    } finally {
      loading.value = false;
    }
  });

  const filteredTrajectories = () => {
    if (filter.value === "all") return trajectories.value;
    return trajectories.value.filter((t) => t.status === filter.value);
  };

  return (
    <>
      <div class="section-header">
        <h2>Trajectories</h2>
        <p>All agent trajectories processed through the shadow arena</p>
      </div>

      <div class="card" style={{ marginBottom: "1.5rem" }}>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          {["all", "verified", "pending", "failed"].map((f) => (
            <button
              key={f}
              class={`btn ${filter.value === f ? "btn-primary" : "btn-secondary"}`}
              style={{ padding: "0.5rem 1rem", fontSize: "0.75rem", textTransform: "capitalize" }}
              onClick$={() => (filter.value = f)}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      <div class="card">
        {loading.value ? (
          <div class="empty-state">
            <p>Loading trajectories...</p>
          </div>
        ) : filteredTrajectories().length === 0 ? (
          <div class="empty-state">
            <h3>No trajectories found</h3>
            <p>Trajectories will appear here once processed through the shadow arena.</p>
          </div>
        ) : (
          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Status</th>
                  <th>Model</th>
                  <th>Messages</th>
                  <th>Score</th>
                  <th>Certificate</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {filteredTrajectories().map((t) => (
                  <tr key={t.id}>
                    <td><code>{t.id}</code></td>
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
                    <td>{t.messageCount}</td>
                    <td>{t.score !== null ? t.score.toFixed(2) : "-"}</td>
                    <td>
                      {t.certificateId ? (
                        <code style={{ fontSize: "0.75rem" }}>{t.certificateId}</code>
                      ) : (
                        "-"
                      )}
                    </td>
                    <td style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>
                      {new Date(t.createdAt).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
});

export const head: DocumentHead = {
  title: "Trajectories - Shadow Arena",
};
