import { component$ } from "@builder.io/qwik";
import type { DocumentHead } from "@builder.io/qwik-city";

export default component$(() => {
  const verifiers = [
    {
      name: "Leash",
      type: "Network Policy",
      engine: "Cedar",
      status: "online",
      description: "Kernel-level network policy enforcement using Cedar policies",
      endpoint: "http://leash:8091",
      lastCheck: "2 seconds ago",
      policiesLoaded: 12,
    },
    {
      name: "OPA",
      type: "Infrastructure Policy",
      engine: "Rego",
      status: "online",
      description: "Infrastructure compliance checking with Rego policies",
      endpoint: "http://opa:8181",
      lastCheck: "5 seconds ago",
      policiesLoaded: 24,
    },
    {
      name: "Firecracker",
      type: "Code Execution",
      engine: "MicroVM",
      status: "online",
      description: "Sandboxed code execution in ephemeral microVMs",
      endpoint: "http://firecracker:8090",
      lastCheck: "3 seconds ago",
      vmsAvailable: 8,
    },
    {
      name: "SGLang",
      type: "Model Serving",
      engine: "RadixAttention",
      status: "online",
      description: "High-performance model serving with prefix caching",
      endpoint: "http://sglang-router:30000",
      lastCheck: "1 second ago",
      cacheHitRate: "73.5%",
    },
  ];

  return (
    <>
      <div class="section-header">
        <h2>Verifiers</h2>
        <p>Hard verification infrastructure status and configuration</p>
      </div>

      <div style={{ display: "grid", gap: "1rem" }}>
        {verifiers.map((v) => (
          <div class="card" key={v.name}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
              <div>
                <h3 style={{ fontSize: "1.125rem", marginBottom: "0.25rem" }}>{v.name}</h3>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>{v.description}</p>
              </div>
              <span class={`badge badge-${v.status === "online" ? "success" : "error"}`}>
                {v.status}
              </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "1rem" }}>
              <div>
                <div class="stat-label">Type</div>
                <div style={{ fontSize: "0.875rem" }}>{v.type}</div>
              </div>
              <div>
                <div class="stat-label">Engine</div>
                <div style={{ fontSize: "0.875rem" }}>{v.engine}</div>
              </div>
              <div>
                <div class="stat-label">Endpoint</div>
                <code style={{ fontSize: "0.75rem" }}>{v.endpoint}</code>
              </div>
              <div>
                <div class="stat-label">Last Check</div>
                <div style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>{v.lastCheck}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div class="card" style={{ marginTop: "1.5rem" }}>
        <div class="card-header">
          <h3 class="card-title">Verification Pipeline</h3>
        </div>
        <div class="code-block">
{`┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Request   │────►│    Leash    │────►│     OPA     │
│   (Agent)   │     │   (Cedar)   │     │   (Rego)    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │ Firecracker │     │   SGLang    │
                    │  (Sandbox)  │     │   (Model)   │
                    └─────────────┘     └─────────────┘
                           │                   │
                           └─────────┬─────────┘
                                     ▼
                           ┌─────────────────┐
                           │   Certificate   │
                           │    (Proof)      │
                           └─────────────────┘`}
        </div>
      </div>
    </>
  );
});

export const head: DocumentHead = {
  title: "Verifiers - Shadow Arena",
};
