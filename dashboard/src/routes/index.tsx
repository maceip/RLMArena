import { component$ } from "@builder.io/qwik";
import type { DocumentHead } from "@builder.io/qwik-city";

export default component$(() => {
  return (
    <>
      <nav class="nav">
        <div class="nav-inner">
          <a href="/" class="nav-logo">
            RLM<span>Arena</span>
          </a>
          <ul class="nav-links">
            <li><a href="#features">Features</a></li>
            <li><a href="#how-it-works">How it Works</a></li>
            <li><a href="/arena">Shadow Arena</a></li>
            <li><a href="https://github.com/maceip/RLMArena">GitHub</a></li>
          </ul>
        </div>
      </nav>

      <main class="container">
        <section class="hero">
          <h1>Verifier-as-a-Service for AI Agents</h1>
          <p>
            Production-grade verification infrastructure that validates AI agent outputs
            through parallel execution, formal proofs, and continuous evaluation.
          </p>
          <div class="hero-cta">
            <a href="/arena" class="btn btn-primary">
              Try Shadow Arena
            </a>
            <a href="https://github.com/maceip/RLMArena" class="btn btn-secondary">
              View Source
            </a>
          </div>
        </section>

        <section class="features" id="features">
          <div class="features-grid">
            <div class="feature-card">
              <div class="feature-icon">S</div>
              <h3>Shadow Execution</h3>
              <p>
                Run multiple model variations in parallel with RadixAttention caching.
                Compare outputs without impacting production latency.
              </p>
            </div>

            <div class="feature-card">
              <div class="feature-icon">V</div>
              <h3>Hard Verifiers</h3>
              <p>
                Cedar network policies, OPA infrastructure checks, and Firecracker
                sandboxes verify code execution with cryptographic proofs.
              </p>
            </div>

            <div class="feature-card">
              <div class="feature-icon">C</div>
              <h3>Certainty Certificates</h3>
              <p>
                Every verified response includes a certificate with execution proof,
                security audit results, and policy compliance attestation.
              </p>
            </div>

            <div class="feature-card">
              <div class="feature-icon">D</div>
              <h3>Continuous Distillation</h3>
              <p>
                DPO training pairs generated from shadow comparisons feed back into
                model improvement through LlamaFactory integration.
              </p>
            </div>

            <div class="feature-card">
              <div class="feature-icon">E</div>
              <h3>Expert Alignment</h3>
              <p>
                Capture SME decision patterns as trajectory evaluations. DSPy MIPROv2
                optimizes judges to match expert preferences.
              </p>
            </div>

            <div class="feature-card">
              <div class="feature-icon">T</div>
              <h3>Tailscale Mesh</h3>
              <p>
                Zero-trust networking between all services. Every container
                authenticated and encrypted without VPN configuration.
              </p>
            </div>
          </div>
        </section>

        <section class="features" id="how-it-works">
          <div class="section-header" style={{ textAlign: "center", marginBottom: "2rem" }}>
            <h2>How Shadow Arena Works</h2>
            <p>Parallel verification without production impact</p>
          </div>

          <div class="card" style={{ maxWidth: "800px", margin: "0 auto" }}>
            <div class="code-block">
{`1. Request arrives at VaaS Gateway
   └─► Rate limiting, auth, routing

2. Primary model generates response
   └─► Served immediately to user

3. Shadow loop spawns variations (async)
   ├─► Temperature 0.1 (deterministic)
   ├─► Temperature 0.7 (balanced)
   └─► Temperature 1.0 (creative)

4. Hard verifiers check each trajectory
   ├─► Leash: Network policy compliance
   ├─► OPA: Infrastructure policy check
   └─► Firecracker: Sandboxed execution

5. Comparisons generate training signal
   └─► DPO pairs stored for distillation

6. Certificate issued with proof chain
   └─► Attestation of verification steps`}
            </div>
          </div>
        </section>
      </main>

      <footer style={{ padding: "3rem 0", textAlign: "center", borderTop: "1px solid var(--border)", marginTop: "4rem" }}>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
          RLMArena - Open source verification infrastructure
        </p>
      </footer>
    </>
  );
});

export const head: DocumentHead = {
  title: "RLMArena - Verifier-as-a-Service",
  meta: [
    {
      name: "description",
      content: "Production-grade verification infrastructure for AI agents with parallel execution and formal proofs.",
    },
  ],
};
