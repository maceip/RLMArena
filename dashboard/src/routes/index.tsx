/**
 * RLMArena Landing Page
 *
 * X-Ray aesthetic with ATTACK/ALIGN typography intersection.
 * Optimized for speculative module fetching - all handlers use $
 */

import { component$, useSignal, $ } from "@builder.io/qwik";
import { type DocumentHead, Link } from "@builder.io/qwik-city";
import {
  MassiveHeadline,
  SkeletonCard,
  JustifiedBlock,
  MonoLabel,
  ArenaLogo,
  XRayContainer,
  CircuitPattern,
  PublicStatusBar,
} from "~/components/xray";

export default component$(() => {
  const hovered = useSignal(false);

  // Lazy-loaded hover handler
  const onMouseEnter$ = $(() => {
    hovered.value = true;
  });

  const onMouseLeave$ = $(() => {
    hovered.value = false;
  });

  return (
    <>
      {/* Minimal Navigation */}
      <nav class="nav">
        <div class="nav-inner">
          <Link href="/" class="nav-logo">
            RLM<span>Arena</span>
          </Link>
          <ul class="nav-links">
            <li><a href="#manifesto">Manifesto</a></li>
            <li><Link href="/arena">Shadow Arena</Link></li>
            <li><a href="https://github.com/maceip/RLMArena">Source</a></li>
          </ul>
        </div>
      </nav>

      {/* Hero - The Skeleton Card */}
      <main class="hero">
        <SkeletonCard>
          {/* Top Headline - Intersects with X-Ray container */}
          <MassiveHeadline intersect="top">
            Attack
          </MassiveHeadline>

          {/* X-Ray Container - The Hollow Void */}
          <XRayContainer
            class="xray-hero"
          >
            <CircuitPattern density={30} />
            <div
              style={{
                padding: "5rem 2rem",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
                zIndex: 1,
              }}
            >
              {/* Abstract verification diagram */}
              <svg
                width="200"
                height="120"
                viewBox="0 0 200 120"
                style={{
                  filter: "invert(1) brightness(1.3) drop-shadow(0 0 15px rgba(99, 102, 241, 0.6))",
                }}
              >
                {/* Input node */}
                <circle cx="30" cy="60" r="8" fill="none" stroke="white" stroke-width="1.5" />
                <text x="30" y="85" fill="white" font-size="8" text-anchor="middle" font-family="monospace">IN</text>

                {/* Verification nodes */}
                <rect x="70" y="20" width="60" height="25" fill="none" stroke="white" stroke-width="1" />
                <text x="100" y="37" fill="white" font-size="7" text-anchor="middle" font-family="monospace">LEASH</text>

                <rect x="70" y="50" width="60" height="25" fill="none" stroke="white" stroke-width="1" />
                <text x="100" y="67" fill="white" font-size="7" text-anchor="middle" font-family="monospace">OPA</text>

                <rect x="70" y="80" width="60" height="25" fill="none" stroke="white" stroke-width="1" />
                <text x="100" y="97" fill="white" font-size="7" text-anchor="middle" font-family="monospace">SANDBOX</text>

                {/* Output node */}
                <circle cx="170" cy="60" r="8" fill="white" stroke="white" stroke-width="1.5" />
                <text x="170" y="85" fill="white" font-size="8" text-anchor="middle" font-family="monospace">CERT</text>

                {/* Connections */}
                <path d="M38 60 L70 32" stroke="white" stroke-width="1" fill="none" />
                <path d="M38 60 L70 62" stroke="white" stroke-width="1" fill="none" />
                <path d="M38 60 L70 92" stroke="white" stroke-width="1" fill="none" />
                <path d="M130 32 L162 60" stroke="white" stroke-width="1" fill="none" />
                <path d="M130 62 L162 60" stroke="white" stroke-width="1" fill="none" />
                <path d="M130 92 L162 60" stroke="white" stroke-width="1" fill="none" />
              </svg>
            </div>
          </XRayContainer>

          {/* Bottom Headline - Intersects with X-Ray container */}
          <MassiveHeadline intersect="bottom">
            Align
          </MassiveHeadline>

          {/* Manifesto Text */}
          <div style={{ marginTop: "2.5rem", padding: "0 1rem" }} id="manifesto">
            <JustifiedBlock>
              <p>
                Data is compute. The compute you don't spend on static judges is compute saved
                for verification that matters. Shadow execution runs parallel trajectories through
                hard verifiers—Cedar network policies, Rego compliance checks, Firecracker sandboxes—producing
                cryptographic certainty certificates for every response.
              </p>
              <p style={{ marginTop: "1.5rem" }}>
                The distillation loop captures expert preferences as DPO training pairs.
                Each shadow comparison improves the next generation. No human labels required.
                The arena selects. The model aligns.
              </p>
              <p style={{ marginTop: "1.5rem", fontWeight: 700 }}>
                Enter the RLMArena today.
              </p>
            </JustifiedBlock>
          </div>

          {/* CTA Buttons */}
          <div style={{ marginTop: "2.5rem", display: "flex", gap: "1rem", justifyContent: "center" }}>
            <Link
              href="/arena"
              class="btn btn-primary"
              onMouseEnter$={onMouseEnter$}
              onMouseLeave$={onMouseLeave$}
            >
              Shadow Arena
            </Link>
            <a href="https://github.com/maceip/RLMArena" class="btn btn-secondary">
              View Source
            </a>
          </div>

          {/* Logo Footer */}
          <div style={{ marginTop: "3rem", textAlign: "right" }}>
            <ArenaLogo size="lg" />
          </div>
        </SkeletonCard>
      </main>

      {/* Features Grid */}
      <section style={{ padding: "6rem 2rem", background: "var(--bg-primary)" }}>
        <div class="container">
          <div style={{ textAlign: "center", marginBottom: "3rem" }}>
            <MonoLabel size="sm" as="div" style={{ color: "var(--text-secondary)", marginBottom: "0.5rem" }}>
              Verification Infrastructure
            </MonoLabel>
            <h2 style={{ fontSize: "2rem", fontWeight: 700, color: "var(--text-primary)" }}>
              Hard Proofs, Not Soft Scores
            </h2>
          </div>

          <div class="features-grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1.5rem" }}>
            <FeatureCard
              icon="S"
              title="Shadow Execution"
              description="RadixAttention caching enables parallel trajectory generation without latency impact. Compare N variations per request."
            />
            <FeatureCard
              icon="V"
              title="Hard Verifiers"
              description="Cedar policies enforce network boundaries. OPA checks infrastructure compliance. Firecracker sandboxes execute untrusted code."
            />
            <FeatureCard
              icon="C"
              title="Certainty Certificates"
              description="Every verified response includes a certificate with execution proof, security audit hash, and policy attestation chain."
            />
            <FeatureCard
              icon="D"
              title="Continuous Distillation"
              description="DPO pairs generated from shadow comparisons flow into LlamaFactory for continuous model improvement."
            />
            <FeatureCard
              icon="E"
              title="Expert Alignment"
              description="Capture SME decision patterns as trajectory scores. DSPy MIPROv2 optimizes composite judges to match expert preferences."
            />
            <FeatureCard
              icon="T"
              title="Zero-Trust Mesh"
              description="Tailscale authenticates every container. No VPN configuration. Encrypted by default. Edge-ready deployment."
            />
          </div>
        </div>
      </section>

      {/* Public Status Bar */}
      <div class="container" style={{ padding: "1.5rem 2rem", background: "var(--bg-primary)" }}>
        <PublicStatusBar />
      </div>

      {/* Footer */}
      <footer style={{ padding: "2rem", textAlign: "center", background: "var(--bg-secondary)", borderTop: "1px solid var(--border)" }}>
        <MonoLabel size="sm" as="p" style={{ color: "var(--text-secondary)" }}>
          RLMArena — Verifier-as-a-Service
        </MonoLabel>
      </footer>
    </>
  );
});

/**
 * Feature Card Component
 * Lazy-loaded via component$ for optimal chunking
 */
const FeatureCard = component$<{
  icon: string;
  title: string;
  description: string;
}>(({ icon, title, description }) => {
  return (
    <div
      class="card"
      style={{
        background: "var(--bg-secondary)",
        border: "1px solid var(--border)",
        padding: "1.5rem",
      }}
    >
      <div
        style={{
          width: "2.5rem",
          height: "2.5rem",
          background: "var(--bg-tertiary)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: "1rem",
          fontFamily: "'JetBrains Mono', monospace",
          fontWeight: 700,
          color: "var(--accent)",
        }}
      >
        {icon}
      </div>
      <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem", color: "var(--text-primary)" }}>
        {title}
      </h3>
      <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
        {description}
      </p>
    </div>
  );
});

export const head: DocumentHead = {
  title: "RLMArena — Attack / Align",
  meta: [
    {
      name: "description",
      content: "Verifier-as-a-Service for AI agents. Shadow execution, hard proofs, continuous distillation.",
    },
  ],
};
