import { component$, Slot } from "@builder.io/qwik";
import { Link, useLocation } from "@builder.io/qwik-city";

export default component$(() => {
  const loc = useLocation();
  const path = loc.url.pathname;

  const isActive = (href: string) => {
    if (href === "/arena/" || href === "/arena") {
      return path === "/arena/" || path === "/arena";
    }
    return path.startsWith(href);
  };

  return (
    <>
      <nav class="nav">
        <div class="nav-inner">
          <Link href="/" class="nav-logo">
            RLM<span>Arena</span>
          </Link>
          <ul class="nav-links">
            <li><Link href="/">Home</Link></li>
            <li><Link href="/arena">Dashboard</Link></li>
            <li><a href="https://github.com/maceip/RLMArena">GitHub</a></li>
          </ul>
        </div>
      </nav>

      <div class="dashboard">
        <aside class="sidebar">
          <div class="sidebar-section">
            <h4>Shadow Arena</h4>
            <Link href="/arena" class={`sidebar-link ${isActive("/arena/") || path === "/arena" ? "active" : ""}`}>
              Overview
            </Link>
            <Link href="/arena/trajectories" class={`sidebar-link ${isActive("/arena/trajectories") ? "active" : ""}`}>
              Trajectories
            </Link>
            <Link href="/arena/comparisons" class={`sidebar-link ${isActive("/arena/comparisons") ? "active" : ""}`}>
              Comparisons
            </Link>
          </div>

          <div class="sidebar-section">
            <h4>Verification</h4>
            <Link href="/arena/certificates" class={`sidebar-link ${isActive("/arena/certificates") ? "active" : ""}`}>
              Certificates
            </Link>
            <Link href="/arena/verifiers" class={`sidebar-link ${isActive("/arena/verifiers") ? "active" : ""}`}>
              Verifiers
            </Link>
          </div>

          <div class="sidebar-section">
            <h4>Training</h4>
            <Link href="/arena/distillation" class={`sidebar-link ${isActive("/arena/distillation") ? "active" : ""}`}>
              Distillation
            </Link>
            <Link href="/arena/experts" class={`sidebar-link ${isActive("/arena/experts") ? "active" : ""}`}>
              Expert Alignment
            </Link>
          </div>

          <div class="sidebar-section">
            <h4>System</h4>
            <Link href="/arena/telemetry" class={`sidebar-link ${isActive("/arena/telemetry") ? "active" : ""}`}>
              Telemetry
            </Link>
            <Link href="/arena/costs" class={`sidebar-link ${isActive("/arena/costs") ? "active" : ""}`}>
              Usage & Costs
            </Link>
          </div>
        </aside>

        <main class="main-content">
          <Slot />
        </main>
      </div>
    </>
  );
});
