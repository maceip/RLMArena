/**
 * Public Area Status Component
 *
 * Compact sparklines showing real-time system metrics.
 * Uses SVG for minimal bundle size, useVisibleTask$ for lazy updates.
 */

import { component$, useSignal, useVisibleTask$ } from "@builder.io/qwik";

export interface SparklineData {
  values: number[];
  label: string;
  unit?: string;
  color?: string;
}

/**
 * Sparkline - Minimal inline chart
 */
export const Sparkline = component$<{
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  showDot?: boolean;
}>(({ data, width = 60, height = 20, color = "var(--accent)", showDot = true }) => {
  if (data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x},${y}`;
  }).join(" ");

  const lastX = width;
  const lastY = height - ((data[data.length - 1] - min) / range) * height;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ display: "block" }}
    >
      <polyline
        points={points}
        fill="none"
        stroke={color}
        stroke-width="1.5"
        stroke-linecap="round"
        stroke-linejoin="round"
      />
      {showDot && (
        <circle
          cx={lastX}
          cy={lastY}
          r="2"
          fill={color}
        />
      )}
    </svg>
  );
});

/**
 * Status Metric - Label + value + sparkline
 */
export const StatusMetric = component$<{
  label: string;
  value: string | number;
  data: number[];
  unit?: string;
  color?: string;
}>(({ label, value, data, unit = "", color = "var(--accent)" }) => {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.75rem",
        padding: "0.5rem 0",
      }}
    >
      <div style={{ minWidth: "60px" }}>
        <Sparkline data={data} color={color} width={50} height={16} />
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: "0.25rem" }}>
        <span
          class="mono"
          style={{
            fontSize: "0.8125rem",
            fontWeight: 600,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {value}{unit}
        </span>
        <span
          class="mono-label"
          style={{
            fontSize: "0.625rem",
            color: "var(--text-secondary)",
          }}
        >
          {label}
        </span>
      </div>
    </div>
  );
});

/**
 * Public Area Status Bar
 *
 * Compact horizontal bar showing key system metrics with sparklines.
 * Self-updates via useVisibleTask$ - only runs when visible.
 */
export const PublicStatusBar = component$<{
  vertical?: boolean;
}>(({ vertical = false }) => {
  // Simulated real-time data - in production, fetch from /api/arena/stats
  const latencyData = useSignal<number[]>([245, 232, 251, 239, 244, 228, 255, 241, 237, 249]);
  const rpsData = useSignal<number[]>([42, 45, 38, 51, 47, 44, 49, 53, 46, 48]);
  const cacheData = useSignal<number[]>([71, 73, 72, 74, 73, 75, 74, 73, 76, 74]);
  const errorData = useSignal<number[]>([2, 1, 3, 1, 2, 1, 0, 2, 1, 1]);

  // Current values (last in array)
  const currentLatency = useSignal(249);
  const currentRps = useSignal(48);
  const currentCache = useSignal(74);
  const currentErrors = useSignal(1);

  // Animate data updates - lazy loaded, only when visible
  useVisibleTask$(({ cleanup }) => {
    const interval = setInterval(() => {
      // Shift and add new value
      const newLatency = 220 + Math.random() * 50;
      const newRps = 40 + Math.random() * 20;
      const newCache = 70 + Math.random() * 10;
      const newErrors = Math.random() > 0.8 ? Math.floor(Math.random() * 4) : 0;

      latencyData.value = [...latencyData.value.slice(1), newLatency];
      rpsData.value = [...rpsData.value.slice(1), newRps];
      cacheData.value = [...cacheData.value.slice(1), newCache];
      errorData.value = [...errorData.value.slice(1), newErrors];

      currentLatency.value = Math.round(newLatency);
      currentRps.value = Math.round(newRps);
      currentCache.value = Math.round(newCache);
      currentErrors.value = newErrors;
    }, 2000);

    cleanup(() => clearInterval(interval));
  });

  const containerStyle = vertical
    ? {
        display: "flex",
        flexDirection: "column" as const,
        gap: "0.25rem",
        padding: "0.75rem",
        background: "var(--bg-secondary)",
        border: "1px solid var(--border)",
      }
    : {
        display: "flex",
        alignItems: "center",
        gap: "1.5rem",
        padding: "0.5rem 1rem",
        background: "var(--bg-secondary)",
        border: "1px solid var(--border)",
        overflowX: "auto" as const,
      };

  return (
    <div style={containerStyle}>
      <StatusMetric
        label="latency"
        value={currentLatency.value}
        unit="ms"
        data={latencyData.value}
        color="var(--accent)"
      />
      <StatusMetric
        label="req/s"
        value={currentRps.value}
        data={rpsData.value}
        color="var(--success)"
      />
      <StatusMetric
        label="cache"
        value={currentCache.value}
        unit="%"
        data={cacheData.value}
        color="var(--accent)"
      />
      <StatusMetric
        label="errors"
        value={currentErrors.value}
        data={errorData.value}
        color={currentErrors.value > 2 ? "var(--error)" : "var(--text-secondary)"}
      />
    </div>
  );
});

/**
 * Minimal Status Dot - For nav bar integration
 */
export const StatusDot = component$<{
  status?: "ok" | "degraded" | "down";
}>(({ status = "ok" }) => {
  const colors = {
    ok: "var(--success)",
    degraded: "var(--warning)",
    down: "var(--error)",
  };

  return (
    <span
      style={{
        display: "inline-block",
        width: "6px",
        height: "6px",
        borderRadius: "50%",
        background: colors[status],
        boxShadow: `0 0 4px ${colors[status]}`,
      }}
    />
  );
});
