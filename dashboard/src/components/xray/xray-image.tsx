/**
 * X-Ray Image Component
 *
 * Creates the "illuminated wireframe" effect by inverting dark objects
 * into glowing white circuitry against a black void.
 *
 * CSS Filter Stack: invert(1) brightness(1.2) contrast(1.1)
 */

import { component$, useSignal, useVisibleTask$, type QRL } from "@builder.io/qwik";

export interface XRayImageProps {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  animate?: boolean;
  onLoad$?: QRL<() => void>;
}

export const XRayImage = component$<XRayImageProps>(({
  src,
  alt,
  width,
  height,
  animate = true,
  onLoad$,
}) => {
  const loaded = useSignal(false);
  const intensity = useSignal(1.2);

  // Pulse animation via signal updates - lazy loaded
  useVisibleTask$(({ track, cleanup }) => {
    track(() => animate);

    if (!animate) return;

    let frame: number;
    let direction = 1;

    const pulse = () => {
      intensity.value += direction * 0.01;
      if (intensity.value >= 1.5) direction = -1;
      if (intensity.value <= 1.2) direction = 1;
      frame = requestAnimationFrame(pulse);
    };

    frame = requestAnimationFrame(pulse);
    cleanup(() => cancelAnimationFrame(frame));
  });

  return (
    <div
      class="xray-container"
      style={{
        padding: "4rem 2rem",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <img
        src={src}
        alt={alt}
        width={width}
        height={height}
        onLoad$={() => {
          loaded.value = true;
          onLoad$?.();
        }}
        style={{
          filter: `invert(1) brightness(${intensity.value}) contrast(1.1) drop-shadow(0 0 ${animate ? "20px" : "10px"} rgba(99, 102, 241, 0.6))`,
          mixBlendMode: "screen",
          opacity: loaded.value ? 0.9 : 0,
          transition: "opacity 0.3s ease",
          maxWidth: "100%",
          height: "auto",
        }}
      />
    </div>
  );
});

/**
 * X-Ray Container - Hollow container with edge bleed
 * Use this to wrap content that should appear in the "void"
 */
export const XRayContainer = component$<{
  children?: any;
  class?: string;
}>(({ class: className }) => {
  return (
    <div
      class={["xray-container", className].filter(Boolean).join(" ")}
      style={{
        background: "#000",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <slot />
    </div>
  );
});

/**
 * Circuit Pattern SVG - Procedurally generated circuit lines
 * Lazy-loaded and only rendered when visible
 */
export const CircuitPattern = component$<{
  density?: number;
  color?: string;
}>(({ density = 20, color = "rgba(99, 102, 241, 0.3)" }) => {
  const lines = useSignal<Array<{ x1: number; y1: number; x2: number; y2: number }>>([]);

  useVisibleTask$(() => {
    // Generate random circuit-like lines
    const generated: typeof lines.value = [];
    for (let i = 0; i < density; i++) {
      const x1 = Math.random() * 100;
      const y1 = Math.random() * 100;
      // Lines go either horizontal or vertical
      const horizontal = Math.random() > 0.5;
      const length = 10 + Math.random() * 30;
      generated.push({
        x1,
        y1,
        x2: horizontal ? x1 + length : x1,
        y2: horizontal ? y1 : y1 + length,
      });
    }
    lines.value = generated;
  });

  return (
    <svg
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
      preserveAspectRatio="none"
    >
      {lines.value.map((line, i) => (
        <line
          key={i}
          x1={`${line.x1}%`}
          y1={`${line.y1}%`}
          x2={`${line.x2}%`}
          y2={`${line.y2}%`}
          stroke={color}
          stroke-width="1"
          opacity="0.5"
        />
      ))}
    </svg>
  );
});
