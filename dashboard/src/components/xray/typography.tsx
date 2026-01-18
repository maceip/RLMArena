/**
 * Typography of Urgency
 *
 * Aggressive, high-stakes typography for the LLM frontier.
 * Variable weight fonts at maximum wght, intersection effects.
 */

import { component$, Slot } from "@builder.io/qwik";

export interface MassiveHeadlineProps {
  intersect?: "top" | "bottom" | "none";
  class?: string;
}

/**
 * Massive Headline - Maximum weight variable font
 * Use intersect prop to create text that clips behind containers
 */
export const MassiveHeadline = component$<MassiveHeadlineProps>(({
  intersect = "none",
  class: className,
}) => {
  const intersectClass =
    intersect === "top"
      ? "text-intersect-top"
      : intersect === "bottom"
        ? "text-intersect-bottom"
        : "";

  return (
    <h1
      class={["headline-massive", intersectClass, className].filter(Boolean).join(" ")}
      style={{
        fontVariationSettings: "'wght' 900",
      }}
    >
      <Slot />
    </h1>
  );
});

/**
 * Skeleton Card - Mid-century professional aesthetic
 * The classic advertisement/announcement card style
 */
export const SkeletonCard = component$<{
  dark?: boolean;
  class?: string;
}>(({ dark = false, class: className }) => {
  return (
    <div
      class={[
        "skeleton-card",
        dark && "skeleton-card-dark",
        className,
      ].filter(Boolean).join(" ")}
    >
      <Slot />
    </div>
  );
});

/**
 * Justified Text Block - Legal/technical document style
 * Perfect text justification with proper typography
 */
export const JustifiedBlock = component$<{
  class?: string;
}>(({ class: className }) => {
  return (
    <div class={["justified", className].filter(Boolean).join(" ")}>
      <Slot />
    </div>
  );
});

/**
 * Mono Label - Technical status labels and buttons
 * Pixelated/typewriter style for arena grid system
 */
export const MonoLabel = component$<{
  as?: "span" | "div" | "p";
  size?: "sm" | "md" | "lg";
  class?: string;
}>(({ as = "span", size = "md", class: className }) => {
  const Tag = as;
  const sizeStyles = {
    sm: { fontSize: "0.6875rem", letterSpacing: "0.1em" },
    md: { fontSize: "0.75rem", letterSpacing: "0.1em" },
    lg: { fontSize: "1.5rem", letterSpacing: "0.15em" },
  };

  return (
    <Tag
      class={["mono-label", className].filter(Boolean).join(" ")}
      style={sizeStyles[size]}
    >
      <Slot />
    </Tag>
  );
});

/**
 * Arena Logo - The RLMArena wordmark
 */
export const ArenaLogo = component$<{
  size?: "sm" | "md" | "lg";
}>(({ size = "md" }) => {
  const sizes = {
    sm: "1rem",
    md: "1.5rem",
    lg: "2rem",
  };

  return (
    <span
      class="mono"
      style={{
        fontSize: sizes[size],
        fontWeight: 700,
        letterSpacing: "0.15em",
        textTransform: "uppercase",
      }}
    >
      RLM<span style={{ color: "var(--accent)" }}>Arena</span>
    </span>
  );
});
