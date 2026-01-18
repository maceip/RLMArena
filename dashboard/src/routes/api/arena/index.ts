/**
 * Shadow Arena API Proxy Routes
 *
 * These routes will proxy to the VaaS gateway backend.
 * Currently returns mock data for development.
 */

import type { RequestHandler } from "@builder.io/qwik-city";

// Backend endpoint - configure via environment
const GATEWAY_ENDPOINT = process.env.GATEWAY_ENDPOINT || "http://localhost:8080";

export const onGet: RequestHandler = async ({ json }) => {
  // TODO: Proxy to actual gateway
  // const response = await fetch(`${GATEWAY_ENDPOINT}/api/arena/stats`);
  // const data = await response.json();

  json(200, {
    status: "ok",
    endpoint: GATEWAY_ENDPOINT,
    routes: {
      stats: "/api/arena/stats",
      trajectories: "/api/arena/trajectories",
      comparisons: "/api/arena/comparisons",
      certificates: "/api/arena/certificates",
      verifiers: "/api/arena/verifiers",
      distillation: "/api/arena/distillation",
    },
  });
};
