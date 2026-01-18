/**
 * Arena Stats API
 * GET /api/arena/stats - Returns shadow arena statistics
 */

import type { RequestHandler } from "@builder.io/qwik-city";

const GATEWAY_ENDPOINT = process.env.GATEWAY_ENDPOINT || "http://localhost:8080";

export const onGet: RequestHandler = async ({ json }) => {
  try {
    // TODO: Proxy to gateway when available
    // const response = await fetch(`${GATEWAY_ENDPOINT}/stats`);
    // if (!response.ok) throw new Error(`Gateway error: ${response.status}`);
    // const data = await response.json();
    // json(200, data);

    // Mock response for development
    json(200, {
      totalTrajectories: 12847,
      certificatesIssued: 11203,
      verificationRate: 87.2,
      dpoTripletsGenerated: 3421,
      activeVerifiers: 4,
      shadowLoopVariations: 3,
      cacheHitRate: 73.5,
      averageLatencyMs: 245,
      updatedAt: new Date().toISOString(),
    });
  } catch (error) {
    json(503, {
      error: "Gateway unavailable",
      message: error instanceof Error ? error.message : "Unknown error",
      gatewayEndpoint: GATEWAY_ENDPOINT,
    });
  }
};
