/**
 * Trajectories API
 * GET /api/arena/trajectories - List trajectories
 * POST /api/arena/trajectories - Submit new trajectory for verification
 */

import type { RequestHandler } from "@builder.io/qwik-city";

const GATEWAY_ENDPOINT = process.env.GATEWAY_ENDPOINT || "http://localhost:8080";

export const onGet: RequestHandler = async ({ json, query }) => {
  const limit = parseInt(query.get("limit") || "20");
  const offset = parseInt(query.get("offset") || "0");
  const status = query.get("status"); // verified, pending, failed

  try {
    // TODO: Proxy to gateway
    // const url = new URL(`${GATEWAY_ENDPOINT}/trajectories`);
    // url.searchParams.set("limit", limit.toString());
    // url.searchParams.set("offset", offset.toString());
    // if (status) url.searchParams.set("status", status);
    // const response = await fetch(url);
    // const data = await response.json();

    // Mock response
    const mockTrajectories = [
      {
        id: "traj-001",
        status: "verified",
        model: "llama-3.1-70b",
        createdAt: new Date(Date.now() - 2 * 60000).toISOString(),
        score: 0.94,
        certificateId: "cert-001",
        messageCount: 4,
      },
      {
        id: "traj-002",
        status: "pending",
        model: "llama-3.1-70b",
        createdAt: new Date(Date.now() - 5 * 60000).toISOString(),
        score: null,
        certificateId: null,
        messageCount: 2,
      },
      {
        id: "traj-003",
        status: "failed",
        model: "llama-3.1-8b",
        createdAt: new Date(Date.now() - 8 * 60000).toISOString(),
        score: 0.31,
        certificateId: null,
        messageCount: 6,
        failureReason: "Network policy violation",
      },
      {
        id: "traj-004",
        status: "verified",
        model: "llama-3.1-70b",
        createdAt: new Date(Date.now() - 12 * 60000).toISOString(),
        score: 0.89,
        certificateId: "cert-002",
        messageCount: 3,
      },
      {
        id: "traj-005",
        status: "verified",
        model: "mistral-7b",
        createdAt: new Date(Date.now() - 15 * 60000).toISOString(),
        score: 0.76,
        certificateId: "cert-003",
        messageCount: 5,
      },
    ];

    const filtered = status
      ? mockTrajectories.filter((t) => t.status === status)
      : mockTrajectories;

    json(200, {
      trajectories: filtered.slice(offset, offset + limit),
      total: filtered.length,
      limit,
      offset,
    });
  } catch (error) {
    json(503, {
      error: "Gateway unavailable",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
};

export const onPost: RequestHandler = async ({ json, parseBody }) => {
  try {
    const body = await parseBody();

    // TODO: Proxy to gateway
    // const response = await fetch(`${GATEWAY_ENDPOINT}/trajectories`, {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify(body),
    // });
    // const data = await response.json();

    // Mock response
    const trajectoryId = `traj-${Date.now().toString(36)}`;

    json(202, {
      trajectoryId,
      status: "pending",
      message: "Trajectory submitted for verification",
      estimatedCompletionMs: 5000,
    });
  } catch (error) {
    json(500, {
      error: "Failed to submit trajectory",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
};
