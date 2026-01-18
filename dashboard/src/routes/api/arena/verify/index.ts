/**
 * Verify API
 * POST /api/arena/verify - Submit messages for shadow verification
 *
 * This is the main entry point for the shadow arena.
 * Accepts a conversation and returns verification results.
 */

import type { RequestHandler } from "@builder.io/qwik-city";

const GATEWAY_ENDPOINT = process.env.GATEWAY_ENDPOINT || "http://localhost:8080";

interface VerifyRequest {
  messages: Array<{
    role: "user" | "assistant" | "system";
    content: string;
  }>;
  model?: string;
  shadowConfig?: {
    variations?: number;
    temperatures?: number[];
    enableVerifiers?: string[];
  };
}

export const onPost: RequestHandler = async ({ json, parseBody }) => {
  try {
    const body = (await parseBody()) as VerifyRequest;

    if (!body.messages || !Array.isArray(body.messages)) {
      json(400, { error: "messages array is required" });
      return;
    }

    // TODO: Proxy to gateway
    // const response = await fetch(`${GATEWAY_ENDPOINT}/verify`, {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify(body),
    // });
    // const data = await response.json();

    // Mock shadow verification response
    const trajectoryId = `traj-${Date.now().toString(36)}`;
    const now = new Date();

    json(200, {
      trajectoryId,
      status: "verified",
      certificate: {
        certificateId: `cert-${Date.now().toString(36)}`,
        status: "valid",
        issuedAt: now.toISOString(),
        expiresAt: new Date(now.getTime() + 24 * 60 * 60 * 1000).toISOString(),
        verificationResults: {
          leash: { passed: true, policyChecks: 3 },
          opa: { passed: true, policyChecks: 5 },
          execution: { passed: true, sandboxed: true },
        },
      },
      shadowResults: [
        {
          variation: 0,
          temperature: 0.1,
          score: 0.92,
          latencyMs: 234,
        },
        {
          variation: 1,
          temperature: 0.7,
          score: 0.88,
          latencyMs: 245,
        },
        {
          variation: 2,
          temperature: 1.0,
          score: 0.79,
          latencyMs: 267,
        },
      ],
      primaryResponse: {
        model: body.model || "llama-3.1-70b",
        content: "This is a mock response. Connect to the gateway for real verification.",
        latencyMs: 234,
      },
      metadata: {
        inputTokens: body.messages.reduce((acc, m) => acc + m.content.length / 4, 0),
        outputTokens: 50,
        totalLatencyMs: 312,
      },
    });
  } catch (error) {
    json(500, {
      error: "Verification failed",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
};
