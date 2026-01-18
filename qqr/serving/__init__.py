"""
VaaS Serving Infrastructure.

Production serving components for the Verifier-as-a-Service platform:
- VaaS Router for request routing
- Edge Verifier for low-latency local verification
- Telemetry Collector for audit logging
- Cost Tracker for API accounting
- ShadowArena Proxy for OpenAI-compatible API interception
"""

from qqr.serving.router import VaaSRouter, RouterConfig
from qqr.serving.edge import EdgeVerifier, EdgeConfig
from qqr.serving.telemetry import TelemetryCollector, TelemetryEvent
from qqr.serving.costs import CostTracker, UsageRecord
from qqr.serving.proxy import (
    ProxyConfig,
    ProxyMetrics,
    ProxyMode,
    ResponseCache,
    RateLimiter,
    ShadowArenaProxy,
    create_app,
)

__all__ = [
    "VaaSRouter",
    "RouterConfig",
    "EdgeVerifier",
    "EdgeConfig",
    "TelemetryCollector",
    "TelemetryEvent",
    "CostTracker",
    "UsageRecord",
    "ProxyConfig",
    "ProxyMetrics",
    "ProxyMode",
    "ResponseCache",
    "RateLimiter",
    "ShadowArenaProxy",
    "create_app",
]
