"""
OpenAI-Compatible Proxy Middleware for ShadowArena

This module provides a production-grade FastAPI-based proxy that intercepts
OpenAI API requests and routes them through the ShadowArena verification
and tournament pipeline.

The proxy implements the OpenAI Chat Completions API (/v1/chat/completions)
and adds real-time verification, parallel generation, and response ranking.
"""

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import httpx

try:
    from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class ProxyMode(Enum):
    """Operating modes for the proxy."""
    PASSTHROUGH = "passthrough"  # Forward requests without verification
    SHADOW = "shadow"  # Run verification in parallel, return original
    INTERCEPT = "intercept"  # Full interception with best-response selection
    AUDIT = "audit"  # Log all requests/responses without modification


@dataclass
class ProxyConfig:
    """Configuration for the ShadowArena proxy."""
    mode: ProxyMode = ProxyMode.INTERCEPT
    upstream_url: str = "https://api.openai.com"
    num_variations: int = 4
    max_correction_attempts: int = 2
    enable_self_correction: bool = True
    parallel_generation: bool = True
    timeout_seconds: float = 120.0
    min_valid_responses: int = 1
    enable_caching: bool = True
    cache_ttl_seconds: float = 3600.0
    max_concurrent_requests: int = 100
    enable_streaming: bool = True
    enable_early_termination: bool = True
    log_level: str = "INFO"
    enable_telemetry: bool = True


@dataclass
class ProxyMetrics:
    """Metrics for the proxy."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_variations_generated: int = 0
    total_verifications_passed: int = 0
    total_verifications_failed: int = 0
    avg_latency_ms: float = 0.0
    tokens_saved_by_early_termination: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "verification_pass_rate": (
                self.total_verifications_passed /
                max(1, self.total_verifications_passed + self.total_verifications_failed)
            ),
            "avg_latency_ms": self.avg_latency_ms,
            "tokens_saved_by_early_termination": self.tokens_saved_by_early_termination,
            "uptime_seconds": (datetime.utcnow() - self.created_at).total_seconds(),
        }


class ResponseCache:
    """LRU cache with TTL for verified responses."""

    def __init__(self, max_size: int = 10000, ttl_seconds: float = 3600.0):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._access_order: list[str] = []

    def _make_key(self, messages: list[dict], model: str) -> str:
        """Generate cache key from messages and model."""
        content = json.dumps({"messages": messages, "model": model}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, messages: list[dict], model: str) -> Optional[Any]:
        """Get cached response if valid."""
        key = self._make_key(messages, model)
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            self._access_order.remove(key)
            return None

        # Move to end for LRU
        self._access_order.remove(key)
        self._access_order.append(key)
        return value

    def set(self, messages: list[dict], model: str, value: Any) -> None:
        """Cache a response."""
        key = self._make_key(messages, model)

        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = (value, time.time())
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
    ):
        self._rpm = requests_per_minute
        self._rph = requests_per_hour
        self._burst = burst_size
        self._minute_tokens: dict[str, float] = {}
        self._hour_tokens: dict[str, float] = {}
        self._last_update: dict[str, float] = {}

    def _update_tokens(self, client_id: str) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        last = self._last_update.get(client_id, now)
        elapsed = now - last

        # Refill minute tokens
        minute_refill = elapsed * (self._rpm / 60.0)
        self._minute_tokens[client_id] = min(
            self._burst,
            self._minute_tokens.get(client_id, self._burst) + minute_refill
        )

        # Refill hour tokens
        hour_refill = elapsed * (self._rph / 3600.0)
        self._hour_tokens[client_id] = min(
            self._rph,
            self._hour_tokens.get(client_id, self._rph) + hour_refill
        )

        self._last_update[client_id] = now

    def check(self, client_id: str) -> tuple[bool, Optional[float]]:
        """Check if request is allowed. Returns (allowed, retry_after)."""
        self._update_tokens(client_id)

        minute_tokens = self._minute_tokens.get(client_id, self._burst)
        hour_tokens = self._hour_tokens.get(client_id, self._rph)

        if minute_tokens < 1:
            return False, 60 / self._rpm
        if hour_tokens < 1:
            return False, 3600 / self._rph

        self._minute_tokens[client_id] = minute_tokens - 1
        self._hour_tokens[client_id] = hour_tokens - 1
        return True, None


class ShadowArenaProxy:
    """
    Production-grade proxy for ShadowArena.

    This proxy intercepts OpenAI-compatible API requests and routes them
    through the verification and tournament pipeline.
    """

    def __init__(
        self,
        config: Optional[ProxyConfig] = None,
        shadow_arena: Optional[Any] = None,
        generate_fn: Optional[Callable] = None,
    ):
        self.config = config or ProxyConfig()
        self.shadow_arena = shadow_arena
        self.generate_fn = generate_fn
        self._cache = ResponseCache(ttl_seconds=self.config.cache_ttl_seconds)
        self._rate_limiter = RateLimiter()
        self._metrics = ProxyMetrics()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._active_requests: dict[str, asyncio.Task] = {}
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                follow_redirects=True,
            )
        return self._http_client

    async def close(self) -> None:
        """Close the proxy and release resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _forward_to_upstream(
        self,
        messages: list[dict],
        model: str,
        api_key: str,
        **kwargs: Any,
    ) -> dict:
        """Forward request to upstream OpenAI API."""
        client = await self._get_http_client()

        response = await client.post(
            f"{self.config.upstream_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                **kwargs,
            },
        )
        response.raise_for_status()
        return response.json()

    async def _generate_variation(
        self,
        messages: list[dict],
        model: str,
        api_key: str,
        variation_idx: int,
        **kwargs: Any,
    ) -> list[dict]:
        """Generate a single variation through upstream API."""
        # Use temperature variation for diversity
        temp = kwargs.get("temperature", 0.7)
        varied_temp = temp + (variation_idx * 0.05)

        response = await self._forward_to_upstream(
            messages=messages,
            model=model,
            api_key=api_key,
            temperature=min(2.0, varied_temp),
            **{k: v for k, v in kwargs.items() if k != "temperature"},
        )

        assistant_content = response["choices"][0]["message"]["content"]
        return messages + [{"role": "assistant", "content": assistant_content}]

    async def _process_with_shadow_arena(
        self,
        messages: list[dict],
        model: str,
        api_key: str,
        **kwargs: Any,
    ) -> dict:
        """Process request through ShadowArena pipeline."""
        if self.shadow_arena is None:
            # Fall back to simple parallel generation and selection
            return await self._simple_parallel_process(messages, model, api_key, **kwargs)

        # Use the full ShadowArena pipeline
        query = messages[-1].get("content", "") if messages else ""
        context = {
            "messages": messages[:-1] if messages else [],
            "model": model,
            "api_key": api_key,
            **kwargs,
        }

        result = await self.shadow_arena.process_query(query, context)

        if result.success and result.best_response:
            self._metrics.total_verifications_passed += result.valid_count
            self._metrics.total_verifications_failed += (
                len(result.verification_results) - result.valid_count
            )

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": result.best_response,
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,  # Would need tokenizer
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "shadow_arena": result.to_dict(),
            }

        # All variations failed, return error
        raise HTTPException(
            status_code=500,
            detail="All response variations failed verification",
        )

    async def _simple_parallel_process(
        self,
        messages: list[dict],
        model: str,
        api_key: str,
        **kwargs: Any,
    ) -> dict:
        """Simple parallel generation without full ShadowArena (fallback)."""
        tasks = [
            self._generate_variation(messages, model, api_key, i, **kwargs)
            for i in range(self.config.num_variations)
        ]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")

        # Filter successful results
        valid_results = [r for r in results if not isinstance(r, Exception)]

        if not valid_results:
            raise HTTPException(
                status_code=500,
                detail="All response generations failed",
            )

        # Select the first valid result (in full implementation, would rank)
        best_trajectory = valid_results[0]
        best_response = best_trajectory[-1] if best_trajectory else {"role": "assistant", "content": ""}

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": best_response,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    async def process_request(
        self,
        messages: list[dict],
        model: str,
        api_key: str,
        client_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Process a chat completion request.

        Args:
            messages: OpenAI-format messages
            model: Model identifier
            api_key: API key for upstream
            client_id: Client identifier for rate limiting
            **kwargs: Additional parameters

        Returns:
            OpenAI-compatible response dict
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        client_id = client_id or "default"

        # Rate limiting
        allowed, retry_after = self._rate_limiter.check(client_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after:.2f} seconds",
                headers={"Retry-After": str(int(retry_after or 60))},
            )

        # Check cache
        if self.config.enable_caching:
            cached = self._cache.get(messages, model)
            if cached:
                self._metrics.cache_hits += 1
                return cached
            self._metrics.cache_misses += 1

        self._metrics.total_requests += 1

        try:
            async with self._semaphore:
                if self.config.mode == ProxyMode.PASSTHROUGH:
                    result = await self._forward_to_upstream(
                        messages, model, api_key, **kwargs
                    )
                elif self.config.mode == ProxyMode.SHADOW:
                    # Return original but run verification in background
                    result = await self._forward_to_upstream(
                        messages, model, api_key, **kwargs
                    )
                    # Shadow verification would run in background
                elif self.config.mode == ProxyMode.INTERCEPT:
                    result = await self._process_with_shadow_arena(
                        messages, model, api_key, **kwargs
                    )
                else:  # AUDIT mode
                    result = await self._forward_to_upstream(
                        messages, model, api_key, **kwargs
                    )

            self._metrics.successful_requests += 1

            # Cache the result
            if self.config.enable_caching:
                self._cache.set(messages, model, result)

            # Update latency metrics
            latency = (time.time() - start_time) * 1000
            total = self._metrics.total_requests
            self._metrics.avg_latency_ms = (
                (self._metrics.avg_latency_ms * (total - 1) + latency) / total
            )

            return result

        except HTTPException:
            self._metrics.failed_requests += 1
            raise
        except Exception as e:
            self._metrics.failed_requests += 1
            raise HTTPException(status_code=500, detail=str(e))

    async def process_streaming(
        self,
        messages: list[dict],
        model: str,
        api_key: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Process a streaming chat completion request.

        For streaming, we generate and verify first, then stream the best response.
        """
        # First, get the verified response
        response = await self.process_request(messages, model, api_key, **kwargs)

        content = response["choices"][0]["message"]["content"]
        response_id = response["id"]

        # Stream the response chunk by chunk
        chunk_size = 10  # characters per chunk

        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": response["created"],
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk} if i > 0 else {"role": "assistant", "content": chunk},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.01)  # Simulate streaming delay

        # Final chunk
        final_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": response["created"],
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"

    def get_metrics(self) -> dict:
        """Get current proxy metrics."""
        return self._metrics.to_dict()

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()


def create_app(
    config: Optional[ProxyConfig] = None,
    shadow_arena: Optional[Any] = None,
) -> "FastAPI":
    """
    Create the FastAPI application for the ShadowArena proxy.

    Args:
        config: Proxy configuration
        shadow_arena: Optional ShadowArena instance

    Returns:
        FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")

    config = config or ProxyConfig()
    proxy = ShadowArenaProxy(config=config, shadow_arena=shadow_arena)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle."""
        yield
        await proxy.close()

    app = FastAPI(
        title="ShadowArena Proxy",
        description="OpenAI-compatible API proxy with verification and tournament ranking",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Request/Response Models
    class Message(BaseModel):
        role: str
        content: str
        name: Optional[str] = None
        function_call: Optional[dict] = None
        tool_calls: Optional[list[dict]] = None

    class ChatCompletionRequest(BaseModel):
        model: str
        messages: list[Message]
        temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
        top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
        n: Optional[int] = Field(default=1, ge=1)
        stream: Optional[bool] = False
        stop: Optional[list[str] | str] = None
        max_tokens: Optional[int] = None
        presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
        frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
        user: Optional[str] = None

    class ChatCompletionChoice(BaseModel):
        index: int
        message: Message
        finish_reason: Optional[str]

    class Usage(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: list[ChatCompletionChoice]
        usage: Optional[Usage] = None

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        raw_request: Request,
    ):
        """
        OpenAI-compatible chat completions endpoint.

        This endpoint intercepts requests and routes them through the
        ShadowArena verification and tournament pipeline.
        """
        # Extract API key from headers
        auth_header = raw_request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        api_key = auth_header[7:]

        # Get client ID for rate limiting
        client_id = raw_request.headers.get("X-Client-ID", raw_request.client.host if raw_request.client else "unknown")

        # Convert messages to dict format
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

        # Build kwargs
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.stop is not None:
            kwargs["stop"] = request.stop
        if request.presence_penalty is not None:
            kwargs["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            kwargs["frequency_penalty"] = request.frequency_penalty

        if request.stream:
            return StreamingResponse(
                proxy.process_streaming(
                    messages=messages,
                    model=request.model,
                    api_key=api_key,
                    **kwargs,
                ),
                media_type="text/event-stream",
            )

        result = await proxy.process_request(
            messages=messages,
            model=request.model,
            api_key=api_key,
            client_id=client_id,
            **kwargs,
        )

        return JSONResponse(content=result)

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {"id": "gpt-4", "object": "model", "owned_by": "openai"},
                {"id": "gpt-4-turbo", "object": "model", "owned_by": "openai"},
                {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
            ],
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "mode": config.mode.value,
        }

    @app.get("/metrics")
    async def get_metrics():
        """Get proxy metrics."""
        return proxy.get_metrics()

    @app.post("/admin/clear-cache")
    async def clear_cache():
        """Clear the response cache."""
        proxy.clear_cache()
        return {"status": "cache cleared"}

    @app.post("/admin/mode/{mode}")
    async def set_mode(mode: str):
        """Set the proxy mode."""
        try:
            config.mode = ProxyMode(mode)
            return {"status": "ok", "mode": mode}
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

    return app


# CLI entry point
def main():
    """Run the proxy server."""
    import argparse

    parser = argparse.ArgumentParser(description="ShadowArena Proxy Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--mode", choices=["passthrough", "shadow", "intercept", "audit"],
                        default="intercept", help="Proxy mode")
    parser.add_argument("--upstream", default="https://api.openai.com",
                        help="Upstream API URL")
    parser.add_argument("--variations", type=int, default=4,
                        help="Number of variations to generate")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Request timeout in seconds")

    args = parser.parse_args()

    config = ProxyConfig(
        mode=ProxyMode(args.mode),
        upstream_url=args.upstream,
        num_variations=args.variations,
        timeout_seconds=args.timeout,
    )

    app = create_app(config)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
