# VaaS Infrastructure Agents

This document describes the service accounts, environment variables, and secrets needed to deploy the VaaS platform in production.

## Architecture Overview

All services communicate over a Tailscale mesh network (tailnet) for secure, zero-trust networking. Each container runs a Tailscale client sidecar. High-throughput paths (SGLang <-> Gateway) use Docker compose networking for lower latency.

### Shared Database Access

**PostgreSQL** is the primary database for all services. It was chosen for:
- Structured data storage (trajectories, certificates, training metadata)
- Audit logging and policy versioning for leash/cedar/opa
- Mature async drivers (asyncpg, sqlalchemy)
- Simple connection-string based access

All services receive `DATABASE_URL=postgresql://arena:PASSWORD@postgres:5432/arena` environment variable.

| Service | Database Usage |
|---------|---------------|
| arena-service | Trajectories, comparisons, certificates |
| vaas-gateway | API keys, rate limit state, request logs |
| llama-factory | Training runs, DPO triplets, model metadata |
| sglang-router | Model registry, serving metrics |
| leash-enforcer | Policy audit logs, decision history |
| opa-server | Policy bundles metadata, decision logs |
| telemetry | Uses ClickHouse for high-volume analytics |

**Redis** (`redis://redis:6379/0`) is used for:
- Rate limiting (token bucket state)
- Session caching
- Pub/sub for real-time updates

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Tailnet                                     │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ LlamaFactory│  │  SGLang     │  │   Arena     │  │  Dashboard  │    │
│  │  (training) │  │  (serving)  │  │ (tournament)│  │   (web ui)  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                │                │                │            │
│         └────────────────┼────────────────┼────────────────┘            │
│                          │                │                             │
│                   ┌──────┴──────┐  ┌──────┴──────┐                     │
│                   │   Gateway   │  │  Telemetry  │                     │
│                   │   (api)     │  │  (logging)  │                     │
│                   └─────────────┘  └─────────────┘                     │
│                          │                                              │
│              ─ ─ ─ ─ ─ ─ ┴ ─ ─ ─ ─ ─ ─                                │
│             (compose network for low latency)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Service Accounts

### 1. tailscale-operator
**Purpose:** Manages tailscale authentication for all containers
**Required Secrets:**
- `TS_AUTHKEY` - Tailscale auth key (reusable, ephemeral recommended)
- `TS_STATE_DIR` - State directory path (default: /var/lib/tailscale)

**Service Account Permissions:**
- Create ephemeral nodes
- Tag nodes with service identifiers
- ACL: Allow inter-service communication within vaas namespace

### 2. llama-factory
**Purpose:** Fine-tuning and distillation training jobs
**Required Environment Variables:**
```bash
# Model storage
HF_HOME=/models/huggingface
MODEL_CACHE_DIR=/models/cache

# Training configuration
WANDB_API_KEY=<wandb-api-key>
WANDB_PROJECT=vaas-training

# GPU configuration
CUDA_VISIBLE_DEVICES=0,1,2,3
NCCL_DEBUG=INFO

# S3/MinIO for checkpoints
AWS_ACCESS_KEY_ID=<checkpoint-storage-key>
AWS_SECRET_ACCESS_KEY=<checkpoint-storage-secret>
CHECKPOINT_BUCKET=vaas-checkpoints
CHECKPOINT_ENDPOINT=<s3-compatible-endpoint>

# Tailscale
TS_HOSTNAME=llama-factory
TS_EXTRA_ARGS=--accept-routes
```

**Service Account Permissions:**
- Read/write to model storage
- Read/write to checkpoint bucket
- GPU access (nvidia runtime)
- Network: outbound to huggingface.co, wandb.ai

### 3. sglang-router
**Purpose:** High-performance model serving with RadixAttention
**Required Environment Variables:**
```bash
# Model configuration
MODEL_PATH=/models/served/current
TOKENIZER_PATH=/models/served/current

# Server configuration
SGLANG_HOST=0.0.0.0
SGLANG_PORT=30000
SGLANG_TP_SIZE=4
SGLANG_MEM_FRACTION=0.9

# RadixAttention cache
RADIX_CACHE_SIZE_GB=32
RADIX_CACHE_DIR=/cache/radix

# Metrics
PROMETHEUS_PORT=9090

# Tailscale
TS_HOSTNAME=sglang-router
TS_EXTRA_ARGS=--accept-routes
```

**Service Account Permissions:**
- Read from model storage
- GPU access (nvidia runtime, requires multi-gpu for tensor parallel)
- Network: high-bandwidth internal (compose network to gateway)

### 4. vaas-gateway
**Purpose:** API gateway with rate limiting, caching, and request routing
**Required Environment Variables:**
```bash
# Server configuration
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8080
GATEWAY_WORKERS=4

# Rate limiting
REDIS_URL=redis://redis:6379/0
RATE_LIMIT_DEFAULT=100/minute

# Authentication
JWT_SECRET=<jwt-signing-secret>
API_KEY_SALT=<api-key-hash-salt>

# Backend endpoints (via tailnet)
SGLANG_ENDPOINT=http://sglang-router:30000
ARENA_ENDPOINT=http://arena:8081
TELEMETRY_ENDPOINT=http://telemetry:8082

# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=30

# Tailscale
TS_HOSTNAME=vaas-gateway
TS_EXTRA_ARGS=--accept-routes
```

**Service Account Permissions:**
- Network: inbound from internet (public API)
- Network: outbound to all internal services
- Read/write to Redis for rate limiting

### 5. arena-service
**Purpose:** Tournament bracket execution and trajectory comparison
**Required Environment Variables:**
```bash
# Server configuration
ARENA_HOST=0.0.0.0
ARENA_PORT=8081

# Database
DATABASE_URL=postgresql://arena:password@postgres:5432/arena

# Verification backends
FIRECRACKER_ENDPOINT=http://firecracker:8090
LEASH_ENDPOINT=http://leash:8091
OPA_ENDPOINT=http://opa:8181

# Shadow loop configuration
SHADOW_LOOP_VARIATIONS=5
SHADOW_LOOP_TEMPERATURE_RANGE=0.1,0.7,1.0

# Tailscale
TS_HOSTNAME=arena
TS_EXTRA_ARGS=--accept-routes
```

**Service Account Permissions:**
- Read/write to PostgreSQL
- Network: outbound to verification services
- Network: outbound to sglang for model calls

### 6. telemetry-collector
**Purpose:** Audit logging and distributed tracing
**Required Environment Variables:**
```bash
# Server configuration
TELEMETRY_HOST=0.0.0.0
TELEMETRY_PORT=8082

# Storage
CLICKHOUSE_URL=clickhouse://telemetry:password@clickhouse:9000/telemetry
RETENTION_DAYS=90

# Export
OTLP_ENDPOINT=http://jaeger:4317
PROMETHEUS_PUSHGATEWAY=http://prometheus:9091

# Tailscale
TS_HOSTNAME=telemetry
TS_EXTRA_ARGS=--accept-routes
```

**Service Account Permissions:**
- Write to ClickHouse
- Network: inbound from all services (receives telemetry)
- Network: outbound to observability stack

### 7. dashboard-web
**Purpose:** Web UI for monitoring, configuration, and manual review
**Required Environment Variables:**
```bash
# Server configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=3000

# Backend API
API_ENDPOINT=http://vaas-gateway:8080
WEBSOCKET_ENDPOINT=ws://vaas-gateway:8080/ws

# Auth
OAUTH_CLIENT_ID=<oauth-client-id>
OAUTH_CLIENT_SECRET=<oauth-client-secret>
OAUTH_ISSUER=https://auth.example.com

# Session
SESSION_SECRET=<session-signing-secret>
SESSION_STORE_URL=redis://redis:6379/1

# Tailscale
TS_HOSTNAME=dashboard
TS_EXTRA_ARGS=--accept-routes
```

**Service Account Permissions:**
- Network: inbound from internet (web traffic)
- Network: outbound to gateway only

### 8. firecracker-executor
**Purpose:** Sandboxed code execution in microVMs
**Required Environment Variables:**
```bash
# Server configuration
FIRECRACKER_HOST=0.0.0.0
FIRECRACKER_PORT=8090

# VM configuration
VM_KERNEL_PATH=/kernels/vmlinux
VM_ROOTFS_PATH=/rootfs/base.ext4
VM_MEMORY_MB=512
VM_VCPU_COUNT=2
VM_TIMEOUT_SECONDS=30

# Jailer configuration
JAILER_PATH=/usr/bin/jailer
CHROOT_BASE=/srv/jailer

# Tailscale
TS_HOSTNAME=firecracker
TS_EXTRA_ARGS=--accept-routes
```

**Service Account Permissions:**
- Privileged container (KVM access)
- Read from kernel/rootfs storage
- Network: isolated (VMs have no network by default)

### 9. leash-enforcer
**Purpose:** Cedar-based network policy enforcement
**Required Environment Variables:**
```bash
# Server configuration
LEASH_HOST=0.0.0.0
LEASH_PORT=8091

# Policy configuration
CEDAR_POLICY_DIR=/policies/cedar
POLICY_RELOAD_INTERVAL=60

# Audit
AUDIT_LOG_PATH=/var/log/leash/audit.log

# Tailscale
TS_HOSTNAME=leash
TS_EXTRA_ARGS=--accept-routes
```

**Service Account Permissions:**
- Read from policy storage
- Network: kernel-level packet inspection (requires NET_ADMIN)

### 10. opa-server
**Purpose:** Rego policy evaluation for infrastructure compliance
**Required Environment Variables:**
```bash
# Server configuration
OPA_HOST=0.0.0.0
OPA_PORT=8181

# Policy configuration
OPA_BUNDLE_URL=http://policy-server/bundles/vaas.tar.gz
OPA_BUNDLE_REFRESH=300

# Decision logging
OPA_DECISION_LOG_CONSOLE=true

# Tailscale
TS_HOSTNAME=opa
TS_EXTRA_ARGS=--accept-routes
```

**Service Account Permissions:**
- Read from policy bundle server
- Network: inbound from arena and gateway

## Secrets Management

### Required Secrets (store in Vault, AWS Secrets Manager, or similar)

| Secret Name | Services | Description |
|-------------|----------|-------------|
| `ts-authkey` | All | Tailscale authentication key |
| `jwt-secret` | Gateway, Dashboard | JWT signing secret |
| `api-key-salt` | Gateway | Salt for API key hashing |
| `db-arena-password` | Arena, PostgreSQL | Arena database password |
| `db-telemetry-password` | Telemetry, ClickHouse | Telemetry database password |
| `redis-password` | Gateway, Dashboard, Redis | Redis authentication |
| `wandb-api-key` | LlamaFactory | Weights & Biases API key |
| `checkpoint-aws-key` | LlamaFactory | S3 access key for checkpoints |
| `checkpoint-aws-secret` | LlamaFactory | S3 secret key for checkpoints |
| `oauth-client-secret` | Dashboard | OAuth client secret |
| `session-secret` | Dashboard | Session signing secret |

## GPU Requirements

| Service | GPU Type | Count | VRAM | Notes |
|---------|----------|-------|------|-------|
| LlamaFactory | A100/H100 | 4-8 | 80GB each | Training, FSDP/DeepSpeed |
| SGLang | A100/H100 | 4 | 80GB each | Inference, tensor parallel |
| Firecracker | None | - | - | CPU-only VM execution |

## Storage Requirements

| Volume | Size | Type | Services |
|--------|------|------|----------|
| `/models` | 2TB | NVMe/SSD | LlamaFactory, SGLang |
| `/cache/radix` | 100GB | NVMe | SGLang |
| `/checkpoints` | 1TB | S3/Object | LlamaFactory |
| `/data/postgres` | 500GB | SSD | PostgreSQL |
| `/data/clickhouse` | 1TB | SSD | ClickHouse |
| `/kernels` | 10GB | SSD | Firecracker |
| `/rootfs` | 50GB | SSD | Firecracker |

## Network ACLs (Tailscale)

```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["tag:vaas-gateway"],
      "dst": ["tag:vaas-internal:*"]
    },
    {
      "action": "accept",
      "src": ["tag:vaas-arena"],
      "dst": ["tag:vaas-verifier:*", "tag:vaas-sglang:*"]
    },
    {
      "action": "accept",
      "src": ["tag:vaas-dashboard"],
      "dst": ["tag:vaas-gateway:*"]
    },
    {
      "action": "accept",
      "src": ["tag:vaas-internal"],
      "dst": ["tag:vaas-telemetry:*"]
    }
  ],
  "tagOwners": {
    "tag:vaas-gateway": ["autogroup:admin"],
    "tag:vaas-internal": ["autogroup:admin"],
    "tag:vaas-verifier": ["autogroup:admin"],
    "tag:vaas-sglang": ["autogroup:admin"],
    "tag:vaas-telemetry": ["autogroup:admin"],
    "tag:vaas-dashboard": ["autogroup:admin"]
  }
}
```

## GitHub Packages

Docker images are built and pushed to GitHub Container Registry (ghcr.io) on every merge to main.

```
ghcr.io/maceip/rlmarena/llama-factory:latest
ghcr.io/maceip/rlmarena/sglang-router:latest
ghcr.io/maceip/rlmarena/vaas-gateway:latest
ghcr.io/maceip/rlmarena/arena-service:latest
ghcr.io/maceip/rlmarena/telemetry-collector:latest
ghcr.io/maceip/rlmarena/dashboard-web:latest
ghcr.io/maceip/rlmarena/firecracker-executor:latest
ghcr.io/maceip/rlmarena/leash-enforcer:latest
ghcr.io/maceip/rlmarena/opa-server:latest
ghcr.io/maceip/rlmarena/tailscale-sidecar:latest
```

## Quick Start

1. Copy `.env.example` to `.env` and fill in secrets
2. Create Tailscale auth key at https://login.tailscale.com/admin/settings/keys
3. Run `docker compose up -d`
4. Access dashboard at http://dashboard:3000 (via tailnet)
