#!/bin/sh
set -e

# Source common tailscale initialization
. /app/scripts/common-tailscale.sh 2>/dev/null || true

# Start tailscale if configured
if [ -n "${TS_AUTHKEY}" ]; then
    start_tailscale || echo "Tailscale init failed, continuing without mesh"
fi

# Execute the main command
exec "$@"
