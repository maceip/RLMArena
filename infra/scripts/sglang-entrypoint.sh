#!/bin/sh
set -e

# Start tailscale if configured
if [ -n "${TS_AUTHKEY}" ]; then
    tailscaled --state=/var/lib/tailscale/tailscaled.state &
    sleep 2
    tailscale up --authkey=${TS_AUTHKEY} --hostname=${TS_HOSTNAME:-sglang-router} ${TS_EXTRA_ARGS} || true
fi

# Execute the main command
exec "$@"
