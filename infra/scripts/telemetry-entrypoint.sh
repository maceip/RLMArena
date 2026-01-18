#!/bin/sh
set -e

if [ -n "${TS_AUTHKEY}" ]; then
    tailscaled --state=/var/lib/tailscale/tailscaled.state &
    sleep 2
    tailscale up --authkey=${TS_AUTHKEY} --hostname=${TS_HOSTNAME:-telemetry} ${TS_EXTRA_ARGS} || true
fi

exec "$@"
