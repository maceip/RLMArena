#!/bin/sh
set -e

# Start tailscaled
tailscaled --state=${TS_STATE_DIR}/tailscaled.state --socket=${TS_SOCKET} &

# Wait for socket
while [ ! -S ${TS_SOCKET} ]; do
    sleep 0.1
done

# Authenticate with tailscale
if [ -n "${TS_AUTHKEY}" ]; then
    tailscale up --authkey=${TS_AUTHKEY} --hostname=${TS_HOSTNAME:-$(hostname)} ${TS_EXTRA_ARGS}
fi

# Keep container running
exec tail -f /dev/null
