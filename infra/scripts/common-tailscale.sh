#!/bin/sh
# Common tailscale initialization for all services

start_tailscale() {
    if [ -n "${TS_AUTHKEY}" ]; then
        # Start tailscaled in background
        tailscaled --state=/var/lib/tailscale/tailscaled.state \
                   --socket=/var/run/tailscale/tailscaled.sock &

        # Wait for socket
        while [ ! -S /var/run/tailscale/tailscaled.sock ]; do
            sleep 0.1
        done

        # Authenticate
        tailscale up --authkey=${TS_AUTHKEY} \
                     --hostname=${TS_HOSTNAME:-$(hostname)} \
                     ${TS_EXTRA_ARGS}

        echo "Tailscale connected as ${TS_HOSTNAME:-$(hostname)}"
    else
        echo "TS_AUTHKEY not set, skipping tailscale"
    fi
}
