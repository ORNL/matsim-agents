#!/bin/bash
# Run one Ray node and keep its logs.
#
#   ray-node.sh <log dir> <ray> start --temp-dir=<node-local dir> ... --block
#
# Ray's session directory must be a short node-local path (its Unix sockets have
# a path-length limit), so it is only scratch. When the node stops, or this
# script is terminated (which stops the node), the node's Ray logs are copied to
# <log dir>/<host>/ on shared storage and the session directory is removed.
set -uo pipefail

log_dir=${1:?usage: ray-node.sh <log dir> <ray> start --temp-dir=<dir> ...}
shift
temp_dir=
for arg in "$@"; do
  [[ "$arg" == --temp-dir=* ]] && temp_dir=${arg#--temp-dir=}
done
[[ -n "$temp_dir" ]] || { echo "ray-node.sh: pass --temp-dir=<node-local dir>" >&2; exit 2; }

stop=0 ray_pid= status=0
# `ray start --block` shuts its node down on SIGTERM.
trap 'stop=1; [[ -z $ray_pid ]] || kill -TERM "$ray_pid" 2>/dev/null' TERM INT
rm -rf "$temp_dir"
if (( ! stop )); then
  "$@" &
  ray_pid=$!
  wait "$ray_pid"; status=$?
  # A signal interrupts `wait`; wait again until Ray has shut down.
  while kill -0 "$ray_pid" 2>/dev/null; do wait "$ray_pid"; status=$?; done
fi

host=$(hostname)
dest="$log_dir/$host"
if mkdir -p "$dest" && cp -r "$temp_dir/session_latest/logs/." "$dest/" 2>/dev/null; then
  echo "[ray] logs of $host saved to $dest"
else
  echo "[ray] WARNING: could not save the Ray logs of $host to $dest" >&2
fi
rm -rf "$temp_dir"
exit "$status"
