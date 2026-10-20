#!/bin/bash
# Run one Ray node and keep its logs.
#
#   ray-node.sh <log dir> <ray> start --temp-dir=<node-local dir> ... --block
#
# Ray's session directory must be a short node-local path (its Unix sockets have
# a path-length limit), so it is only scratch. When the node stops, or this
# script is terminated (which stops the node), the node's Ray logs are copied to
# <log dir>/<host>/ on shared storage and the session directory is removed.
#
# Slurm kills a step outright when its srun is signalled, so a node on another
# host is asked to stop through a file instead: with RAY_NODE_STOP_FILE set, the
# node stops as if terminated once that file exists. Only this node's processes
# are stopped, never other Ray instances on the host.
set -uo pipefail

log_dir=${1:?usage: ray-node.sh <log dir> <ray> start --temp-dir=<dir> ...}
shift
temp_dir=
for arg in "$@"; do
  [[ "$arg" == --temp-dir=* ]] && temp_dir=${arg#--temp-dir=}
done
[[ -n "$temp_dir" ]] || { echo "ray-node.sh: pass --temp-dir=<node-local dir>" >&2; exit 2; }

host=$(hostname)
dest="$log_dir/$host"
save_logs() {
  mkdir -p "$dest" && cp -r "$temp_dir/session_latest/logs/." "$dest/" 2>/dev/null
}

stop=0 ray_pid= status=0
# Copy the logs as soon as the node is asked to stop: the scheduler may kill
# this step before Ray finishes shutting down. `ray start --block` shuts its
# node down on SIGTERM; the logs are copied again once it has.
trap 'stop=1; save_logs; [[ -z $ray_pid ]] || kill -TERM "$ray_pid" 2>/dev/null' TERM INT
rm -rf "$temp_dir"
watcher=
if [[ -n "${RAY_NODE_STOP_FILE:-}" ]]; then
  ( while [[ ! -e "$RAY_NODE_STOP_FILE" ]] && kill -0 $$ 2>/dev/null; do sleep 2; done
    kill -TERM $$ 2>/dev/null ) &
  watcher=$!
fi
if (( ! stop )); then
  "$@" &
  ray_pid=$!
  wait "$ray_pid"; status=$?
  # A signal interrupts `wait`; wait again until Ray has shut down.
  while kill -0 "$ray_pid" 2>/dev/null; do wait "$ray_pid"; status=$?; done
fi

[[ -z "$watcher" ]] || kill "$watcher" 2>/dev/null
# Stop any process of this Ray node that outlived `ray start`. They name the
# node's session directory on their command lines, so other Ray instances on
# the host (other servers' nodes) are never touched, unlike `ray stop`.
leftovers=$(ps -u "$(id -u)" -o pid=,ppid=,args= \
  | RAY_NODE_DIR="$temp_dir" SELF=$$ \
    awk 'index($0, ENVIRON["RAY_NODE_DIR"]) && $1 != ENVIRON["SELF"] && $2 != ENVIRON["SELF"] {print $1}')
[[ -z "$leftovers" ]] || kill -KILL $leftovers 2>/dev/null
if save_logs; then
  echo "[ray] logs of $host saved to $dest"
else
  echo "[ray] WARNING: could not save the Ray logs of $host to $dest" >&2
fi
rm -rf "$temp_dir"
exit "$status"
