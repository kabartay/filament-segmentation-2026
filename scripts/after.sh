#!/bin/bash
# Run a command after process <pid> exits (keeps one GPU job at a time).
# Usage: scripts/after.sh <pid> <command...>
cd "$(dirname "$0")/.."
pid=$1; shift
while kill -0 "$pid" 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) start: $*"
exec "$@"
