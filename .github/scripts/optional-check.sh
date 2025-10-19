#!/usr/bin/env bash
set -euo pipefail

# optional-check.sh
# Usage: optional-check.sh <file-to-check> -- <command> [args...]
# If <file-to-check> does not exist, exits 0 (skips the optional check).
# Otherwise runs the provided command and forwards its exit code.

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <file-to-check> -- <command> [args...]"
  exit 2
fi

FILE="$1"
shift

if [ "$1" != "--" ]; then
  echo "Error: expected '--' separator between file and command"
  echo "Usage: $0 <file-to-check> -- <command> [args...]"
  exit 2
fi

shift

# Build a readable command string for logging
CMD="$*"

if [ ! -f "$FILE" ]; then
  echo "⚠️ Optional check skipped: '$FILE' not found — would have run: $CMD"
  exit 0
fi

echo "Running optional check against: $FILE"
"$@"
EXIT_CODE=$?
exit $EXIT_CODE
