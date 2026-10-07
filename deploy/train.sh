#!/usr/bin/env bash
# Triggers a model rebuild: exports the dataset, trains, exports ONNX, publishes.
# Training runs on CPU by design - the host GPU is never touched.
# Usage: ./deploy/train.sh              (full train)
#        ./deploy/train.sh --export-only
set -euo pipefail
cd "$(dirname "$0")/.."

NETWORK="${NETWORK:-frigate_default}"

docker build -q -t errata:local .
docker build -q -t errata-train:local -f - . <<'EOF'
FROM errata:local
USER root
COPY requirements-training.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-training.txt
USER errata
EOF

exec docker run --rm --network "$NETWORK" \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/models:/publish" \
  -v "$(pwd)/config.yaml:/config/errata.yaml:ro" \
  errata-train:local python -m errata.trainer --train "$@"
