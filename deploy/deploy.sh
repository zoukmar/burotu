#!/usr/bin/env bash
# Deploy one Burotu version on the server. Called by GitHub Actions over SSH:
#   deploy.sh <tag-image>
# Idempotent: running it again with the same tag changes nothing.
set -euo pipefail

TAG="${1:?usage: deploy.sh <tag-image>}"
DIR="${BUROTU_DIR:-/opt/burotu}"
cd "$DIR"

if [[ ! -f .env ]]; then
  echo "ERROR: $DIR/.env is missing (see deploy/.env.example)" >&2
  exit 1
fi

PREVIOUS="$(cat .image-tag 2>/dev/null || true)"
echo "Deploying ${TAG} (previous: ${PREVIOUS:-none})"

export BUROTU_IMAGE_TAG="$TAG"
docker compose pull --quiet web
docker compose up -d --remove-orphans web

# Wait for the container to answer before accepting the version.
for i in $(seq 1 30); do
  if curl -fsS -H "Host: burotu.com" http://127.0.0.1:8120/healthz >/dev/null; then
    echo "$TAG" > .image-tag
    echo "OK: Burotu ${TAG} is up (${i}s)."
    docker image prune -f >/dev/null || true  # dangling images only
    exit 0
  fi
  sleep 1
done

echo "FAILED: the site does not answer, rolling back to ${PREVIOUS:-nothing}" >&2
docker compose logs --tail 50 web >&2 || true
if [[ -n "$PREVIOUS" ]]; then
  BUROTU_IMAGE_TAG="$PREVIOUS" docker compose up -d web
fi
exit 1
