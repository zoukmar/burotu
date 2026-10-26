#!/usr/bin/env bash
# Déploie une version de Burotu sur le serveur. Appelé par GitHub Actions via SSH :
#   deploy.sh <tag-image>
# Idempotent : relancer avec le même tag ne change rien.
set -euo pipefail

TAG="${1:?usage: deploy.sh <tag-image>}"
DIR="${BUROTU_DIR:-/opt/burotu}"
cd "$DIR"

if [[ ! -f .env ]]; then
  echo "ERREUR : $DIR/.env manquant (voir deploy/.env.example)" >&2
  exit 1
fi

PREVIOUS="$(cat .image-tag 2>/dev/null || true)"
echo "Déploiement de ${TAG} (précédent : ${PREVIOUS:-aucun})"

export BUROTU_IMAGE_TAG="$TAG"
docker compose pull --quiet web
docker compose up -d --remove-orphans web

# Attend que le conteneur réponde avant de valider la version.
for i in $(seq 1 30); do
  if curl -fsS -H "Host: burotu.com" http://127.0.0.1:8120/healthz >/dev/null; then
    echo "$TAG" > .image-tag
    echo "OK : Burotu ${TAG} répond (${i}s)."
    docker image prune -f >/dev/null || true  # images orphelines uniquement
    exit 0
  fi
  sleep 1
done

echo "ÉCHEC : le site ne répond pas, retour à ${PREVIOUS:-rien}" >&2
docker compose logs --tail 50 web >&2 || true
if [[ -n "$PREVIOUS" ]]; then
  BUROTU_IMAGE_TAG="$PREVIOUS" docker compose up -d web
fi
exit 1
