.DEFAULT_GOAL := help
PY ?= python3

.PHONY: help install data reference demo-samples dev lint format test check docker

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Installe le projet et les outils de développement
	$(PY) -m pip install -e ".[dev]"

data: ## Télécharge les données publiques (TCGA, GEO)
	$(PY) -m burotu.data.gdc
	$(PY) -m burotu.data.geo

reference: ## Construit le panel ciblé et le découpage train/test
	$(PY) -m burotu.data.reference

demo-samples: ## Génère des fichiers labo simulés pour la démo
	$(PY) scripts/make_demo_samples.py

dev: ## Lance le site en local (http://localhost:8000 et http://admin.localhost:8000)
	BUROTU_ADMIN_PASSWORD=$${BUROTU_ADMIN_PASSWORD:-dev} $(PY) -m uvicorn burotu_web.main:app --reload --port 8000

lint: ## Vérifie le style du code
	ruff check src tests scripts
	ruff format --check src tests scripts

format: ## Formate le code
	ruff check --fix src tests scripts
	ruff format src tests scripts

test: ## Lance les tests
	$(PY) -m pytest

check: lint test ## Lint et tests (comme la CI)

docker: ## Construit l'image Docker du site
	docker build -t burotu-web .
