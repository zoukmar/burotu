.DEFAULT_GOAL := help
PY ?= python3

.PHONY: help install data public-data reference baseline logreg models evaluate demo-samples dev lint format test check docker

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the project and development tools
	$(PY) -m pip install -e ".[dev]"

data: ## Download public data (TCGA, GEO)
	$(PY) -m burotu.data.gdc
	$(PY) -m burotu.data.geo

reference: ## Build the targeted panel and train/test split
	$(PY) -m burotu.data.reference

public-data: ## Download the open real-plasma and healthy-blood sets from GEO, cut to the panel
	$(PY) -m burotu.data.public

baseline: ## Train, evaluate and export the classic model (reports/baseline.json, site assets)
	$(PY) -m burotu.models.classic.train
	$(PY) -m burotu.models.classic.export

logreg: ## Train and export the logistic-regression ensembles (reports/logreg.json, site assets)
	$(PY) -m burotu.models.logreg.train

models: baseline logreg ## Train every CPU model (Muse Glimmer and CpGPT train on Modal)

evaluate: ## Score the CPU models on the open real-plasma datasets (reports/public_plasma.json)
	$(PY) -m burotu.evaluation.public_plasma

demo-samples: ## Generate simulated lab files for the demo
	$(PY) scripts/make_demo_samples.py
	$(PY) scripts/make_site_demo_samples.py

dev: ## Run the site locally (http://localhost:8000 and http://admin.localhost:8000)
	BUROTU_ADMIN_PASSWORD=$${BUROTU_ADMIN_PASSWORD:-dev} $(PY) -m uvicorn burotu_web.main:app --reload --port 8000

lint: ## Check code style
	ruff check src tests scripts
	ruff format --check src tests scripts

format: ## Format the code
	ruff check --fix src tests scripts
	ruff format src tests scripts

test: ## Run the tests
	$(PY) -m pytest

check: lint test ## Lint and tests (same as CI)

docker: ## Build the site Docker image
	docker build -t burotu-web .
