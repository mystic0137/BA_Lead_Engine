.ONESHELL:
SHELL := /bin/bash

.DEFAULT_GOAL := setup

.PHONY: help install uninstall train serve ui run dev ci benchmark clean retrain ingest setup test start-clean

PYTHON := venv/bin/python
PIP := venv/bin/pip
UVICORN := venv/bin/uvicorn
STREAMLIT := venv/bin/streamlit
MODELS_DIR := models
DATA_DIR := data
CHROMA_DB_DIR := $(DATA_DIR)/chroma_db
FINETUNING_DIR := $(DATA_DIR)/finetuning
XGB_ONNX := $(MODELS_DIR)/xgboost.onnx
XGB_CONFIG := $(MODELS_DIR)/xgboost_config.json
VENV_DONE := venv/.install_done
PACKAGE_NAME := british_airways_booking_predictor

help:
	@echo "Usage:"
	@echo "  make             Full setup: clean install + train + serve (default)"
	@echo "  make install     Install all dependencies"
	@echo "  make uninstall   Remove stale project installs"
	@echo "  make train       Train models (skips if already trained)"
	@echo "  make retrain     Force retrain models"
	@echo "  make ingest      Ingest policy docs into chromadb"
	@echo "  make benchmark   Run performance verification"
	@echo "  make ci          Run tests and bechmarks"
	@echo "  make serve       Start FastAPI backend"
	@echo "  make ui          Start Streamlit frontend"
	@echo "  make run         Start both services"
	@echo "  make dev         Start both services with hot-reload"
	@echo "  make clean       Delete trained models and sentinel"
	@echo "  make test        Test cases using pytest"
	@echo "  make start-clean starts build from clean state"

uninstall:
	@echo "=== Removing stale project installs ==="
	$(PIP) uninstall $(PACKAGE_NAME) -y 2>/dev/null || true
	find venv/lib -name "direct_url.json" \
		| xargs grep -l "$(PACKAGE_NAME)" 2>/dev/null \
		| xargs rm -fv
	find venv/lib -name "*.egg-link" \
		| xargs grep -l "$(PACKAGE_NAME)" 2>/dev/null \
		| xargs rm -fv
	find venv/lib -path "*$(PACKAGE_NAME)*.dist-info" \
		-exec rm -rfv {} + 2>/dev/null || true
	find venv/lib -path "*$(PACKAGE_NAME)*.egg-info" \
		-exec rm -rfv {} + 2>/dev/null || true
	rm -f $(VENV_DONE)
	@echo "=== Stale installs removed ==="

$(VENV_DONE): pyproject.toml
	@echo "=== Removing stale installs ==="
	-$(PIP) uninstall $(PACKAGE_NAME) -y 2>/dev/null
	rm -f $(VENV_DONE)
	@echo "=== Installing dependencies ==="
	$(PIP) install -e ".[dev,ui,rag]"
	touch $(VENV_DONE)
	@echo "=== Dependencies Installed ==="

install: $(VENV_DONE)

$(XGB_ONNX): $(VENV_DONE) src/train.py src/preprocess.py src/models.py src/config.py
	@echo "=== Training Models ===="
	$(PYTHON) src/train.py
	@echo "=== Training Completed ==="

train: $(XGB_ONNX)

test:
	@echo "=== Running Unit and Integration Tests"
	$(PYTHON) -m pytest tests/ -v
	@echo "=== Tests Completed ==="

benchmark: train
	@echo "=== Running Benchmark ==="
	$(UVICORN) app.main:app --host 127.0.0.1 --port 8000 & \
	sleep 3; \
	$(PYTHON) -m scripts.benchmark
	@echo "=== Benchmark Completed ==="

retrain:
	rm -f $(XGB_ONNX)
	$(PYTHON) src/train.py

serve: train
	$(UVICORN) app.main:app --host 127.0.0.1 --port 8000

ui:
	$(STREAMLIT) run frontend/streamlit_app.py --server.port 8501

run: train
	@echo "=== Starting Backend and Frontend services ==="
	$(UVICORN) app.main:app --host 127.0.0.1 --port 8000 & \
	sleep 3; \
	$(STREAMLIT) run frontend/streamlit_app.py --server.port 8501

dev: train
	@echo "=== Starting dev server with hot-reload ==="
	$(UVICORN) app.main:app --host 127.0.0.1 --port 8000 --reload & \
	sleep 5; \
	$(STREAMLIT) run frontend/streamlit_app.py --server.port 8501

ingest:
	@echo "=== Ingesting Policy Documents ==="
	$(PYTHON) -c "from src.rag.manager import RAGManager; RAGManager().ingest()"
	@echo "=== Policy Documents Ingested ==="

setup: install ingest train run

ci: test benchmark

start-clean: clean setup

clean:
	@echo "=== Cleaning project artifacts ==="

	# Clear ChromaDB contents except .gitkeep
	find "$(CHROMA_DB_DIR)" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +

	# Remove finetuning artifacts
	find "$(FINETUNING_DIR)" -mindepth 1 -exec rm -rf {} +

	# Remove model artifacts (files OR dirs)
	rm -f "$(XGB_ONNX)" "$(XGB_CONFIG)"

	rm -fv "$(VENV_DONE)"

	@echo "=== Clean complete ==="