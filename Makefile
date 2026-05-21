.DEFAULT_GOAL := setup

.PHONY: help install uninstall train serve ui run ci benchmark clean retrain ingest setup test start-clean

PYTHON := venv/bin/python
PIP := venv/bin/pip
UVICORN := venv/bin/uvicorn
STREAMLIT := venv/bin/streamlit
MODELS_DIR := models
DATA_DIR := data
CHROMA_DB_DIR := $(DATA_DIR)/chroma_db
FINETUNING_DIR := $(DATA_DIR)/finetuning
RF_ONNX := $(MODELS_DIR)/random_forest.onnx
XGB_ONNX := $(MODELS_DIR)/xgboost.onnx
VENV_DONE := venv/.install_done
PACKAGE_NAME := british_airways_booking_predictor

help:
	@echo "Usage:"
	@echo "  make             Full setup: clean install + train + serve (default)"
	@echo "  make install     Install all dependencies"
	@echo "  make uninstall   Remove stale project installs"
	@echo "  make train       Train models (skips if already trained)"
	@echo "  make retrain     Force retrain models"
	@echo "  make benchmark   Run performance verification"
	@echo "  make ci          Run tests and bechmarks"
	@echo "  make serve       Start FastAPI backend"
	@echo "  make ui          Start Streamlit frontend"
	@echo "  make run         Start both services"
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

$(VENV_DONE): uninstall pyproject.toml
	@echo "=== Installing dependencies ==="
	$(PIP) install -e ".[dev,ui,rag]"
	touch $(VENV_DONE)

install: $(VENV_DONE)

$(RF_ONNX) $(XGB_ONNX): $(VENV_DONE) src/train.py src/preprocess.py src/models.py src/config.py
	$(PYTHON) src/train.py

train: $(RF_ONNX) $(XGB_ONNX)

test: 
	$(PYTHON) -m pytest tests/ -v

benchmark: train
	$(MAKE) serve & \
	sleep 3; \
	$(PYTHON) -m scripts.main

retrain:
	rm -f $(RF_ONNX) $(XGB_ONNX)
	$(PYTHON) src/train.py

serve: train
	$(UVICORN) app.main:app --host 127.0.0.1 --port 8000 --reload

ui:
	$(STREAMLIT) run frontend/streamlit_app.py --server.port 8501

run: train
	$(MAKE) serve & \
	sleep 3; \
	$(MAKE) ui

ingest:
	$(PYTHON) -m src.rag.ingest

setup: install ingest train run

ci: test benchmark

start-clean: clean setup

clean:
	rm -rf $(CHROMA_DB_DIR) -type f ! -name ".gitkeep" -delete
	rm -f $(FINETUNING_DIR)/*jsonl
	rm -f $(MODELS_DIR)/*.onnx $(MODELS_DIR)/*.json
	rm -f $(VENV_DONE)