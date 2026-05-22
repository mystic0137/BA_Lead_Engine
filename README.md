# British Airways Lead Prioritization Engine

Production-grade ML system: booking propensity prediction → lead segmentation → RAG-grounded outreach emails. Built on a real BA customer booking dataset with end-to-end ML engineering patterns.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                 Streamlit Dashboard               │
│  Batch Scoring │ Lead Queue │ Model Analysis      │
│                          Email Generator          │
└──────────────────────┬───────────────────────────┘
                       │ HTTP
┌──────────────────────▼───────────────────────────┐
│                  FastAPI Backend                   │
│  /predict/*  │  /api/v1/rag/*  │  /health         │
└──────────────────────┬───────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
┌─────────▼─────────┐   ┌──────────▼──────────┐
│  InferenceEngine   │   │  RAG Pipeline        │
│  ONNX Runtime      │   │  ChromaDB → Retriever│
│  XGBoost (ONNX)    │   │  → LLM (fallback:    │
│  BACostCalculator  │   │  Groq→Together→Ollama)│
└───────────────────┘   │  → Feedback Logger    │
                        └──────────────────────┘
```

**Key decisions:**
- **ONNX over pickle** — version-independent, sub-millisecond inference, no Python runtime dependency at serving time; `.pkl` executes arbitrary code on load
- **Column-oriented inference** — Streamlit sends `df.to_dict(orient="list")` as a single POST; fully vectorized via `np.select`/`np.vectorize`, 2.4× faster than per-row loop. See [`BENCHMARKS.md`](BENCHMARKS.md)
- **`threading.Lock` + 30s timeout on ONNX session** — CPU-bound inference blocks the async event loop; the lock prevents concurrent session corruption and the timeout prevents indefinite hangs
- **RAG as token budget** — reduces prompt from ~20k to ~4k tokens (5× reduction), staying within Groq free-tier TPM limits; 128-entry LRU cache on retrieval for duplicate queries
- **Data flywheel** — each feedback save captures retrieved chunks, system prompt version, and human edits; produces SFT and DPO datasets ready for fine-tuning
- **Air-gapped runtime** — embedding model baked into Docker base image (`hf_models/`), `TRANSFORMERS_OFFLINE=1`; only external dependency at runtime is the Groq API
- **CPU-only PyTorch in Docker** — installed before `requirements.txt` with `--extra-index-url pytorch.org/whl/cpu`; intentionally absent from `requirements.txt` to prevent accidental CUDA wheel resolution on GPU hosts

---

## Features

**ML Pipeline** — XGBoost + Random Forest, target encoding + OHE, `scale_pos_weight` for imbalance, Youden's J threshold (0.309), ONNX export via skl2onnx

**Inference Engine** — Dual row/column endpoints, CSV upload, 30s ONNX timeout with `threading.Lock`, thread-safe priority queue

**Lead Segmentation** — 4-quadrant matrix (probability × value):

| Segment | Profile | Action |
|---|---|---|
| The Persuadable | Med/Low prob, High value | Call — highest priority |
| The VIP | High prob, High value | Email nudge — no discount |
| The Window Shopper | High prob, Low value | Drip sequence |
| The Lost Cause | Low prob, Low value | Suppress |

**RAG Email Generation** — Policy docs → `all-MiniLM-L6-v2` embeddings → ChromaDB → semantic retrieval → LLM. Retrieval LRU-cached (128 entries).

**LLM Fallback Chain** — Groq (Llama-4-Scout-17B) → Together AI (Llama-3.1-8B) → Ollama (llama3.1:8b). Enable via `LLM_FALLBACK_ENABLED=true`.

**Human Feedback Loop** — Accept/reject + star rating, auto-logged to `feedback_log.jsonl`, `sft_log.jsonl`, `dpo_log.jsonl`. Contradiction detection flagged in metadata.

**Data Integrity** — Raw dataset locked via SHA-256. Tampering raises `DataIntegrityError` before any pipeline runs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Training | scikit-learn, XGBoost, skl2onnx |
| Inference | ONNX Runtime (CPU), ThreadPoolExecutor (30s timeout) |
| API | FastAPI, Pydantic V2, async middleware |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Groq (primary), Together AI (fallback), Ollama (local) |
| Dashboard | Streamlit, Plotly |
| Container | Docker, Docker Compose (multi-stage build) |
| Config | pydantic-settings (`.env`, env vars) |

---

## Project Structure

```
├── src/
│   ├── config.py              # Paths, constants, Settings (lazy via get_settings())
│   ├── data_check.py          # SHA-256 integrity check
│   ├── models.py              # ModelConfig, get_xgb_model
│   ├── preprocess.py          # ColumnTransformer pipeline
│   ├── train.py               # Training + ONNX export
│   ├── analytics/finance.py   # BACostCalculator, segmentation, priority queue
│   ├── inference/
│   │   ├── engine.py          # ONNX session, row/column inference, threading.Lock
│   │   └── csv_utils.py       # CSV → ColumnOrientedInput parser
│   └── rag/
│       ├── manager.py         # RAGManager orchestration
│       ├── prompts.py         # Prompt templates, mappings, format helpers
│       ├── ingestion.py       # PDF/MD → chunks → ChromaDB
│       ├── feedback.py        # Feedback logging, SFT/DPO export
│       └── llm_client.py      # Multi-provider LLM client with fallback
├── app/
│   ├── main.py                # FastAPI app, lifespan, all endpoints, CORS
│   └── schemas.py             # Pydantic V2 request/response models
├── core/
│   └── state.py               # MLState singleton (eager init, fails fast)
├── frontend/
│   ├── streamlit_app.py       # Entry point (4 tabs)
│   └── tabs/                  # batch_scoring, lead_queue, model_analysis, email_generator
├── tests/                     # 160 tests, all mocked (ONNX, ChromaDB, Groq)
├── docker/
│   ├── dev/Dockerfile         # Hot-reload with cache mounts
│   └── prod/deploy.Dockerfile # Multi-stage: base → fastapi, base → streamlit
├── docker-compose.yml         # Prod: FastAPI + Streamlit + healthcheck
└── docker-compose.dev.yml     # Dev: host networking, volume-mounted hf_models
```

---

## Quickstart

**Recommended: Docker (dev)**
```bash
git clone https://github.com/mystic0137/BA_Lead_Engine
cd BA_Lead_Engine
echo 'GROQ_API_KEY="your_key_here"' > .env
docker compose -f docker-compose.dev.yml up --build
# FastAPI: http://localhost:8000 | Streamlit: http://localhost:8501
```

Dev compose runs with `--network host` to avoid WSL2/Windows port-mapping friction. `hf_models/` is volume-mounted for easy model swaps.

**Alternative: Make**
```bash
export GROQ_API_KEY=your_key_here
make   # install → ingest → train → serve + ui
```

Drop BA policy documents (PDF or markdown) into `data/policies/` before running `make ingest`.

**Targets:** `make install` | `make train` | `make ingest` | `make retrain` | `make serve` | `make ui` | `make test` | `make benchmark`

**Tests:** `pytest tests/ -v` — 160 tests, ~0.5s

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict/single` | Single record (row-oriented) |
| POST | `/predict/row_oriented` | Batch as `list[dict]` |
| POST | `/predict/column_oriented` | Batch as `dict[str, list]` — used by dashboard |
| POST | `/predict/column_oriented_bench` | Columnar JSON (benchmarking) |
| GET | `/health` | Readiness + model status |
| POST | `/api/v1/rag/generate` | RAG-grounded email generation |
| POST | `/api/v1/rag/feedback` | Human feedback logging |

CORS restricted to `CORS_ORIGINS` env var (default: `http://127.0.0.1:8501,http://localhost:8501`).

---

## Model Performance

### Classifier

20% stratified holdout, Youden's J threshold optimization.

| Model | ROC-AUC | Threshold |
|---|---|---|
| XGBoost (production) | 0.8228 | 0.309 |
| Random Forest | 0.7829 | 0.309 |

Class imbalance (~15% positive): precision on positive class ~0.38 at chosen threshold. Deliberate — cost of missing a high-value lead exceeds cost of a wasted outreach.

### Inference Benchmarks

1000 rows, `CPUExecutionProvider`, warmed keepalive connection. Full methodology in [`BENCHMARKS.md`](BENCHMARKS.md).

| Method | Total (ms) | Avg/row (ms) | Python heap delta (MB) |
|---|---|---|---|
| Direct ONNX (in-process) | 2.48 | 0.0025 | 0.039 |
| Column-oriented HTTP | 20.04 | 0.020 | 0.948 |
| Row-oriented HTTP | 48.82 | 0.049 | 2.17 |
| Per-row HTTP (1 POST/record) | 3662.07 | 3.662 | 0.636 |

Direct ONNX at 2.48ms is the irreducible floor. The 17.56ms gap to column-oriented HTTP is fixed cost: JSON deserialization, Pydantic validation, NumPy array construction. Row-oriented is 2.4× slower than column-oriented because `BACostCalculator` runs in a per-record Python loop post-inference rather than a vectorized NumPy pass.

The dashboard sends `df.to_dict(orient="list")` as a single column-oriented POST — the entire batch is one request, so concurrency is not a factor for the standard Streamlit use case.

---

## Limitations

- Revenue figures are proxied via haul-tier estimates, not real BA pricing data
- ROC-AUC ~0.82 reflects predicting intent from behavioral features only — no pricing, competitor, or browsing signals
- Email quality is bounded by ingested policy corpus coverage; the system does not guarantee factual accuracy of generated claims — human review required before sending
- Groq free-tier rate limits apply for large email generation batches

---

## Acknowledged Technical Debt

**No telemetry** — no Prometheus/Grafana for feature drift, no Arize/W&B for model performance tracking. Blind in production.

**No prompt caching** — system prompt and policy context resent on every `/api/v1/rag/generate` call. Migrating to a provider with prompt caching (Anthropic, DeepSeek) would cut per-request cost ~80% on the cached portion.

**ONNX blocking under concurrent load** — `threading.Lock` prevents session corruption but means concurrent batch requests serialize at the ONNX layer. Acceptable for single-user Streamlit; would require a worker-queue pattern (Celery + Redis) under multi-user load.

---

## Fine-Tuning Data Collection

- **`sft_log.jsonl`** — prompt/completion pairs for supervised fine-tuning. Populated on accepted/edited emails with rating ≥ 4, no contradictions.
- **`dpo_log.jsonl`** — chosen/rejected pairs for DPO. Original generation → `rejected`, human-edited version → `chosen`.

```python
from datasets import load_dataset
sft_data = load_dataset("json", data_files="data/finetuning/sft_log.jsonl")
dpo_data = load_dataset("json", data_files="data/finetuning/dpo_log.jsonl")
```

---

## Acknowledgements

- Dataset: British Airways customer booking data (BA Data Science Task 2)
- LLM: [Groq](https://groq.com) — Llama-4-Scout-17B | [Together AI](https://together.ai) — Llama-3.1-8B | [Ollama](https://ollama.com) — llama3.1:8b
- Embeddings: [sentence-transformers](https://www.sbert.net) — all-MiniLM-L6-v2
- Vector store: [ChromaDB](https://www.trychroma.com)
- ONNX conversion: [skl2onnx](https://onnx.ai/sklearn-onnx/) + [onnxmltools](https://github.com/onnx/onnxmltools)
- Package management: [uv](https://github.com/astral-sh/uv)

---

## License

MIT — see [LICENSE](LICENSE).