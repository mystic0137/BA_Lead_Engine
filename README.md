# British Airways Lead Prioritization Engine

A production-grade ML system that predicts booking propensity, segments leads by revenue potential, and generates policy-grounded outreach emails — built as a portfolio project demonstrating end-to-end ML engineering.

---

## Overview

This project combines a booking prediction pipeline with a lead prioritization engine and an LLM-powered email generation system. It is designed around a real British Airways customer booking dataset and simulates how an airline's sales team might operationalize an ML model — from raw CSV upload to prioritized lead queue to personalized outreach email.

The system is split into two interconnected components:

**Component 1 — Booking Predictor**
A Random Forest and XGBoost classifier trained on customer booking behaviour, exported to ONNX for runtime-agnostic inference, and served via a FastAPI REST API.

**Component 2 — RAG Email Generator**
A Retrieval-Augmented Generation pipeline that retrieves relevant British Airways policy chunks from a ChromaDB vector store and generates compliance-grounded outreach emails via Llama-4-Scout on Groq.

Both components are surfaced through a Streamlit business intelligence dashboard designed for non-technical sales operators.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│  Batch Scoring │ Lead Queue │ Model Analysis │ Email Gen    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend                          │
│         /predict  │  /predict/batch  │  /health             │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐             ┌────────▼────────┐
│  ONNX Runtime   │             │  Lead Valuation  │
│  XGBoost Model  │             │  BACostCalculator│
│  (Inference)    │             │  (Segmentation)  │
└─────────────────┘             └─────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    RAG Email Pipeline                        │
│                                                             │
│  BA Policy Docs → ChromaDB → Retriever → Llama-4-Scout     │
│                                           (Groq API)        │
│                                               │             │
│                              Feedback Logger ─┘             │
│                    (sft_log.jsonl / dpo_log.jsonl)          │
└─────────────────────────────────────────────────────────────┘
```

---

**Data Integrity**
- To ensure **Reproducibility**, this system implements strict data integrity check. The raw dataset (`customer_booking.csv`) is locked via **SHA-256** checksum defined in (`src/config.py`). If the dataset is tampered with, corrupted, or modified, the pipeline will raise a `Data Integrity Violation` and halt.


---

## Key Decisions

**Why ONNX for models instead of Pickle(.pkl)**
- Standard Python .pkl are version dependent and pose security risks (Contain malicious code that can execute commands that pose threat to your system). By exporting XGBoost to ONNX, I've decoupled training environment from production API. This ensures sub millisecond latency and allows model to be served in any environment without python runtime dependency.

**RAG as a Token Budgeting Strategy**
- While BA policy documents are small to fit in a large context window. I implemented a ChromaDB-backed RAG pipeline to simulate production constraints. This serves as a Semantic Pre-filtering layer, reducing the prompt size from ~20k tokens to ~4k tokens. This 5x reduction in "token ingress" is critical for staying within the 30k TPM limits of the Groq API while maintaining high grounding accuracy.
- Whereas in production, LLMs often provide million-token context windows and Prompt Caching capabilities that could technically ingest the entire policy library at once. In that scenario, a complex RAG pipeline might seem like overkill; you could simply use Few-Shot Prompting to load the full text into the system prompt once and letting the model's internal attention mechanism handle the retrieval.

**The "Data Flywheel" logging**
- Rather than just logging raw outputs, the system captures Atomic State. Each log entry includes the specific RAG chunks retrieved, the system prompt version, and human-in-the-loop edits. This creates a high-fidelity dataset ready for Offline Evaluation and Parameter-Efficient Fine-Tuning (LoRA), ensuring the model's "sophisticated" tone can eventually be baked into the weights.

---

## Features

**ML Pipeline**
- Random Forest and XGBoost classifiers trained with sklearn pipelines
- Target encoding for high-cardinality categorical features (`route`, `booking_origin`, `flight_day`)
- One-hot encoding for low-cardinality categoricals (`sales_channel`, `trip_type`)
- Class imbalance handled via `scale_pos_weight` (XGBoost) and `class_weight="balanced"` (RF)
- Youden's J threshold optimization (threshold = 0.309)
- ONNX export for runtime-agnostic inference via `skl2onnx` + `onnxmltools`
- Per-model config JSON serialization (threshold, features, model type)

**Lead Segmentation**
Four segments derived from booking probability × lead value:

| Segment | Probability | Lead Value | Action |
|---|---|---|---|
| The Persuadable | Medium/Low | High | Priority call — highest ROI |
| The VIP | High | High | Automated nudge — no discount |
| The Window Shopper | High | Low | Email drip — let them book naturally |
| The Lost Cause | Low | Low | Suppression — no action |

Lead value calculated from flight duration tier (short/medium/long haul), add-on selections, and passenger count.

**RAG Email Generation**
- BA policy corpus ingested, chunked, embedded (`all-MiniLM-L6-v2`), and stored in ChromaDB
- Semantic retrieval at generation time — relevant policy chunks grounded in each email
- Prompt pre-processing maps raw lead data into narrative creative briefs before hitting the LLM
- Llama-4-Scout-17B via Groq API for generation
- Failsafe parser handles all observed LLM output format variations

**Human Feedback Loop**
- Accept/reject + star rating per generated email
- Auto-saves to three separate logs:
  - `feedback_log.jsonl` — full audit trail
  - `sft_log.jsonl` — accepted/edited emails for supervised fine-tuning
  - `dpo_log.jsonl` — chosen/rejected pairs for direct preference optimization
- Contradiction detection (e.g. high rating + reject) flagged in metadata, never silently dropped

**Dashboard**
- Batch CSV upload and scoring (vectorized ONNX inference — single `session.run` per batch)
- Lead queue with segment, value tier, and recommended action filters
- Model analysis tab: confusion matrix, ROC curve, probability distribution — threshold slider updates all metrics live
- Email generator tab: generate, review, edit, accept/reject, save feedback, open in Outlook as draft

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Training | scikit-learn, XGBoost |
| ONNX Export | skl2onnx, onnxmltools, onnxconverter-common |
| Inference | ONNX Runtime |
| API | FastAPI, Pydantic V2, Uvicorn |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Llama-4-Scout-17B via Groq API |
| Dashboard | Streamlit, Plotly |
| Validation | Pydantic V2 |
| Python | 3.10+ |

---

## Project Structure

```
british_airways_booking_predictor/
├── src/
│   ├── config.py               # All paths, constants, system prompt registry
│   ├── models.py               # ModelConfig, get_rf_model, get_xgb_model
│   ├── preprocess.py           # ColumnTransformer pipeline (TargetEncoder + OHE)
│   ├── train.py                # Training, ONNX export, config serialization
│   └── analytics/
│       └── finance.py          # BACostCalculator, lead segmentation logic
│   └── rag/
│       ├── ingest.py           # PDF/MD ingestion, chunking, embedding, ChromaDB storage
│       ├── retriever.py        # Semantic retrieval from ChromaDB
│       ├── prompt_builder.py   # Prompt construction from lead data + mappings
│       ├── email_generator.py  # Groq API call, failsafe parser
│       ├── mappings.py         # Segment/haul/amenity narrative maps, IATA resolver
│       └── feedback.py         # Feedback logging, SFT/DPO export, contradiction detection
├── app/
│   ├── main.py                 # FastAPI app, lifespan, batch inference endpoint
│   └── schemas.py              # Pydantic request/response models
├── frontend/
│   └── streamlit_app.py        # 4-tab Streamlit dashboard
├── tests/
│   ├── verify_performance.py   # ONNX model verification against test split
│   └── test_api.py             # FastAPI endpoint tests
├── data/
│   ├── raw/                    # customer_booking.csv
│   ├── policies/               # BA T&C markdown/PDF documents
│   ├── chroma_db/              # Persisted ChromaDB vector store
│   └── finetuning/             # feedback_log.jsonl, sft_log.jsonl, dpo_log.jsonl
├── models/                     # Trained ONNX models + config JSONs
├── pyproject.toml
└── Makefile
```

---

## Quickstart

**Prerequisites**
- Python 3.10+
- `make`
- A Groq API key (free tier sufficient)

**Setup**

```bash
# Clone the repo
git clone https://github.com/mystic0137/mlproject.git
cd BA_DS_Task_2/british_airways_booking_predictor

# Set your Groq API key
export GROQ_API_KEY=your_key_here

# Drop BA policy documents into data/policies/
# (PDF or markdown — see Limitations section)

# Run everything: install → ingest → train → serve
make
```

`make` will:
1. Install all dependencies from `pyproject.toml`
2. Ingest policy documents into ChromaDB
3. Train both models and export to ONNX
4. Start FastAPI on `http://127.0.0.1:8000`
5. Start Streamlit on `http://127.0.0.1:8501`

**Individual targets**

```bash
make install    # install dependencies only
make ingest     # ingest policy documents into ChromaDB
make train      # train models (skips if already trained)
make retrain    # force retrain
make verify     # run performance verification
make serve      # start FastAPI only
make ui         # start Streamlit only
make clean      # delete trained models
```

---

## Model Performance

Evaluated on a held-out 20% test split (stratified). Threshold optimized via Youden's J statistic.

| Model | ROC-AUC | Threshold |
|---|---|---|
| Random Forest | 0.7829 | 0.309 |
| XGBoost | 0.7828 | 0.309 |

The production API uses XGBoost. Both models are available for verification via `make verify`.

---

## Limitations

**Dataset**
- The British Airways customer booking dataset does not contain actual ticket prices. Revenue figures are proxied using haul-tier estimates (short haul $150, medium haul $400, long haul $800) and are illustrative only — not derived from real BA pricing data.
- Customer names, emails, and phone numbers in the dashboard are entirely synthetic. The dataset contains no real personal data.
- The dataset is a static snapshot. The model has no knowledge of seasonal pricing, route availability changes, or real-time demand signals.

**Model**
- ROC-AUC of ~0.78 reflects the inherent difficulty of predicting booking intent from behavioural features alone. The model does not have access to pricing, competitor offers, or browsing history.
- The 0.309 threshold was optimized on this specific dataset split. Performance on a different data distribution (e.g. different routes, booking periods) may vary.
- Class imbalance (~15% positive class) means precision on the positive class remains low (~0.38 for RF at the chosen threshold). The threshold trades precision for recall deliberately — the cost of missing a high-value lead exceeds the cost of a wasted outreach.

**RAG Pipeline**
- Email generation quality is bounded by the quality and coverage of the ingested policy documents. Gaps in the corpus (e.g. dynamic fare tables, route-specific promotions) will result in generic emails.
- The system retrieves semantically similar policy chunks — it does not guarantee factual accuracy of generated claims. Every email must be reviewed by a human before sending.
- The Groq free tier has rate limits. Generating emails for large batches may hit limits.

---

## Acknowledged Technical Debt
 - **Data Provenance & Drift:** Current logs reference local .md files. A production-hardened version would use Content-Addressing (SHA-256) for every retrieved chunk to ensure that fine-tuning data remains reproducible even if the underlying policy documents are updated.
 - **Stateless Inference vs. Prompt Caching:** The current pipeline resends the system prompt and policy context for every request. In a high-volume production environment, I would migrate to a provider supporting Prompt Caching (e.g., Anthropic or DeepSeek) to reduce costs by ~80%.
 - **Observability Gap:** The system lacks a dedicated telemetry layer. For a real deployment, I would integrate Prometheus/Grafana to monitor feature drift (e.g., shifting distributions in purchase_lead) and Arize/Weights & Biases for real-time model performance tracking.
 - **Concurrency Bottlenecks:** The FastAPI wrapper is async, but the ONNX Runtime calls are CPU-bound and blocking. High-scale usage would require a Worker-Queue pattern (Celery/Redis) to prevent event-loop starvation during batch scoring.

---

## Fine-Tuning Data Collection

The email generator accumulates training data as sales operators review generated emails:

- **`sft_log.jsonl`** — prompt/completion pairs for supervised fine-tuning. Populated when an email is accepted or edited with rating ≥ 4 and no contradictions detected.
- **`dpo_log.jsonl`** — chosen/rejected pairs for Direct Preference Optimization. Populated only when an email is edited — the original generation becomes the `rejected` completion, the human-edited version becomes `chosen`.

Both files are compatible with HuggingFace `datasets` and TRL's `SFTTrainer`/`DPOTrainer` directly:

```python
from datasets import load_dataset
sft_data = load_dataset("json", data_files="data/finetuning/sft_log.jsonl")
dpo_data = load_dataset("json", data_files="data/finetuning/dpo_log.jsonl")
```

---

## Acknowledgements

- Dataset: British Airways customer booking data (British Airways Data Science Task 2)
- LLM inference: [Groq](https://groq.com) — Llama-4-Scout-17B
- Embeddings: [sentence-transformers](https://www.sbert.net) — all-MiniLM-L6-v2
- Vector store: [ChromaDB](https://www.trychroma.com)
- ONNX conversion: [skl2onnx](https://onnx.ai/sklearn-onnx/) + [onnxmltools](https://github.com/onnx/onnxmltools)

---

## License

MIT