# Inference Benchmark Results

> Generated: 2026-05-21 11:21:23
> Rows benchmarked: 1000
> Platform: FastAPI + ONNX Runtime (CPUExecutionProvider)
> Model: XGBoost → ONNX
> Connection: Single warmed-up keepalive (max_connections=1)

---

## Methods

| Method | Description |
|---|---|
| **Direct ONNX** | `session.run()` called directly in-process — no HTTP, no Pydantic, no serialization. Performance floor. |
| **Row-oriented HTTP** | Single POST to `/predict/row_oriented`. Records sent as a list of JSON objects. Per-row Python loop in `BACostCalculator`. |
| **Column-oriented HTTP** | Single POST to `/predict/column_oriented`. Records sent as columnar arrays. Fully vectorized `BACostCalculator` via NumPy. |
| **Per-row HTTP (legacy)** | One POST to `/predict/single` per record, sequential. Included for historical reference only — not a meaningful comparison baseline. |

---

## Memory Measurement Notes

Two memory metrics are reported per method:

- **Incremental Peak (MB)** — Python heap delta measured by `tracemalloc` during the
  timed block only. Excludes ONNX runtime C++ allocations, pre-loaded model weights,
  and NumPy buffer handoffs to the runtime. Near-zero for Direct ONNX because the
  session and weights are already resident — this is *additional* Python allocation only.

- **RSS Delta (MB)** — OS-level resident set size delta measured by `psutil` before and
  after the timed block. Captures C++ heap growth inside the ONNX runtime that
  `tracemalloc` cannot see. Subject to OS page reclaim timing — treat as an
  approximation, not a precise allocator trace.

Neither metric measures total process memory. For that, sample
`psutil.Process().memory_info().rss` at process start and peak.

---

## Results

| Method | Total (ms) | Avg/row (ms) | p95 (ms) | p99 (ms) | Incremental Peak (MB) ¹ | RSS Delta (MB) ² |
|---|---|---|---|---|---|---|
| Direct ONNX | 3.16 | 0.0032 | — | — | 0.038 | 0.938 |
| Column-oriented HTTP | 21.54 | 0.0215 | — | — | 0.946 | 0.938 |
| Row-oriented HTTP | 58.84 | 0.0588 | — | — | 2.17 | 1.934 |
| Per-row HTTP (legacy) | 4587.07 | 4.5871 | 6.4762 | 8.0615 | 0.591 | 0.0 |

¹ `tracemalloc` — Python heap only. ONNX C++ allocations excluded.
² `psutil` RSS delta — OS-level. Includes C++ heap growth. Subject to page reclaim timing.

---

## Interpretation

### Direct ONNX — 3.16ms for 1000 rows
Irreducible model cost with zero HTTP overhead. ONNX runtime processes the entire
batch in a single `session.run()` call using vectorized matrix operations.
This is the performance floor — no HTTP method can beat this number.

### Column-oriented HTTP — 21.54ms for 1000 rows
Single POST to `/predict/column_oriented`. Columnar layout eliminates per-row
Python overhead in `BACostCalculator` — all valuation logic runs as NumPy
vectorized ops on the full array. The 18.38ms gap over Direct ONNX
is fixed overhead: JSON deserialization, Pydantic validation, and NumPy array
construction. The ONNX model accounts for 14.7% of this total.

### Row-oriented HTTP — 58.84ms for 1000 rows
Single POST to `/predict/row_oriented`. Same ONNX batch inference as column-oriented
but `BACostCalculator.calculate_lead_value()` runs in a Python loop per record.
The 55.68ms gap over Direct ONNX includes that loop cost on top of
serialization overhead. The ONNX model accounts for 5.4% of this total.

### Per-row HTTP (legacy) — 4587.07ms for 1000 rows
Included for historical context. One POST per record, sequential — every row paid
full FastAPI dispatch, Pydantic validation, and ONNX session fixed cost independently.
Not a meaningful comparison point against batch methods; the architectural difference
is categorical, not a tuning knob.

---

## Key Metrics

| Metric | Value |
|---|---|
| Speedup: row-oriented → column-oriented | **2.7x** |
| HTTP overhead over Direct ONNX (row-oriented) | **55.68ms** |
| HTTP overhead over Direct ONNX (column-oriented) | **18.38ms** |
| Pure model cost as % of row-oriented total | **5.4%** |
| Pure model cost as % of column-oriented total | **14.7%** |

---

## Known Bottleneck

Row-oriented HTTP overhead is dominated by the per-record
`BACostCalculator.calculate_lead_value()` Python loop. Column-oriented resolves this
via `vectorized_calculate_lead_value()`. If row-oriented is still required (e.g. for
single-record streaming), the next optimization target is replacing the Python loop
with a batched NumPy pass post-inference before serializing results.

## Previous Benchmark Issue

Previous Benchmark included cold start time of httpx connection. App sends sequential requests
so I limited max connections to 1, warmed up connection using health check before using it to 
measure for benchmark. Now benchmark numbers are more accurate than previous ones.