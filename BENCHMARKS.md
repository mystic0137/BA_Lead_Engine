# Inference Benchmark Results

> Generated: 2026-04-03 08:14:05
> Rows benchmarked: 1000
> Platform: FastAPI + ONNX Runtime (CPUExecutionProvider)
> Model: XGBoost → ONNX

---

## Memory Measurement Notes

Two memory metrics are reported per method:

- **Incremental Peak (MB)** — Python heap delta measured by `tracemalloc` during the
  timed block only. Excludes ONNX runtime C++ allocations, pre-loaded model weights,
  and numpy buffer handoffs to the runtime. The near-zero value for Direct ONNX reflects
  that the session and weights are already resident from the lifespan load — this is
  *additional* Python allocation only, not total footprint.

- **RSS Delta (MB)** — OS-level resident set size delta measured by `psutil` before and
  after the timed block. Captures C++ heap growth inside the ONNX runtime that
  `tracemalloc` cannot see. Subject to OS page reclaim timing — treat as an approximation,
  not a precise allocator trace.

Neither metric measures total process memory. For that, sample
`psutil.Process().memory_info().rss` at process start and peak.

---

## Results

| Method | Total (ms) | Avg/row (ms) | p95 (ms) | p99 (ms) | Incremental Peak (MB) ¹ | RSS Delta (MB) ² |
|---|---|---|---|---|---|---|
| Direct ONNX (no HTTP) | 2.43 | 0.0024 | — | — | 0.039 | 0.188 |
| Vectorized Batch HTTP | 156.73 | 0.1567 | — | — | 4.233 | 3.938 |
| Per-row HTTP (legacy) | 5681.27 | 5.6813 | 7.0737 | 7.8331 | 0.634 | 0.188 |

¹ `tracemalloc` — Python heap only. ONNX C++ allocations excluded.
² `psutil` RSS delta — OS-level. Includes C++ heap growth. Subject to page reclaim timing.

---

## Interpretation

### Direct ONNX — 2.43ms for 1000 rows
Irreducible model cost with zero HTTP overhead. ONNX runtime processes the entire
batch in a single `session.run()` call using vectorized matrix operations.
This is the performance floor — no approach can beat this number.

### Vectorized Batch HTTP — 156.73ms for 1000 rows
Single HTTP request carrying all 1000 records. The 154.3ms gap over
direct ONNX is fixed overhead: JSON deserialization, Pydantic validation across
all records, `pd.DataFrame` construction, and the per-row `BACostCalculator`
Python loop. The ONNX model itself accounts for only 2.43ms
(1.6%) of this total.

### Per-row HTTP (legacy) — 5681.27ms for 1000 rows
Each row paid the full HTTP round-trip independently: TCP overhead, Pydantic
validation, and ONNX dispatch fixed cost — multiplied 1000 times.
Abandoned in favour of the batch endpoint.

---

## Key Metrics

| Metric | Value |
|---|---|
| Speedup (per-row → batch) | **36.2x** |
| HTTP overhead over pure ONNX | **154.3ms** |
| Pure model cost (% of batch total) | **1.6%** |
| Wasted latency per 1k rows (legacy) | **5524.5ms** |

---

## Known Bottleneck

The 154.3ms HTTP overhead is currently dominated by the per-row
`BACostCalculator.calculate_lead_value()` Python loop. Vectorizing this
to operate on the full NumPy array at once is the next optimization target.
Expected to reduce batch HTTP latency from ~156.73ms to the 10–20ms range.