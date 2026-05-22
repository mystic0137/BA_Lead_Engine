# Inference Benchmark Results

> Generated: 2026-05-22 22:12:27
> Rows benchmarked: 1000
> Platform: FastAPI + ONNX Runtime (CPUExecutionProvider)
> Model: XGBoost → ONNX
> Connection: Single warmed-up keepalive (max_connections=1)

---

## Methods

| Method | Description |
|---|---|
| **Row-oriented HTTP** | Single POST to `/predict/row_oriented`. Records sent as a list of JSON objects. Per-row Python loop in `BACostCalculator`. |
| **Column-oriented HTTP** | Single POST to `/predict/column_oriented_bench`. Records sent as columnar arrays. Fully vectorized `BACostCalculator` via NumPy. |
| **Per-row HTTP (legacy)** | One POST to `/predict/single` per record, sequential. Included for historical reference only — not a meaningful comparison baseline. |

---

## Memory Measurement Notes

Two memory metrics are reported per method:

- **Incremental Peak (MB)** — Python heap delta measured by `tracemalloc` during the
  timed block only. Excludes ONNX runtime C++ allocations, pre-loaded model weights,
  and NumPy buffer handoffs to the runtime.

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
| Column-oriented HTTP | 14.3 | 0.0143 | — | — | 0.95 | 2.25 |
| Row-oriented HTTP | 106.91 | 0.1069 | — | — | 2.17 | 2.812 |
| Per-row HTTP (legacy) | 4028.07 | 4.0281 | 5.06 | 5.67 | 0.609 | 0.0 |

¹ `tracemalloc` — Python heap only. ONNX C++ allocations excluded.
² `psutil` RSS delta — OS-level. Includes C++ heap growth. Subject to page reclaim timing.

---

## Key Metrics

| Metric | Value |
|---|---|
| Speedup: row-oriented → column-oriented | **7.5x** |

---

## Known Bottleneck

Row-oriented HTTP overhead is dominated by the per-record
`BACostCalculator.calculate_lead_value()` Python loop. Column-oriented resolves this
via `vectorized_calculate_lead_value()`. If row-oriented is still required (e.g. for
single-record streaming), the next optimization target is replacing the Python loop
with a batched NumPy pass post-inference before serializing results.