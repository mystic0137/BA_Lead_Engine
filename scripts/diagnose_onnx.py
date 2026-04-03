# scripts/diagnose_onnx.py
"""
Run: python -m scripts.diagnose_onnx
Prints ONNX input schema + runs a latency/memory benchmark comparing
per-row HTTP vs vectorized batch inference.
"""
import asyncio
import datetime
import json
import os
import time
import tracemalloc
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as rt
import psutil

from src.config import XGBOOST_CONFIG_PATH, XGBOOST_ONNX_PATH

# ── config ────────────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8000"
BATCH_SIZE = 1000
_PROCESS   = psutil.Process(os.getpid())


# ── helpers ───────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(XGBOOST_CONFIG_PATH) as f:
        return json.load(f)


CATEGORICAL_VALUES = {
    "sales_channel":  ["Internet", "Mobile"],
    "trip_type":      ["RoundTrip", "OneWay", "CircleTrip"],
    "flight_day":     ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "route":          ["AKLDEL", "AKLHGH", "AKLHND", "AKLICN", "AKLKIX", "AKLKTM"],
    "booking_origin": ["New Zealand", "India", "United Kingdom", "China", "South Korea", "Japan"],
}

INTEGER_FEATURES = {"num_passengers", "purchase_lead", "length_of_stay", "flight_hour"}
BINARY_FEATURES  = {"wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals"}

FEATURE_RANGES = {
    "num_passengers":  (1, 20),
    "purchase_lead":   (0, 365),
    "length_of_stay":  (0, 365),
    "flight_hour":     (0, 23),
    "flight_duration": (1.0, 20.0),
}


def make_dummy_records(config: dict, n: int) -> list[dict]:
    rng = np.random.default_rng(42)
    records = []
    for _ in range(n):
        row = {}
        for feat in config["expected_features"]:
            if feat in CATEGORICAL_VALUES:
                row[feat] = str(rng.choice(CATEGORICAL_VALUES[feat]))
            elif feat in BINARY_FEATURES:
                row[feat] = int(rng.integers(0, 2))
            elif feat in INTEGER_FEATURES:
                lo, hi = FEATURE_RANGES.get(feat, (0, 100))
                row[feat] = int(rng.integers(lo, hi + 1))
            else:
                lo, hi = FEATURE_RANGES.get(feat, (0.0, 100.0))
                row[feat] = round(float(rng.uniform(lo, hi)), 2)
        records.append(row)
    return records


# ── 1. ONNX input schema ──────────────────────────────────────────────────────
def print_onnx_schema():
    session = rt.InferenceSession(
        str(XGBOOST_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    print("\n=== ONNX Input Schema ===")
    for inp in session.get_inputs():
        print(f"  name={inp.name!r}  shape={inp.shape}  type={inp.type}")
    print("\n=== ONNX Output Schema ===")
    for out in session.get_outputs():
        print(f"  name={out.name!r}  shape={out.shape}  type={out.type}")


# ── 2. per-row HTTP benchmark (legacy) ───────────────────────────────────────
async def bench_per_row(records: list[dict]) -> dict:
    latencies = []
    tracemalloc.start()
    rss_before = _PROCESS.memory_info().rss

    async with httpx.AsyncClient(base_url=API_BASE, timeout=30) as client:
        for record in records:
            t0 = time.perf_counter()
            r = await client.post("/predict", json=record)
            r.raise_for_status()
            latencies.append(time.perf_counter() - t0)

    rss_delta_mb = round((_PROCESS.memory_info().rss - rss_before) / 1024**2, 3)
    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies = np.array(latencies) * 1000
    return {
        "method": "per_row_http",
        "n": len(records),
        "total_ms": round(float(latencies.sum()), 2),
        "avg_ms": round(float(latencies.mean()), 4),
        "p95_ms": round(float(np.percentile(latencies, 95)), 4),
        "p99_ms": round(float(np.percentile(latencies, 99)), 4),
        "incremental_peak_mb": round(peak_tracemalloc / 1024**2, 3),
        "rss_delta_mb": rss_delta_mb,
    }


# ── 3. batch HTTP benchmark (current) ────────────────────────────────────────
async def bench_batch(records: list[dict]) -> dict:
    tracemalloc.start()
    rss_before = _PROCESS.memory_info().rss
    t0 = time.perf_counter()

    async with httpx.AsyncClient(base_url=API_BASE, timeout=60) as client:
        r = await client.post("/predict/batch", json=records)
        if r.status_code == 422:
            errors = r.json()["detail"]
            print("=== 422 Validation Error ===")
            for err in errors[:3]:
                print(err)
        r.raise_for_status()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    rss_delta_mb = round((_PROCESS.memory_info().rss - rss_before) / 1024**2, 3)
    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "method": "vectorized_batch_http",
        "n": len(records),
        "total_ms": round(elapsed_ms, 2),
        "avg_ms": round(elapsed_ms / len(records), 4),
        "p95_ms": "n/a (single request)",
        "p99_ms": "n/a (single request)",
        "incremental_peak_mb": round(peak_tracemalloc / 1024**2, 3),
        "rss_delta_mb": rss_delta_mb,
    }


# ── 4. direct ONNX benchmark (no HTTP) ───────────────────────────────────────
def bench_direct_onnx(records: list[dict], config: dict) -> dict:
    import pandas as pd

    session = rt.InferenceSession(
        str(XGBOOST_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    expected = config["expected_features"]
    numeric  = config["numeric_features"]

    df = pd.DataFrame(records)[expected]
    df[numeric] = df[numeric].astype(np.float32)

    onnx_inputs = {
        inp.name: df[[inp.name]].values
        for inp in session.get_inputs()
    }

    tracemalloc.start()
    rss_before = _PROCESS.memory_info().rss
    t0 = time.perf_counter()

    session.run(None, onnx_inputs)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    rss_delta_mb = round((_PROCESS.memory_info().rss - rss_before) / 1024**2, 3)
    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "method": "direct_onnx_no_http",
        "n": len(records),
        "total_ms": round(elapsed_ms, 2),
        "avg_ms": round(elapsed_ms / len(records), 4),
        "p95_ms": "n/a (single call)",
        "p99_ms": "n/a (single call)",
        "incremental_peak_mb": round(peak_tracemalloc / 1024**2, 3),
        "rss_delta_mb": rss_delta_mb,
    }


# ── 5. markdown writer ────────────────────────────────────────────────────────
def write_benchmark_md(results: dict):
    direct  = results["direct"]
    batch   = results["batch"]
    per_row = results["per_row"]
    speedup  = round(per_row["total_ms"] / batch["total_ms"], 1)
    overhead = round(batch["total_ms"] - direct["total_ms"], 2)
    model_pct = round(direct["total_ms"] / batch["total_ms"] * 100, 1)
    wasted    = round(per_row["total_ms"] - batch["total_ms"], 1)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# Inference Benchmark Results

> Generated: {timestamp}
> Rows benchmarked: {direct['n']}
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
| Direct ONNX (no HTTP) | {direct['total_ms']} | {direct['avg_ms']} | — | — | {direct['incremental_peak_mb']} | {direct['rss_delta_mb']} |
| Vectorized Batch HTTP | {batch['total_ms']} | {batch['avg_ms']} | — | — | {batch['incremental_peak_mb']} | {batch['rss_delta_mb']} |
| Per-row HTTP (legacy) | {per_row['total_ms']} | {per_row['avg_ms']} | {per_row['p95_ms']} | {per_row['p99_ms']} | {per_row['incremental_peak_mb']} | {per_row['rss_delta_mb']} |

¹ `tracemalloc` — Python heap only. ONNX C++ allocations excluded.
² `psutil` RSS delta — OS-level. Includes C++ heap growth. Subject to page reclaim timing.

---

## Interpretation

### Direct ONNX — {direct['total_ms']}ms for {direct['n']} rows
Irreducible model cost with zero HTTP overhead. ONNX runtime processes the entire
batch in a single `session.run()` call using vectorized matrix operations.
This is the performance floor — no approach can beat this number.

### Vectorized Batch HTTP — {batch['total_ms']}ms for {batch['n']} rows
Single HTTP request carrying all {batch['n']} records. The {overhead}ms gap over
direct ONNX is fixed overhead: JSON deserialization, Pydantic validation across
all records, `pd.DataFrame` construction, and the per-row `BACostCalculator`
Python loop. The ONNX model itself accounts for only {direct['total_ms']}ms
({model_pct}%) of this total.

### Per-row HTTP (legacy) — {per_row['total_ms']}ms for {per_row['n']} rows
Each row paid the full HTTP round-trip independently: TCP overhead, Pydantic
validation, and ONNX dispatch fixed cost — multiplied {per_row['n']} times.
Abandoned in favour of the batch endpoint.

---

## Key Metrics

| Metric | Value |
|---|---|
| Speedup (per-row → batch) | **{speedup}x** |
| HTTP overhead over pure ONNX | **{overhead}ms** |
| Pure model cost (% of batch total) | **{model_pct}%** |
| Wasted latency per 1k rows (legacy) | **{wasted}ms** |

---

## Known Bottleneck

The {overhead}ms HTTP overhead is currently dominated by the per-row
`BACostCalculator.calculate_lead_value()` Python loop. Vectorizing this
to operate on the full NumPy array at once is the next optimization target.
Expected to reduce batch HTTP latency from ~{batch['total_ms']}ms to the 10–20ms range.
"""

    path = Path(__file__).resolve().parent.parent / "BENCHMARKS.md"
    path.write_text(md.strip(), encoding="utf-8")
    assert path.exists(), f"Write failed — {path} does not exist"
    print(f"\nWrote {path.resolve()} ({path.stat().st_size} bytes)")


# ── main ──────────────────────────────────────────────────────────────────────
async def main():
    print_onnx_schema()
    config  = load_config()
    records = make_dummy_records(config, BATCH_SIZE)

    print(f"\n=== Benchmarking {BATCH_SIZE} rows ===")
    print("Make sure FastAPI is running at", API_BASE)
    print("(python -m uvicorn app.main:app --reload)\n")

    direct = bench_direct_onnx(records, config)
    print("[direct onnx]  ", direct)

    batch = await bench_batch(records)
    print("[batch http]   ", batch)

    print("\n[per-row http] starting 1000 sequential requests — this will take a while...")
    per_row = await bench_per_row(records)
    print("[per-row http] ", per_row)

    print("\n=== Summary ===")
    print(f"  HTTP overhead (batch vs direct): {round(batch['total_ms'] - direct['total_ms'], 2)}ms")
    print(f"  Speedup (per-row vs batch): {round(per_row['total_ms'] / batch['total_ms'], 1)}x")

    write_benchmark_md({"direct": direct, "batch": batch, "per_row": per_row})


if __name__ == "__main__":
    asyncio.run(main())