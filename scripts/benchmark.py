"""
Run: python -m scripts.benchmark
Runs a latency/memory benchmark comparing per-row HTTP vs vectorized batch inference.
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
import psutil

from src.config import XGBOOST_CONFIG_PATH

API_BASE   = "http://localhost:8000"
BATCH_SIZE = 1000
_PROCESS   = psutil.Process(os.getpid())

SEM = asyncio.Semaphore(4)


def load_config() -> dict:
    with open(XGBOOST_CONFIG_PATH) as f:
        return json.load(f)

async def _send_record(client, record):
    async with SEM:
        t0 = time.perf_counter()

        r = await client.post("/predict/single", json=record)
        r.raise_for_status()

        return time.perf_counter() - t0


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

def make_dummy_records_row_oriented(config: dict, n: int) -> list[dict]:
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

def make_dummy_records_column_oriented(config: dict, n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    records = {}

    for feat in config["expected_features"]:
        if feat in CATEGORICAL_VALUES:
            records[feat] = rng.choice(CATEGORICAL_VALUES[feat], size=n).tolist()
        elif feat in BINARY_FEATURES:
            records[feat] = rng.integers(0, 2, size=n).tolist()
        elif feat in INTEGER_FEATURES:
            lo, hi = FEATURE_RANGES.get(feat, (0, 100))
            records[feat] = rng.integers(lo, hi + 1, size=n).tolist()
        else:
            lo, hi = FEATURE_RANGES.get(feat, (0.0, 100.0))
            records[feat] = np.round(rng.uniform(lo, hi, size=n), 2).tolist()
    return records


# ── 1. per-row HTTP benchmark (legacy) ───────────────────────────────────────
async def bench_per_row(client: httpx.AsyncClient, records: list[dict]) -> dict:
    latencies = []
    tracemalloc.start()
    rss_before = _PROCESS.memory_info().rss

    for record in records:
        t0 = time.perf_counter()
        r = await client.post("/predict/single", json=record)
        r.raise_for_status()
        latencies.append(time.perf_counter() - t0)

    rss_delta_mb = round((_PROCESS.memory_info().rss - rss_before) / 1024**2, 3)
    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies = np.array(latencies) * 1000
    latencies.sort()
    n = len(latencies)
    return {
        "method": "per_row_http",
        "n": n,
        "total_ms": round(latencies.sum(), 2),
        "avg_ms": round(float(latencies.mean()), 4),
        "p95_ms": round(float(latencies[int(n * 0.95)]), 2),
        "p99_ms": round(float(latencies[int(n * 0.99)]), 2),
        "incremental_peak_mb": round(peak_tracemalloc / 1024**2, 3),
        "rss_delta_mb": rss_delta_mb,
    }

# ── 2. row-oriented HTTP benchmark (batch) ──────────────────────────────────
async def bench_row_oriented(client: httpx.AsyncClient, records: list[dict]) -> dict:
    tracemalloc.start()
    rss_before = _PROCESS.memory_info().rss
    t0 = time.perf_counter()

    r = await client.post("/predict/row_oriented", json=records)
    r.raise_for_status()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    rss_delta_mb = round((_PROCESS.memory_info().rss - rss_before) / 1024**2, 3)
    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "method": "row_oriented_http",
        "n": len(records),
        "total_ms": round(elapsed_ms, 2),
        "avg_ms": round(elapsed_ms / len(records), 4),
        "p95_ms": "n/a (single call)",
        "p99_ms": "n/a (single call)",
        "incremental_peak_mb": round(peak_tracemalloc / 1024**2, 3),
        "rss_delta_mb": rss_delta_mb,
    }

# ── 3. column-oriented HTTP benchmark ───────────────────────────────────────
async def bench_column_oriented(client: httpx.AsyncClient, records: dict[str, list]) -> dict:
    tracemalloc.start()
    rss_before = _PROCESS.memory_info().rss
    t0 = time.perf_counter()

    r = await client.post("/predict/column_oriented_bench", json=records)
    r.raise_for_status()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    rss_delta_mb = round((_PROCESS.memory_info().rss - rss_before) / 1024**2, 3)
    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "method": "column_oriented_http",
        "n": len(next(iter(records.values()))),
        "total_ms": round(elapsed_ms, 2),
        "avg_ms": round(elapsed_ms / len(next(iter(records.values()))), 4),
        "p95_ms": "n/a (single call)",
        "p99_ms": "n/a (single call)",
        "incremental_peak_mb": round(peak_tracemalloc / 1024**2, 3),
        "rss_delta_mb": rss_delta_mb,
    }

# ── 4. markdown writer ──────────────────────────────────────────────────────
def write_benchmark_md(results: dict):
    row_oriented     = results["row_oriented"]
    column_oriented  = results["column_oriented"]
    per_row          = results["per_row"]

    speedup_col_vs_row      = round(row_oriented["total_ms"] / column_oriented["total_ms"], 1)
    timestamp               = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# Inference Benchmark Results

> Generated: {timestamp}
> Rows benchmarked: {row_oriented['n']}
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
| Column-oriented HTTP | {column_oriented['total_ms']} | {column_oriented['avg_ms']} | — | — | {column_oriented['incremental_peak_mb']} | {column_oriented['rss_delta_mb']} |
| Row-oriented HTTP | {row_oriented['total_ms']} | {row_oriented['avg_ms']} | — | — | {row_oriented['incremental_peak_mb']} | {row_oriented['rss_delta_mb']} |
| Per-row HTTP (legacy) | {per_row['total_ms']} | {per_row['avg_ms']} | {per_row['p95_ms']} | {per_row['p99_ms']} | {per_row['incremental_peak_mb']} | {per_row['rss_delta_mb']} |

¹ `tracemalloc` — Python heap only. ONNX C++ allocations excluded.
² `psutil` RSS delta — OS-level. Includes C++ heap growth. Subject to page reclaim timing.

---

## Key Metrics

| Metric | Value |
|---|---|
| Speedup: row-oriented → column-oriented | **{speedup_col_vs_row}x** |

---

## Known Bottleneck

Row-oriented HTTP overhead is dominated by the per-record
`BACostCalculator.calculate_lead_value()` Python loop. Column-oriented resolves this
via `vectorized_calculate_lead_value()`. If row-oriented is still required (e.g. for
single-record streaming), the next optimization target is replacing the Python loop
with a batched NumPy pass post-inference before serializing results.
"""

    path = Path(__file__).resolve().parent.parent / "BENCHMARKS.md"
    path.write_text(md.strip(), encoding="utf-8")
    assert path.exists(), f"Write failed — {path} does not exist"
    print(f"\nWrote {path.resolve()} ({path.stat().st_size} bytes)")

# ── main ──────────────────────────────────────────────────────────────────────
async def main():
    config = load_config()
    records_row_oriented = make_dummy_records_row_oriented(config, BATCH_SIZE)
    records_column_oriented = make_dummy_records_column_oriented(config, BATCH_SIZE)

    print(f"\n=== Benchmarking {BATCH_SIZE} rows ===")
    print("Make sure FastAPI is running at", API_BASE)

    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
    async with httpx.AsyncClient(base_url=API_BASE, timeout=60, limits=limits) as client:
        r = await client.get("/health")
        r.raise_for_status()
        print("Warmup complete.\n")

        column_oriented = await bench_column_oriented(client, records_column_oriented)
        print("[column oriented http]", column_oriented)

        row_oriented = await bench_row_oriented(client, records_row_oriented)
        print("[row oriented http]   ", row_oriented)

        print("\n[per-row http] 1000 sequential requests — this will take a while...")
        per_row = await bench_per_row(client, records_row_oriented)
        print("[per-row http]        ", per_row)

    print("\n=== Summary ===")
    print(f"  Speedup (per-row vs batch):    {round(row_oriented['total_ms'] - column_oriented['total_ms'], 1)}ms  {round(row_oriented['total_ms'] / column_oriented['total_ms'], 1)}x")

    write_benchmark_md({
        "column_oriented": column_oriented,
        "row_oriented": row_oriented,
        "per_row": per_row,
    })

if __name__ == "__main__":
    asyncio.run(main())
