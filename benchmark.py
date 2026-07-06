"""
Tremor acceleration benchmark — times the heavy stage (score + aggregate) on
whatever engine `pandas` currently resolves to.

    CPU:  python benchmark.py --data comments.parquet
    GPU:  python -m cudf.pandas benchmark.py --data comments.parquet

Run both, and the two runs write into the SAME benchmark_results.json so the
dashboard can show the speedup. Nothing in this file mentions cuDF — that is the
point: identical code, two engines.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import tremor_core as tc


def engine_name():
    # cudf.pandas installs a proxy module; detect it without importing cudf.
    if "cudf" in sys.modules or type(pd).__module__.startswith("cudf"):
        return "GPU (NVIDIA cuDF via cudf.pandas)"
    return "CPU (pandas)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="comments.parquet")
    ap.add_argument("--freq", default="10min")
    ap.add_argument("--out", default="benchmark_results.json")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    eng = engine_name()
    print(f"engine: {eng}")

    t0 = time.perf_counter()
    df = pd.read_parquet(args.data)
    t_load = time.perf_counter() - t0
    n_rows = len(df)
    print(f"loaded {n_rows:,} comments in {t_load:.3f}s")

    # warm-up (JIT/allocator) then timed runs
    _ = tc.score_and_aggregate(df.head(10_000).copy(), freq=args.freq)

    best = None
    for i in range(args.repeats):
        work = df.copy()  # fresh copy so we re-score every run
        t0 = time.perf_counter()
        agg = tc.score_and_aggregate(work, freq=args.freq)
        n_out = len(agg)
        dt = time.perf_counter() - t0
        print(f"  run {i+1}: score+aggregate {dt:.3f}s  ->  {n_out:,} buckets")
        best = dt if best is None else min(best, dt)

    throughput = n_rows / best if best else 0.0
    result = {
        "engine": eng,
        "rows": int(n_rows),
        "buckets": int(n_out),
        "load_s": round(t_load, 4),
        "process_s": round(best, 4),
        "rows_per_sec": int(throughput),
    }
    print(f"RESULT  {eng}: {n_rows:,} rows in {best:.3f}s "
          f"({throughput:,.0f} rows/s)")

    # merge into shared results file (CPU + GPU runs both land here)
    store = {}
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                store = json.load(f)
        except Exception:
            store = {}
    key = "gpu" if eng.startswith("GPU") else "cpu"
    store[key] = result
    if "cpu" in store and "gpu" in store and store["gpu"]["process_s"] > 0:
        store["speedup"] = round(store["cpu"]["process_s"] / store["gpu"]["process_s"], 1)
    with open(args.out, "w") as f:
        json.dump(store, f, indent=2)
    print(f"wrote {args.out}")
    if store.get("speedup"):
        print(f"SPEEDUP (CPU/GPU): {store['speedup']}x")


if __name__ == "__main__":
    main()
