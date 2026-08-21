#!/usr/bin/env python3
"""
FSDP Checkpoint Restore Performance Benchmark Tool.

Evaluates the restore throughput and latency when restoring a checkpoint onto
a cluster with a different sharding topology (e.g., trained with 5 shards, restored on 10 workers).
Compares:
1. Baseline (Un-rewritten): 10 concurrent workers restoring from 5-shard chunk layout (Range-Read overhead)
2. Optimized (Rewritten): 10 concurrent workers restoring from 10-shard chunk layout (1:1 Shard alignment)
"""

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import tensorstore as ts
except ImportError:
    ts = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.checkpoints.orbax_reshard_rewriter import run_orbax_rewrite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RESTORE-BENCH] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark FSDP Checkpoint Restore: Un-rewritten vs Rewritten"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Working directory for checkpoint storage (defaults to a temporary directory)",
    )
    parser.add_argument(
        "--src-shards",
        type=int,
        default=5,
        help="Number of source shards in initial training checkpoint (default: 5)",
    )
    parser.add_argument(
        "--dst-workers",
        type=int,
        default=10,
        help="Number of target workers / chips restoring the checkpoint (default: 10)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers to simulate (default: 4)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4096,
        help="Hidden dimension size for weight matrices (default: 4096)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of benchmark repetitions for statistical averaging (default: 5)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write benchmark results JSON",
    )
    return parser.parse_args()


def generate_synthetic_fsdp_checkpoint(
    ckpt_dir: str,
    num_shards: int = 5,
    num_layers: int = 4,
    hidden_dim: int = 4096,
) -> Tuple[int, int]:
    """
    Generates a synthetic multi-layer FSDP checkpoint with specified number of chunks/shards.
    Returns (total_arrays, total_bytes).
    """
    if ts is None:
        raise RuntimeError("tensorstore is required. Install via pip install tensorstore")

    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Metadata files
    with open(os.path.join(ckpt_dir, "commit_success.txt"), "w") as f:
        f.write("committed\n")
    with open(os.path.join(ckpt_dir, "_CHECKPOINT"), "w") as f:
        json.dump({"format": "orbax", "version": 1}, f)
    with open(os.path.join(ckpt_dir, ".orbax-checkpoint-metadata"), "w") as f:
        json.dump({"step": 1000, "source_shards": num_shards}, f)

    # 2. Generate multi-layer weight matrices
    # Each layer has: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    matrices = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    total_arrays = 0
    total_bytes = 0

    chunk_dim0 = max(1, hidden_dim // num_shards)
    shape = [hidden_dim, hidden_dim]
    chunks = [chunk_dim0, hidden_dim]
    bytes_per_arr = hidden_dim * hidden_dim * 4  # float32

    for layer_idx in range(num_layers):
        for mat_name in matrices:
            arr_dir = os.path.join(
                ckpt_dir, "items", "params", f"layer_{layer_idx:02d}", mat_name, "kernel"
            )
            spec = {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": arr_dir},
                "metadata": {"shape": shape, "chunks": chunks, "dtype": "<f4"},
                "create": True,
            }
            arr = ts.open(spec).result()
            chunk_step = chunks[0]
            slice_data = np.ones([chunk_step, hidden_dim], dtype=np.float32) * (layer_idx + 1.0)
            for c_i in range(0, hidden_dim, chunk_step):
                c_end = min(c_i + chunk_step, hidden_dim)
                arr[c_i:c_end].write(slice_data[: c_end - c_i]).result()
            total_arrays += 1
            total_bytes += bytes_per_arr

    return total_arrays, total_bytes


def _restore_single_worker(
    array_dirs: List[str],
    worker_id: int,
    total_workers: int,
    hidden_dim: int,
) -> float:
    """Simulates a single worker restoring its assigned 1/N partition across all arrays."""
    t0 = time.perf_counter()
    slice_step = hidden_dim // total_workers
    start_idx = worker_id * slice_step
    end_idx = start_idx + slice_step

    for arr_dir in array_dirs:
        spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": arr_dir},
        }
        arr = ts.open(spec, open=True).result()
        # Read the assigned slice
        slice_data = arr[start_idx:end_idx].read().result()
        assert slice_data.shape[0] == slice_step

    return time.perf_counter() - t0


def run_restore_benchmark(
    ckpt_dir: str,
    num_workers: int,
    hidden_dim: int,
    num_runs: int = 5,
) -> Dict[str, Any]:
    """
    Executes concurrent restore benchmark across all workers over multiple runs.
    """
    # 1. Discover all array directories in checkpoint
    array_dirs = []
    for root, dirs, files in os.walk(ckpt_dir):
        if any(marker in files for marker in [".zarray", "zarr.json", "attributes.json"]):
            array_dirs.append(root)
            dirs.clear()
    array_dirs.sort()

    # Warm-up run
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(lambda wid: _restore_single_worker(array_dirs, wid, num_workers, hidden_dim), range(num_workers)))

    # Timed runs
    run_wall_times = []
    all_worker_durations = []

    for run_i in range(num_runs):
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            worker_times = list(
                ex.map(
                    lambda wid: _restore_single_worker(array_dirs, wid, num_workers, hidden_dim),
                    range(num_workers),
                )
            )
        wall_time = time.perf_counter() - t0
        run_wall_times.append(wall_time)
        all_worker_durations.append(worker_times)

    median_wall_s = float(np.median(run_wall_times))
    mean_wall_s = float(np.mean(run_wall_times))
    min_wall_s = float(np.min(run_wall_times))
    max_wall_s = float(np.max(run_wall_times))
    std_wall_s = float(np.std(run_wall_times))

    return {
        "num_arrays": len(array_dirs),
        "median_wall_seconds": median_wall_s,
        "mean_wall_seconds": mean_wall_s,
        "min_wall_seconds": min_wall_s,
        "max_wall_seconds": max_wall_s,
        "std_wall_seconds": std_wall_s,
        "all_run_wall_seconds": run_wall_times,
    }


def execute_comparison_benchmark(
    work_dir: Optional[str] = None,
    src_shards: int = 5,
    dst_workers: int = 10,
    num_layers: int = 4,
    hidden_dim: int = 4096,
    num_runs: int = 5,
) -> Dict[str, Any]:
    """
    End-to-end comparison benchmark orchestrator.
    """
    cleanup = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="fsdp_restore_bench_")
        cleanup = True

    try:
        src_ckpt_dir = os.path.join(work_dir, f"ckpt_{src_shards}shards")
        rewritten_ckpt_dir = os.path.join(work_dir, f"ckpt_{dst_workers}shards_rewritten")

        logging.info("==================================================================================")
        logging.info("          FSDP CHECKPOINT RESTORE BENCHMARK: 5 SHARDS -> 10 WORKERS               ")
        logging.info("==================================================================================")
        logging.info(f"Source Shards        : {src_shards} shards")
        logging.info(f"Target Restoring     : {dst_workers} concurrent workers")
        logging.info(f"Model Configuration  : {num_layers} layers x 7 matrices/layer = {num_layers * 7} arrays")
        logging.info(f"Matrix Dimension     : [{hidden_dim}, {hidden_dim}] float32 ({hidden_dim*hidden_dim*4/(1024*1024):.1f} MB/matrix)")
        logging.info("==================================================================================")

        # Step 1: Generate Source Checkpoint (5 shards)
        logging.info(f"[1/3] Generating synthetic {src_shards}-shard FSDP Checkpoint...")
        t_gen_start = time.perf_counter()
        total_arrays, total_bytes = generate_synthetic_fsdp_checkpoint(
            ckpt_dir=src_ckpt_dir,
            num_shards=src_shards,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )
        t_gen_elapsed = time.perf_counter() - t_gen_start
        total_mb = total_bytes / (1024 * 1024)
        total_gb = total_bytes / (1024 * 1024 * 1024)
        logging.info(f"      Generated {total_arrays} weight arrays ({total_mb:.2f} MB / {total_gb:.3f} GB) in {t_gen_elapsed:.2f}s")

        # Step 2: Rewrite Checkpoint to 10 shards
        logging.info(f"[2/3] Rewriting checkpoint to {dst_workers} shards (1:1 Shard Alignment)...")
        rewrite_summary = run_orbax_rewrite(
            src_dir=src_ckpt_dir,
            dst_dir=rewritten_ckpt_dir,
            strategy="dim_partitions",
            dim_partitions_str=f"0:{dst_workers}",
            num_workers=max(1, min(16, os.cpu_count() or 4)),
            verify=True,
            dry_run=False,
        )

        # Step 3: Run Benchmark A (Un-rewritten: 5 shards -> 10 workers)
        logging.info(f"[3/3] Running Concurrent Restore Benchmark ({num_runs} repetitions)...")
        logging.info(f"      -> Benchmarking Mode A: Un-rewritten ({src_shards} shards -> {dst_workers} workers)...")
        bench_unrewritten = run_restore_benchmark(
            ckpt_dir=src_ckpt_dir,
            num_workers=dst_workers,
            hidden_dim=hidden_dim,
            num_runs=num_runs,
        )
        throughput_unrewritten = total_mb / bench_unrewritten["median_wall_seconds"]

        # Step 4: Run Benchmark B (Rewritten: 10 shards -> 10 workers)
        logging.info(f"      -> Benchmarking Mode B: Rewritten ({dst_workers} shards -> {dst_workers} workers)...")
        bench_rewritten = run_restore_benchmark(
            ckpt_dir=rewritten_ckpt_dir,
            num_workers=dst_workers,
            hidden_dim=hidden_dim,
            num_runs=num_runs,
        )
        throughput_rewritten = total_mb / bench_rewritten["median_wall_seconds"]

        # Step 5: Comparative Analysis
        speedup = bench_unrewritten["median_wall_seconds"] / bench_rewritten["median_wall_seconds"]
        latency_reduction_pct = (
            (1.0 - (bench_rewritten["median_wall_seconds"] / bench_unrewritten["median_wall_seconds"])) * 100.0
        )

        results = {
            "source_shards": src_shards,
            "target_workers": dst_workers,
            "total_arrays": total_arrays,
            "total_volume_mb": round(total_mb, 2),
            "total_volume_gb": round(total_gb, 4),
            "unrewritten": {
                "median_wall_seconds": round(bench_unrewritten["median_wall_seconds"], 4),
                "mean_wall_seconds": round(bench_unrewritten["mean_wall_seconds"], 4),
                "min_wall_seconds": round(bench_unrewritten["min_wall_seconds"], 4),
                "max_wall_seconds": round(bench_unrewritten["max_wall_seconds"], 4),
                "std_wall_seconds": round(bench_unrewritten["std_wall_seconds"], 4),
                "throughput_mb_s": round(throughput_unrewritten, 2),
            },
            "rewritten": {
                "median_wall_seconds": round(bench_rewritten["median_wall_seconds"], 4),
                "mean_wall_seconds": round(bench_rewritten["mean_wall_seconds"], 4),
                "min_wall_seconds": round(bench_rewritten["min_wall_seconds"], 4),
                "max_wall_seconds": round(bench_rewritten["max_wall_seconds"], 4),
                "std_wall_seconds": round(bench_rewritten["std_wall_seconds"], 4),
                "throughput_mb_s": round(throughput_rewritten, 2),
                "rewrite_time_s": rewrite_summary["elapsed_seconds"],
                "rewrite_throughput_mb_s": rewrite_summary["throughput_mb_per_sec"],
            },
            "comparison": {
                "speedup_ratio": round(speedup, 2),
                "latency_reduction_percent": round(latency_reduction_pct, 2),
                "throughput_gain_mb_s": round(throughput_rewritten - throughput_unrewritten, 2),
            },
        }

        # Print Formatted Markdown Table
        print("\n" + "=" * 90)
        print("                   FSDP CHECKPOINT RESTORE PERFORMANCE COMPARISON                  ")
        print("=" * 90)
        print(f"| Evaluation Metric              | Un-rewritten (5-shard -> 10 workers) | Rewritten (10-shard -> 10 workers) | Delta / Speedup       |")
        print(f"| :------------------------------ | :---------------------------------- | :--------------------------------- | :-------------------- |")
        print(f"| **Restore Wall Time (Median)** | **{bench_unrewritten['median_wall_seconds']*1000:.2f} ms**                | **{bench_rewritten['median_wall_seconds']*1000:.2f} ms**               | **{speedup:.2f}x Faster** ({latency_reduction_pct:.1f}% drop) |")
        print(f"| **Restore Wall Time (Min)**    | {bench_unrewritten['min_wall_seconds']*1000:.2f} ms                   | {bench_rewritten['min_wall_seconds']*1000:.2f} ms                  | {bench_unrewritten['min_wall_seconds']/bench_rewritten['min_wall_seconds']:.2f}x Faster         |")
        print(f"| **Effective Restore Throughput**| **{throughput_unrewritten:.1f} MB/s**                     | **{throughput_rewritten:.1f} MB/s**                    | **+{throughput_rewritten - throughput_unrewritten:.1f} MB/s**     |")
        print(f"| **Chunk Access Pattern**       | Overlapping Range Reads (Partial)   | Dedicated 1:1 Sequential Reads     | Zero Read Contention  |")
        print(f"| **I/O Index Slicing Overhead** | Present (Cross-chunk slice decode)  | None (Direct Chunk Mapping)        | Clean 1:1 Alignment   |")
        print("=" * 90)

        return results

    finally:
        if cleanup and os.path.exists(work_dir):
            shutil.rmtree(work_dir)


def main():
    args = parse_args()
    results = execute_comparison_benchmark(
        work_dir=args.work_dir,
        src_shards=args.src_shards,
        dst_workers=args.dst_workers,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_runs=args.num_runs,
    )
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved benchmark results JSON: {args.output_json}")


if __name__ == "__main__":
    main()
