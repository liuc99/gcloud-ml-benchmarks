#!/usr/bin/env python3
"""
Orbax Checkpoint Resharding and Restore Benchmark Workload for GKE & GCS.

Simulates and evaluates:
1. Source Checkpoint Generation on GCS (e.g. 5 shards or 100 shards).
2. High-throughput bounded-memory CPU Rewrite to Target Sharding (e.g. 10 shards or 500 shards).
3. Multi-worker concurrent restore comparing un-rewritten (Range-Read storm) vs rewritten (1:1 sequential).
4. Outputting standardized logs for automated metric collection.
"""

import argparse
import concurrent.futures
import json
import logging
import math
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import tensorstore as ts
except ImportError:
    ts = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ORBAX-BENCHMARK] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Orbax Checkpoint Benchmark Workload on GKE")
    parser.add_argument("--mount-path", type=str, default="/gcs/checkpoints", help="Checkpoint mount path")
    parser.add_argument("--mode", type=str, default="compare", choices=["compare", "generate", "rewrite", "restore"], help="Benchmark execution mode")
    parser.add_argument("--src-shards", type=int, default=5, help="Number of source shards")
    parser.add_argument("--dst-workers", type=int, default=10, help="Number of target restoring workers")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension size")
    parser.add_argument("--strategy", type=str, default="dim_partitions", choices=["dim_partitions", "optimal_size", "unsharded"])
    parser.add_argument("--dim-partitions", type=str, default="0:10", help="Dimension partition mapping")
    parser.add_argument("--target-chunk-mb", type=float, default=64.0, help="Target MB per chunk")
    parser.add_argument("--strip-opt-state", action="store_true", help="Strip optimizer states")
    parser.add_argument("--cast-dtype", type=str, default="keep", choices=["keep", "bfloat16", "float16", "float32"])
    parser.add_argument("--num-runs", type=int, default=5, help="Benchmark repetitions")
    parser.add_argument("--node-rank", type=int, default=0, help="Node rank in multi-node job")
    parser.add_argument("--num-nodes", type=int, default=1, help="Total number of nodes")
    return parser.parse_args()


def generate_synthetic_checkpoint(
    ckpt_dir: str,
    num_shards: int = 5,
    num_layers: int = 4,
    hidden_dim: int = 4096,
) -> Tuple[int, int]:
    """Generates synthetic multi-layer FSDP checkpoint directly on GCS/POSIX."""
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "commit_success.txt"), "w") as f:
        f.write("committed\n")
    with open(os.path.join(ckpt_dir, "_CHECKPOINT"), "w") as f:
        json.dump({"format": "orbax", "version": 1, "source_shards": num_shards}, f)
    with open(os.path.join(ckpt_dir, ".orbax-checkpoint-metadata"), "w") as f:
        json.dump({"step": 1000, "timestamp": int(time.time()), "num_layers": num_layers}, f)

    matrices = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    total_arrays = 0
    total_bytes = 0
    shape = [hidden_dim, hidden_dim]
    chunk_dim0 = max(1, hidden_dim // num_shards)
    chunks = [chunk_dim0, hidden_dim]
    bytes_per_arr = hidden_dim * hidden_dim * 4

    for layer_i in range(num_layers):
        for mat in matrices:
            arr_dir = os.path.join(ckpt_dir, "items", "params", f"layer_{layer_i:02d}", mat, "kernel")
            if os.path.exists(os.path.join(arr_dir, ".zarray")):
                total_arrays += 1
                total_bytes += bytes_per_arr
                continue
            os.makedirs(arr_dir, exist_ok=True)
            spec = {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": arr_dir},
                "metadata": {"shape": shape, "chunks": chunks, "dtype": "<f4"},
                "create": True,
                "delete_existing": True,
            }
            arr = ts.open(spec).result()
            chunk_step = chunks[0]
            slice_data = np.ones([chunk_step, hidden_dim], dtype=np.float32) * (layer_i + 1.0)
            for c_i in range(0, hidden_dim, chunk_step):
                c_end = min(c_i + chunk_step, hidden_dim)
                arr[c_i:c_end].write(slice_data[: c_end - c_i]).result()
            total_arrays += 1
            total_bytes += bytes_per_arr

    return total_arrays, total_bytes


def rewrite_single_array(
    src_dir: str,
    dst_dir: str,
    target_chunks: List[int],
    cast_dtype: str = "keep",
) -> Dict[str, Any]:
    """Rewrites a single TensorStore array into destination with target chunking."""
    start_t = time.time()
    src_ts = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": src_dir}}, open=True).result()
    shape = list(src_ts.shape)
    src_meta = src_ts.spec().to_json().get("metadata", {})
    dtype_str = src_meta.get("dtype", "<f4")
    src_chunks = src_meta.get("chunks", list(shape))

    ts_target_dtype = None
    target_dtype_str = dtype_str
    if cast_dtype == "bfloat16":
        ts_target_dtype = getattr(ts, "bfloat16", None)
        target_dtype_str = "bfloat16"
    elif cast_dtype == "float16":
        ts_target_dtype = np.float16
        target_dtype_str = "<f2"

    os.makedirs(dst_dir, exist_ok=True)
    dst_spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": dst_dir},
        "metadata": {"shape": shape, "chunks": target_chunks, "dtype": target_dtype_str},
        "create": True,
        "delete_existing": True,
    }
    dst_ts = ts.open(dst_spec).result()

    dim0 = shape[0]
    step = target_chunks[0] if target_chunks else dim0
    for i in range(0, dim0, step):
        end = min(i + step, dim0)
        src_slice = src_ts[i:end]
        if ts_target_dtype is not None:
            dst_ts[i:end].write(ts.cast(src_slice, ts_target_dtype).read().result()).result()
        else:
            dst_ts[i:end].write(src_slice.read().result()).result()

    elapsed = time.time() - start_t
    return {"path": src_dir, "shape": shape, "src_chunks": src_chunks, "dst_chunks": target_chunks, "elapsed_s": elapsed}


def rewrite_checkpoint(
    src_ckpt: str,
    dst_ckpt: str,
    dst_shards: int = 10,
    hidden_dim: int = 4096,
    cast_dtype: str = "keep",
) -> float:
    """Orchestrates multi-threaded offline rewrite of all arrays in checkpoint."""
    t0 = time.time()
    os.makedirs(dst_ckpt, exist_ok=True)

    # Copy metadata
    for f in ["commit_success.txt", "_CHECKPOINT", ".orbax-checkpoint-metadata"]:
        src_f = os.path.join(src_ckpt, f)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(dst_ckpt, f))

    # Find arrays
    array_dirs = []
    for root, dirs, files in os.walk(src_ckpt):
        if any(m in files for m in [".zarray", "zarr.json", "attributes.json"]):
            array_dirs.append(root)
            dirs.clear()

    target_chunks = [max(1, hidden_dim // dst_shards), hidden_dim]

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
        futures = []
        for src_arr in array_dirs:
            rel = os.path.relpath(src_arr, src_ckpt)
            dst_arr = os.path.join(dst_ckpt, rel)
            futures.append(ex.submit(rewrite_single_array, src_arr, dst_arr, target_chunks, cast_dtype))
        concurrent.futures.wait(futures)

    return time.time() - t0


def restore_worker(array_dirs: List[str], worker_id: int, total_workers: int, hidden_dim: int) -> float:
    """Worker restore benchmark simulating 1/N partition reading."""
    t0 = time.perf_counter()
    step = hidden_dim // total_workers
    s_idx = worker_id * step
    e_idx = s_idx + step

    for arr_dir in array_dirs:
        arr = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": arr_dir}}).result()
        slice_data = arr[s_idx:e_idx].read().result()
        assert slice_data.shape[0] == step

    return time.perf_counter() - t0


def benchmark_restore(ckpt_dir: str, num_workers: int, hidden_dim: int, num_runs: int = 5) -> Tuple[float, float]:
    """Runs timed multi-worker restore benchmarks across num_runs repetitions."""
    array_dirs = []
    for root, dirs, files in os.walk(ckpt_dir):
        if any(m in files for m in [".zarray", "zarr.json", "attributes.json"]):
            array_dirs.append(root)
            dirs.clear()

    # Warmup
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(lambda wid: restore_worker(array_dirs, wid, num_workers, hidden_dim), range(num_workers)))

    run_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            list(ex.map(lambda wid: restore_worker(array_dirs, wid, num_workers, hidden_dim), range(num_workers)))
        run_times.append(time.perf_counter() - t0)

    median_s = float(np.median(run_times))
    return median_s, float(np.mean(run_times))


def main():
    args = parse_args()
    logging.info("==================================================================================")
    logging.info("       ORBAX CHECKPOINT RESHARDING & RESTORE BENCHMARK (GKE & GCS)                ")
    logging.info("==================================================================================")
    logging.info(f"Mount Path    : {args.mount_path}")
    logging.info(f"Mode          : {args.mode}")
    logging.info(f"Source Shards : {args.src_shards}")
    logging.info(f"Target Workers: {args.dst_workers}")
    logging.info(f"Model Layers  : {args.num_layers} layers x 7 matrices (Hidden Dim: {args.hidden_dim})")
    logging.info(f"Num Runs      : {args.num_runs}")
    logging.info("==================================================================================")

    src_ckpt = os.path.join(args.mount_path, f"source_ckpt_{args.src_shards}shards")
    dst_ckpt = os.path.join(args.mount_path, f"rewritten_ckpt_{args.dst_workers}shards")

    # 1. Generate Source Checkpoint
    logging.info("[Step 1/3] Generating Source Checkpoint on Storage...")
    t_start = time.time()
    num_arrays, total_bytes = generate_synthetic_checkpoint(src_ckpt, args.src_shards, args.num_layers, args.hidden_dim)
    gen_time = time.time() - t_start
    total_mb = total_bytes / (1024 * 1024)
    logging.info(f"[BENCHMARK] CHECKPOINT_GEN_TIME_SECONDS={gen_time:.2f}")
    logging.info(f"[BENCHMARK] TOTAL_CHECKPOINT_SIZE_MB={total_mb:.2f}")

    # 2. Rewrite Checkpoint
    logging.info("[Step 2/3] Performing Offline CPU Resharding to Target Layout...")
    rewrite_time = rewrite_checkpoint(src_ckpt, dst_ckpt, args.dst_workers, args.hidden_dim, args.cast_dtype)
    rewrite_mbps = total_mb / rewrite_time if rewrite_time > 0 else 0
    logging.info(f"[BENCHMARK] REWRITE_DURATION_SECONDS={rewrite_time:.2f}")
    logging.info(f"[BENCHMARK] REWRITE_THROUGHPUT_MBPS={rewrite_mbps:.2f}")

    # 3. Benchmark Restoring
    logging.info(f"[Step 3/3] Running Concurrent Restore Benchmarks ({args.num_runs} runs)...")
    
    # Un-rewritten
    unrewritten_median_s, _ = benchmark_restore(src_ckpt, args.dst_workers, args.hidden_dim, args.num_runs)
    unrewritten_mbps = total_mb / unrewritten_median_s
    logging.info(f"[BENCHMARK] UNREWRITTEN_RESTORE_TIME_SECONDS={unrewritten_median_s:.4f}")
    logging.info(f"[BENCHMARK] UNREWRITTEN_RESTORE_THROUGHPUT_MBPS={unrewritten_mbps:.2f}")

    # Rewritten
    rewritten_median_s, _ = benchmark_restore(dst_ckpt, args.dst_workers, args.hidden_dim, args.num_runs)
    rewritten_mbps = total_mb / rewritten_median_s
    logging.info(f"[BENCHMARK] REWRITTEN_RESTORE_TIME_SECONDS={rewritten_median_s:.4f}")
    logging.info(f"[BENCHMARK] REWRITTEN_RESTORE_THROUGHPUT_MBPS={rewritten_mbps:.2f}")

    speedup = unrewritten_median_s / rewritten_median_s
    latency_reduction = (1.0 - (rewritten_median_s / unrewritten_median_s)) * 100.0
    logging.info(f"[BENCHMARK] RESTORE_SPEEDUP_RATIO={speedup:.2f}")
    logging.info(f"[BENCHMARK] LATENCY_REDUCTION_PERCENT={latency_reduction:.2f}")

    logging.info("==================================================================================")
    logging.info("                      BENCHMARK EXECUTION SUMMARY                                 ")
    logging.info("==================================================================================")
    logging.info(f"Un-rewritten Restore Time : {unrewritten_median_s*1000:.2f} ms ({unrewritten_mbps:.1f} MB/s)")
    logging.info(f"Rewritten Restore Time    : {rewritten_median_s*1000:.2f} ms ({rewritten_mbps:.1f} MB/s)")
    logging.info(f"Performance Speedup       : {speedup:.2f}x Faster ({latency_reduction:.1f}% drop)")
    logging.info("==================================================================================")


if __name__ == "__main__":
    main()
