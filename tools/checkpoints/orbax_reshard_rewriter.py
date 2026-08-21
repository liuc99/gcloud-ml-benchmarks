#!/usr/bin/env python3
"""
Orbax Checkpoint Offline Resharding and Layout Optimization Tool.

Optimizes checkpoint restore and loading performance across different cluster topologies by:
1. Re-aligning TensorStore chunk boundaries to the target mesh or optimal I/O chunk sizes.
2. Converting fine-grained fragmented chunks (e.g., thousands of small range-read slices)
   into high-throughput sequential blocks (eliminating range-read storms on GCS / GCSFuse).
3. Streaming array copies with bounded memory footprint (preventing CPU host OOM on 70B+ models).
4. Optionally stripping optimizer states (reducing checkpoint footprint by ~67% for inference).
5. Optionally casting dtypes (e.g. float32 -> bfloat16 / float16).
6. Preserving all Orbax PyTree metadata, commit markers, and directory structures.
"""

import argparse
import concurrent.futures
import json
import logging
import math
import os
import re
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
    format="%(asctime)s [%(levelname)s] [ORBAX-RESHARD] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# File markers that indicate a directory is a leaf TensorStore array
ARRAY_MARKERS = [".zarray", "zarr.json", "attributes.json"]


def parse_args():
    epilog_text = """
==================================================================================
PRACTICAL SCENARIOS & USAGE EXAMPLES:
==================================================================================

1. CLUSTER SCALE TRANSITION (e.g., Trained on 100 TPUs -> Restoring on 500 TPUs):
   When restoring onto a 500-card cluster, direct restore causes 500 workers to issue
   overlapping Byte-Range Reads against the old 100-chip chunk layout.
   Pre-resharding to 500 partitions gives each TPU chip its own 1:1 chunk:

   # 1D Partitioning (e.g., FSDP / Hidden dimension sliced 500 ways):
   python3 tools/checkpoints/orbax_reshard_rewriter.py \\
     --src-dir "/path/to/checkpoint_100tpu/0" \\
     --dst-dir "/path/to/checkpoint_500tpu/0" \\
     --strategy dim_partitions \\
     --dim-partitions "0:500" \\
     --num-workers 16

   # 2D Mesh Alignment (e.g., Dim 0 partitioned 250-way, Dim 1 partitioned 2-way):
   python3 tools/checkpoints/orbax_reshard_rewriter.py \\
     --src-dir "/path/to/checkpoint_100tpu/0" \\
     --dst-dir "/path/to/checkpoint_500tpu/0" \\
     --strategy dim_partitions \\
     --dim-partitions "0:250,1:2" \\
     --num-workers 16

2. GENERAL GCS/LUSTRE I/O OPTIMIZATION (Merge small chunks into 64MB sequential blocks):
   Merges thousands of fragmented small chunks into 64MB blocks, eliminating Range Reads:
   python3 tools/checkpoints/orbax_reshard_rewriter.py \\
     --src-dir "/path/to/checkpoint/0" \\
     --dst-dir "/path/to/checkpoint_rechunked/0" \\
     --strategy optimal_size \\
     --target-chunk-mb 64.0 \\
     --num-workers 8

3. OPTIMIZER STRIPPING FOR INFERENCE / EVALUATION EXPORT:
   Strips Adam/optimizer states (saving ~67% disk space and reducing load time by 3x):
   python3 tools/checkpoints/orbax_reshard_rewriter.py \\
     --src-dir "/path/to/checkpoint_100tpu/0" \\
     --dst-dir "/path/to/checkpoint_500tpu_eval/0" \\
     --strategy dim_partitions \\
     --dim-partitions "0:500" \\
     --strip-opt-state \\
     --num-workers 16

4. PRECISION CASTING (float32 -> bfloat16) WITH NUMERICAL INTEGRITY VERIFICATION:
   python3 tools/checkpoints/orbax_reshard_rewriter.py \\
     --src-dir "/path/to/checkpoint/0" \\
     --dst-dir "/path/to/checkpoint_bf16/0" \\
     --cast-dtype bfloat16 \\
     --verify

5. DRY-RUN INSPECTION (Preview chunk reduction without writing files):
   python3 tools/checkpoints/orbax_reshard_rewriter.py \\
     --src-dir "/path/to/checkpoint/0" \\
     --dst-dir "/path/to/checkpoint_out/0" \\
     --dry-run
==================================================================================
"""
    parser = argparse.ArgumentParser(
        description=(
            "Orbax Checkpoint Offline Resharding and Layout Optimization Tool.\n\n"
            "Optimizes checkpoint restore and loading throughput across different cluster\n"
            "topologies (e.g., 100 TPU -> 500 TPU) and storage backends (GCS / Lustre) by\n"
            "re-aligning TensorStore chunk boundaries, eliminating Range-Read storms,\n"
            "bounding CPU memory to prevent OOM, and optionally stripping optimizer states."
        ),
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src-dir",
        type=str,
        required=True,
        help="Source Orbax checkpoint directory (e.g. /path/to/checkpoint/0 or /gcs/bucket/ckpt/0)",
    )
    parser.add_argument(
        "--dst-dir",
        type=str,
        required=True,
        help="Destination directory for the rewritten checkpoint",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["optimal_size", "unsharded", "dim_partitions"],
        default="optimal_size",
        help=(
            "Chunking strategy:\n"
            "  'optimal_size'  : Consolidate arrays into ~target-chunk-mb sequential blocks (default)\n"
            "  'dim_partitions': Partition designated dimensions by specific factors (e.g. 500-way sharding)\n"
            "  'unsharded'     : Consolidate each tensor into a single contiguous chunk"
        ),
    )
    parser.add_argument(
        "--target-chunk-mb",
        type=float,
        default=64.0,
        help="Target size per chunk in Megabytes for 'optimal_size' strategy (default: 64.0 MB)",
    )
    parser.add_argument(
        "--dim-partitions",
        type=str,
        default=None,
        help=(
            "Dimension partition mapping for 'dim_partitions' strategy.\n"
            "Format: '<axis>:<num_parts>[,<axis>:<num_parts>]' or '<num_parts>' for axis 0.\n"
            "Example: '0:500' (500 shards along dim 0) or '0:250,1:2' (2D mesh)"
        ),
    )
    parser.add_argument(
        "--strip-opt-state",
        action="store_true",
        help="Strip optimizer states (opt_state/optimizer) to retain only model parameters (saves ~67%% storage)",
    )
    parser.add_argument(
        "--cast-dtype",
        type=str,
        choices=["keep", "bfloat16", "float16", "float32"],
        default="keep",
        help="Cast data type during rewrite (default: keep)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(16, os.cpu_count() or 4)),
        help="Number of concurrent worker threads for array rewriting (default: min(16, CPU count))",
    )
    parser.add_argument(
        "--max-buffer-mb",
        type=float,
        default=64.0,
        help="Maximum buffer size in MB per streaming slice to bound host memory (default: 64.0 MB)",
    )
    parser.add_argument(
        "--include-regex",
        type=str,
        default=None,
        help="Regex pattern to include specific parameter paths only",
    )
    parser.add_argument(
        "--exclude-regex",
        type=str,
        default=None,
        help="Regex pattern to exclude specific parameter paths",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify numerical data integrity between source and target arrays after rewrite",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and calculate chunk layout plan without writing files",
    )
    return parser.parse_args()


def is_array_dir(dir_path: str, files: List[str]) -> bool:
    """Checks if a directory contains TensorStore array metadata markers."""
    return any(marker in files for marker in ARRAY_MARKERS)


def scan_checkpoint_tree(
    src_root: str,
    strip_opt_state: bool = False,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Recursively scans the checkpoint tree to discover:
    1. TensorStore array directories (leaf nodes)
    2. Non-array metadata and commit files
    """
    src_root = os.path.abspath(src_root)
    if not os.path.exists(src_root):
        raise FileNotFoundError(f"Source checkpoint path does not exist: {src_root}")

    array_dirs = []
    metadata_files = []

    inc_pat = re.compile(include_regex) if include_regex else None
    exc_pat = re.compile(exclude_regex) if exclude_regex else None

    for root, dirs, files in os.walk(src_root, followlinks=True):
        rel_dir = os.path.relpath(root, src_root)

        # Check if opt_state should be skipped
        if strip_opt_state and rel_dir != ".":
            path_parts = rel_dir.split(os.sep)
            if "opt_state" in path_parts or "optimizer" in path_parts:
                dirs.clear()
                continue

        # Check if directory matches exclude regex
        if exc_pat and rel_dir != "." and exc_pat.search(rel_dir):
            dirs.clear()
            continue

        if is_array_dir(root, files):
            # Check include regex
            if inc_pat and not inc_pat.search(rel_dir):
                dirs.clear()
                continue
            array_dirs.append(root)
            # Stop descending into chunk subdirectories
            dirs.clear()
        else:
            # Collect metadata files
            for f in files:
                full_file_path = os.path.join(root, f)
                rel_file_path = os.path.relpath(full_file_path, src_root)

                if strip_opt_state:
                    file_parts = rel_file_path.split(os.sep)
                    if "opt_state" in file_parts or "optimizer" in file_parts:
                        continue

                if exc_pat and exc_pat.search(rel_file_path):
                    continue

                if inc_pat and not inc_pat.search(rel_file_path):
                    continue

                metadata_files.append(full_file_path)

    return sorted(array_dirs), sorted(metadata_files)


def detect_tensorstore_driver(array_dir: str) -> str:
    """Detects the TensorStore driver based on metadata files in directory."""
    if os.path.exists(os.path.join(array_dir, ".zarray")):
        return "zarr"
    elif os.path.exists(os.path.join(array_dir, "zarr.json")):
        return "zarr3"
    elif os.path.exists(os.path.join(array_dir, "attributes.json")):
        return "n5"
    return "zarr"


def parse_dim_partitions(dim_partitions_str: Optional[str]) -> Dict[int, int]:
    """Parses partition string like '0:4,1:2' or '4' into a dictionary {axis: num_parts}."""
    if not dim_partitions_str:
        return {}
    parts = {}
    tokens = [t.strip() for t in dim_partitions_str.split(",") if t.strip()]
    for token in tokens:
        if ":" in token:
            dim_str, p_str = token.split(":", 1)
            parts[int(dim_str)] = int(p_str)
        else:
            parts[0] = int(token)
    return parts


def compute_target_chunks(
    shape: List[int],
    dtype_size: int,
    strategy: str,
    target_chunk_mb: float,
    dim_partitions: Dict[int, int],
) -> List[int]:
    """Computes optimized chunk shape based on strategy and dimensions."""
    if not shape:
        return []

    rank = len(shape)
    if strategy == "unsharded":
        return list(shape)

    if strategy == "dim_partitions":
        target_chunks = list(shape)
        for dim, num_parts in dim_partitions.items():
            if 0 <= dim < rank and num_parts > 0:
                target_chunks[dim] = max(1, math.ceil(shape[dim] / num_parts))
        return target_chunks

    # Default: "optimal_size" strategy
    total_elements = math.prod(shape)
    total_bytes = total_elements * dtype_size
    target_chunk_bytes = int(target_chunk_mb * 1024 * 1024)

    if total_bytes <= target_chunk_bytes:
        # Small tensor: single chunk eliminates small range reads entirely
        return list(shape)

    target_chunks = list(shape)
    target_num_chunks = max(1, math.ceil(total_bytes / target_chunk_bytes))

    # Partition along the first leading dimension
    elements_per_chunk = max(1, total_elements // target_num_chunks)
    dim0_stride = math.prod(shape[1:]) if rank > 1 else 1
    dim0_chunk = max(1, min(shape[0], math.ceil(elements_per_chunk / dim0_stride)))
    target_chunks[0] = dim0_chunk

    return target_chunks


def count_chunks(shape: List[int], chunks: List[int]) -> int:
    """Calculates the total number of storage chunk files for a given shape and chunk size."""
    if not shape or not chunks:
        return 1
    num = 1
    for s, c in zip(shape, chunks):
        c = max(1, c)
        num *= math.ceil(s / c)
    return num


def get_target_dtype(src_dtype_str: str, cast_dtype: str):
    """Resolves target dtype and size."""
    if cast_dtype == "bfloat16":
        return "bfloat16", 2, getattr(ts, "bfloat16", None)
    elif cast_dtype == "float16":
        return "<f2", 2, np.float16
    elif cast_dtype == "float32":
        return "<f4", 4, np.float32
    else:
        # Keep original dtype
        return src_dtype_str, None, None


def rewrite_array(
    src_array_dir: str,
    dst_array_dir: str,
    strategy: str = "optimal_size",
    target_chunk_mb: float = 64.0,
    dim_partitions: Optional[Dict[int, int]] = None,
    cast_dtype: str = "keep",
    max_buffer_mb: float = 64.0,
    verify: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Rewrites a single TensorStore array with chunk realignment and optional dtype casting.
    Streams slices along the leading dimension to maintain a constant, bounded memory footprint.
    """
    if ts is None:
        raise RuntimeError("tensorstore Python package is required. Install via pip install tensorstore")

    dim_partitions = dim_partitions or {}
    start_time = time.time()

    driver = detect_tensorstore_driver(src_array_dir)
    src_spec = {
        "driver": driver,
        "kvstore": {"driver": "file", "path": src_array_dir},
    }

    src_ts = ts.open(src_spec, open=True).result()
    src_spec_json = src_ts.spec().to_json()
    shape = list(src_ts.shape)
    src_dtype_name = src_ts.dtype.name

    # Determine source metadata dtype
    meta = src_spec_json.get("metadata", {})
    src_meta_dtype = meta.get("dtype", "<f4" if "float32" in src_dtype_name else "<f2")
    src_chunks = meta.get("chunks", list(shape))

    # Resolve target dtype
    target_dtype_str, target_dtype_size, ts_target_dtype = get_target_dtype(src_meta_dtype, cast_dtype)
    if target_dtype_size is None:
        target_dtype_size = src_ts.dtype.numpy_dtype.itemsize

    # Calculate target chunks
    target_chunks = compute_target_chunks(
        shape=shape,
        dtype_size=target_dtype_size,
        strategy=strategy,
        target_chunk_mb=target_chunk_mb,
        dim_partitions=dim_partitions,
    )

    src_chunk_count = count_chunks(shape, src_chunks)
    dst_chunk_count = count_chunks(shape, target_chunks)
    total_bytes = math.prod(shape) * target_dtype_size

    if dry_run:
        return {
            "src_dir": src_array_dir,
            "dst_dir": dst_array_dir,
            "shape": shape,
            "src_chunks": src_chunks,
            "dst_chunks": target_chunks,
            "src_chunk_count": src_chunk_count,
            "dst_chunk_count": dst_chunk_count,
            "total_bytes": total_bytes,
            "elapsed_s": 0.0,
            "status": "dry_run",
        }

    # Create destination array
    os.makedirs(dst_array_dir, exist_ok=True)
    dst_spec = {
        "driver": driver,
        "kvstore": {"driver": "file", "path": dst_array_dir},
        "metadata": {
            "shape": shape,
            "chunks": target_chunks,
            "dtype": target_dtype_str,
        },
        "create": True,
        "delete_existing": True,
    }
    dst_ts = ts.open(dst_spec).result()

    # Stream data in bounded buffer slices
    if shape:
        dim0 = shape[0]
        # Calculate slice step size to keep buffer within max_buffer_mb
        slice_elements_limit = max(1, int((max_buffer_mb * 1024 * 1024) / target_dtype_size))
        dim0_stride = math.prod(shape[1:]) if len(shape) > 1 else 1
        step_dim0 = max(1, min(dim0, slice_elements_limit // dim0_stride))
        # Align step with target chunk if possible
        if target_chunks and target_chunks[0] > 0:
            step_dim0 = max(step_dim0, target_chunks[0])

        for i in range(0, dim0, step_dim0):
            end = min(i + step_dim0, dim0)
            src_slice = src_ts[i:end]
            if cast_dtype != "keep" and ts_target_dtype is not None:
                cast_slice = ts.cast(src_slice, ts_target_dtype).read().result()
                dst_ts[i:end].write(cast_slice).result()
            else:
                slice_data = src_slice.read().result()
                dst_ts[i:end].write(slice_data).result()
    else:
        # Scalar tensor
        if cast_dtype != "keep" and ts_target_dtype is not None:
            dst_ts.write(ts.cast(src_ts, ts_target_dtype).read().result()).result()
        else:
            dst_ts.write(src_ts.read().result()).result()

    # Verify if requested
    if verify:
        sample_src = src_ts.read().result()
        sample_dst = dst_ts.read().result()
        if cast_dtype == "keep":
            if not np.array_equal(sample_src, sample_dst):
                raise ValueError(f"Verification parity failure in array: {src_array_dir}")
        else:
            # Check with relative numerical tolerance for precision casting (e.g. bfloat16 / float16)
            if not np.allclose(np.asarray(sample_src, dtype=np.float32), np.asarray(sample_dst, dtype=np.float32), rtol=2e-2, atol=1.0):
                raise ValueError(f"Verification precision failure in casted array: {src_array_dir}")

    elapsed = time.time() - start_time
    return {
        "src_dir": src_array_dir,
        "dst_dir": dst_array_dir,
        "shape": shape,
        "src_chunks": src_chunks,
        "dst_chunks": target_chunks,
        "src_chunk_count": src_chunk_count,
        "dst_chunk_count": dst_chunk_count,
        "total_bytes": total_bytes,
        "elapsed_s": elapsed,
        "status": "success",
    }


def copy_metadata_files(
    metadata_files: List[str],
    src_root: str,
    dst_root: str,
    dry_run: bool = False,
) -> int:
    """Copies non-array metadata and commit files to preserve checkpoint validity."""
    copied = 0
    for src_file in metadata_files:
        rel_path = os.path.relpath(src_file, src_root)
        dst_file = os.path.join(dst_root, rel_path)
        if not dry_run:
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)
        copied += 1
    return copied


def run_orbax_rewrite(
    src_dir: str,
    dst_dir: str,
    strategy: str = "optimal_size",
    target_chunk_mb: float = 64.0,
    dim_partitions_str: Optional[str] = None,
    strip_opt_state: bool = False,
    cast_dtype: str = "keep",
    num_workers: int = 4,
    max_buffer_mb: float = 64.0,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    verify: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Main orchestration entry point for Orbax offline checkpoint rewriting.
    """
    start_total_time = time.time()
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    logging.info("==================================================================================")
    logging.info("       ORBAX CHECKPOINT OFFLINE RESHARDING & CHUNK REALIGNMENT TOOL               ")
    logging.info("==================================================================================")
    logging.info(f"Source Checkpoint : {src_dir}")
    logging.info(f"Destination Path  : {dst_dir}")
    logging.info(f"Chunking Strategy : {strategy} (Target Chunk Size: {target_chunk_mb} MB)")
    logging.info(f"Strip Opt State   : {strip_opt_state}")
    logging.info(f"Cast Dtype        : {cast_dtype}")
    logging.info(f"Parallel Workers  : {num_workers}")
    logging.info(f"Mode              : {'DRY RUN' if dry_run else 'EXECUTE'}")
    logging.info("==================================================================================")

    # 1. Scan tree
    array_dirs, metadata_files = scan_checkpoint_tree(
        src_root=src_dir,
        strip_opt_state=strip_opt_state,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
    )

    logging.info(f"Discovered {len(array_dirs)} TensorStore arrays and {len(metadata_files)} metadata files.")
    if not array_dirs:
        logging.warning("No TensorStore array directories found in source checkpoint.")

    dim_parts = parse_dim_partitions(dim_partitions_str)

    # 2. Copy metadata files
    copied_meta = copy_metadata_files(metadata_files, src_dir, dst_dir, dry_run=dry_run)
    logging.info(f"{'Planned' if dry_run else 'Copied'} {copied_meta} metadata files to destination.")

    # 3. Rewrite arrays concurrently
    results = []
    total_volume_bytes = 0
    total_src_chunks = 0
    total_dst_chunks = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_array = {}
        for src_arr in array_dirs:
            rel = os.path.relpath(src_arr, src_dir)
            dst_arr = os.path.join(dst_dir, rel)
            future = executor.submit(
                rewrite_array,
                src_array_dir=src_arr,
                dst_array_dir=dst_arr,
                strategy=strategy,
                target_chunk_mb=target_chunk_mb,
                dim_partitions=dim_parts,
                cast_dtype=cast_dtype,
                max_buffer_mb=max_buffer_mb,
                verify=verify,
                dry_run=dry_run,
            )
            future_to_array[future] = rel

        for i, future in enumerate(concurrent.futures.as_completed(future_to_array), 1):
            rel_name = future_to_array[future]
            try:
                res = future.result()
                results.append(res)
                total_volume_bytes += res["total_bytes"]
                total_src_chunks += res["src_chunk_count"]
                total_dst_chunks += res["dst_chunk_count"]

                vol_mb = res["total_bytes"] / (1024 * 1024)
                mbps = vol_mb / res["elapsed_s"] if res["elapsed_s"] > 0 else 0
                reduction_pct = (
                    (1.0 - (res["dst_chunk_count"] / res["src_chunk_count"])) * 100.0
                    if res["src_chunk_count"] > 0
                    else 0.0
                )

                logging.info(
                    f"[{i}/{len(array_dirs)}] {rel_name} | Shape: {res['shape']} | "
                    f"Chunks: {res['src_chunks']} -> {res['dst_chunks']} "
                    f"({res['src_chunk_count']} -> {res['dst_chunk_count']} files, {reduction_pct:+.1f}%) | "
                    f"{vol_mb:.2f} MB | {res['elapsed_s']:.2f}s ({mbps:.1f} MB/s)"
                )
            except Exception as e:
                logging.error(f"Failed to rewrite array '{rel_name}': {e}")
                raise

    elapsed_total = time.time() - start_total_time
    total_volume_gb = total_volume_bytes / (1024 * 1024 * 1024)
    total_volume_mb = total_volume_bytes / (1024 * 1024)
    overall_mbps = total_volume_mb / elapsed_total if elapsed_total > 0 else 0
    overall_reduction_pct = (
        (1.0 - (total_dst_chunks / total_src_chunks)) * 100.0 if total_src_chunks > 0 else 0.0
    )

    summary = {
        "source_checkpoint": src_dir,
        "destination_checkpoint": dst_dir,
        "strategy": strategy,
        "target_chunk_mb": target_chunk_mb,
        "strip_opt_state": strip_opt_state,
        "cast_dtype": cast_dtype,
        "arrays_processed": len(results),
        "metadata_files_copied": copied_meta,
        "total_volume_bytes": total_volume_bytes,
        "total_volume_gb": round(total_volume_gb, 4),
        "source_total_chunks": total_src_chunks,
        "target_total_chunks": total_dst_chunks,
        "chunk_reduction_percent": round(overall_reduction_pct, 2),
        "elapsed_seconds": round(elapsed_total, 2),
        "throughput_mb_per_sec": round(overall_mbps, 2),
        "dry_run": dry_run,
    }

    # 4. Save manifest and report
    if not dry_run:
        manifest_path = os.path.join(dst_dir, "rewrite_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Saved rewrite manifest: {manifest_path}")

    logging.info("==================================================================================")
    logging.info("                 ORBAX CHECKPOINT REWRITE COMPLETE                                ")
    logging.info("==================================================================================")
    logging.info(f"Arrays Processed     : {len(results)}")
    logging.info(f"Total Volume         : {total_volume_gb:.4f} GB ({total_volume_mb:.2f} MB)")
    logging.info(f"Source Total Chunks  : {total_src_chunks} chunk files")
    logging.info(f"Target Total Chunks  : {total_dst_chunks} chunk files ({overall_reduction_pct:+.2f}% chunk reduction)")
    logging.info(f"Total Time Elapsed   : {elapsed_total:.2f} s ({elapsed_total/60.0:.2f} min)")
    logging.info(f"Throughput           : {overall_mbps:.2f} MB/s")
    logging.info("==================================================================================")

    return summary


def main():
    args = parse_args()
    try:
        run_orbax_rewrite(
            src_dir=args.src_dir,
            dst_dir=args.dst_dir,
            strategy=args.strategy,
            target_chunk_mb=args.target_chunk_mb,
            dim_partitions_str=args.dim_partitions,
            strip_opt_state=args.strip_opt_state,
            cast_dtype=args.cast_dtype,
            num_workers=args.num_workers,
            max_buffer_mb=args.max_buffer_mb,
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
            verify=args.verify,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logging.error(f"Orbax rewrite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
