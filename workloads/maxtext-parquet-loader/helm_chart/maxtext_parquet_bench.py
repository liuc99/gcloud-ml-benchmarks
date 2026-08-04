#!/usr/bin/env python3
"""
MaxText Parquet Dataset Loading & GCS Range Read Benchmark & Demo.

Simulates the MaxText LLM training data pipeline reading multi-file Parquet datasets
via GCS Range Read operations. Supports both:
  1. Native GCS Client (gcsfs / pyarrow.fs.GCSFileSystem over gs://...)
  2. GCSFuse Sidecar Mount (/gcs/... POSIX seek+read over GCSFuse)
"""

import argparse
import json
import logging
import os
import sys
import time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MAXTEXT_PARQUET] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="MaxText Parquet GCS Range Read Benchmark")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to Parquet dataset directory (e.g. gs://my-bucket/parquet_data or /gcs/my-bucket/parquet_data)",
    )
    parser.add_argument(
        "--access-mode",
        type=str,
        default="auto",
        choices=["auto", "native_gcs", "gcsfuse", "posix"],
        help="Access mode: native_gcs (gs:// via pyarrow/gcsfs), gcsfuse (/gcs/ via FUSE mount), posix",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="input_ids,label",
        help="Comma-separated column names to project/read (simulates reading specific columns without full file read)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per step",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Max batches to read during benchmark",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of concurrent PyArrow / Grain reader threads",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=int(os.environ.get("RANK", os.environ.get("JOB_COMPLETION_INDEX", "0"))),
        help="Process rank / JAX process index",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", os.environ.get("NNODES", "1"))),
        help="Total ranks / JAX processes across cluster",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# GCS Range Read Wrapper for PyArrow
# -----------------------------------------------------------------------------
class MetricFileSystemWrapper:
    """
    Wraps a PyArrow / POSIX filesystem to intercept and measure seek & range read calls.
    """
    def __init__(self, target_fs, access_mode):
        self.fs = target_fs
        self.access_mode = access_mode
        self.range_read_calls = 0
        self.total_bytes_read = 0
        self.range_read_durations = []

    def open_input_file(self, path):
        class MetricFile:
            def __init__(outer, file_obj):
                outer.file_obj = file_obj

            def read(outer, nbytes=None):
                start = time.perf_counter()
                data = outer.file_obj.read(nbytes)
                dur = time.perf_counter() - start
                self.range_read_calls += 1
                self.total_bytes_read += len(data)
                self.range_read_durations.append(dur)
                return data

            def read_at(outer, nbytes, offset):
                start = time.perf_counter()
                if hasattr(outer.file_obj, "read_at"):
                    data = outer.file_obj.read_at(nbytes, offset)
                else:
                    outer.file_obj.seek(offset)
                    data = outer.file_obj.read(nbytes)
                dur = time.perf_counter() - start
                self.range_read_calls += 1
                self.total_bytes_read += len(data)
                self.range_read_durations.append(dur)
                return data

            def seek(outer, offset, whence=0):
                return outer.file_obj.seek(offset, whence)

            def tell(outer):
                return outer.file_obj.tell()

            def size(outer):
                if hasattr(outer.file_obj, "size"):
                    return outer.file_obj.size()
                outer.file_obj.seek(0, 2)
                sz = outer.file_obj.tell()
                return sz

            @property
            def closed(outer):
                return getattr(outer.file_obj, "closed", False)

            def readable(outer):
                return True

            def seekable(outer):
                return True

            @property
            def mode(outer):
                return "rb"

            def close(outer):
                if hasattr(outer.file_obj, "close"):
                    return outer.file_obj.close()

            def __enter__(outer):
                return outer

            def __exit__(outer, exc_type, exc_val, exc_tb):
                outer.close()

        file_obj = self.fs.open_input_file(path)
        return MetricFile(file_obj)


def setup_filesystem(dataset_path, access_mode):
    import pyarrow.fs as pafs

    dataset_path = dataset_path.rstrip("/")

    if access_mode == "auto":
        if dataset_path.startswith("gs://"):
            access_mode = "native_gcs"
        elif dataset_path.startswith("/gcs/"):
            access_mode = "gcsfuse"
        else:
            access_mode = "posix"

    logging.info(f"Setting up filesystem for path: '{dataset_path}' in access_mode: '{access_mode}'")

    if access_mode == "native_gcs":
        # Remove gs:// prefix for PyArrow GcsFileSystem
        clean_path = dataset_path.replace("gs://", "")
        if hasattr(pafs, "GcsFileSystem"):
            raw_fs = pafs.GcsFileSystem()
        elif hasattr(pafs, "GCSFileSystem"):
            raw_fs = pafs.GCSFileSystem()
        else:
            import gcsfs
            raw_fs = gcsfs.GCSFileSystem()
        fs_wrapper = MetricFileSystemWrapper(raw_fs, access_mode)
        return fs_wrapper, clean_path, access_mode
    elif access_mode in ("gcsfuse", "posix"):
        raw_fs = pafs.LocalFileSystem()
        fs_wrapper = MetricFileSystemWrapper(raw_fs, access_mode)
        return fs_wrapper, dataset_path, access_mode
    else:
        raise ValueError(f"Unknown access_mode: {access_mode}")


# -----------------------------------------------------------------------------
# MaxText Parquet Range Read Pipeline
# -----------------------------------------------------------------------------
def run_maxtext_parquet_benchmark(dataset_path, access_mode, columns_to_read, batch_size, max_batches, num_threads, rank, world_size):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logging.error("pyarrow is required for MaxText Parquet Range Read benchmark. Please install pyarrow (e.g. pip install pyarrow).")
        sys.exit(1)

    fs_wrapper, clean_path, access_mode = setup_filesystem(dataset_path, access_mode)

    logging.info(f"[MAXTEXT] Discovering Parquet shard files under: {clean_path}")
    
    start_discovery = time.perf_counter()
    if access_mode == "native_gcs":
        selector = pa.fs.FileSelector(clean_path, recursive=True)
        file_infos = fs_wrapper.fs.get_file_info(selector)
        parquet_files = [info.path for info in file_infos if info.path.endswith(".parquet")]
    else:
        parquet_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(clean_path)
            for file in files
            if file.endswith(".parquet")
        ]

    parquet_files = sorted(parquet_files)
    discovery_duration = time.perf_counter() - start_discovery
    logging.info(f"[MAXTEXT] Found {len(parquet_files)} Parquet files in {discovery_duration:.4f}s")

    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {dataset_path}")

    # Shard files across JAX process ranks (MaxText multi-node data parallelism)
    rank_files = parquet_files[rank::world_size]
    logging.info(f"[MAXTEXT] [Rank {rank}/{world_size}] Assigned {len(rank_files)} Parquet shards")

    # Step 1: Measure Parquet Metadata / Footer Range Read Overhead
    logging.info("[MAXTEXT] Step 1: Measuring Parquet Footer & Metadata Range Read Latency...")
    footer_start = time.perf_counter()
    total_rows = 0

    from concurrent.futures import ThreadPoolExecutor

    def get_meta(fp):
        m = pq.read_metadata(fp, filesystem=fs_wrapper.fs)
        return fp, m

    parquet_metadatas = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(get_meta, rank_files)
        for fpath, meta in results:
            parquet_metadatas.append((fpath, meta))
            total_rows += meta.num_rows

    footer_duration = time.perf_counter() - footer_start

    logging.info(
        f"[MAXTEXT] ✅ Footer Parse Complete: {len(rank_files)} files parsed in {footer_duration * 1000:.2f} ms "
        f"(Total dataset rows discovered: {total_rows})"
    )

    # Step 2: MaxText Column Projection Range Read Benchmark
    logging.info(
        f"[MAXTEXT] Step 2: Executing Column Projection Range Reads for columns={columns_to_read} "
        f"(batch_size={batch_size}, max_batches={max_batches})..."
    )

    bench_start = time.perf_counter()
    first_batch_time = None

    loaded_batches = 0
    total_samples = 0
    total_feature_bytes = 0

    if columns_to_read.strip().lower() in ("auto", "all", ""):
        # Auto-detect column names from first parquet file schema
        first_pq = pq.ParquetFile(rank_files[0], filesystem=fs_wrapper.fs)
        columns_list = first_pq.schema.names
        logging.info(f"[MAXTEXT] Auto-detected dataset schema columns: {columns_list}")
    else:
        columns_list = [c.strip() for c in columns_to_read.split(",") if c.strip()]

    # Iterate Parquet files and read targeted column row groups
    for fpath, meta in parquet_metadatas:
        if loaded_batches >= max_batches:
            break

        parquet_file = pq.ParquetFile(fpath, filesystem=fs_wrapper.fs)

        for rg_idx in range(parquet_file.num_row_groups):
            if loaded_batches >= max_batches:
                break

            rg_start = time.perf_counter()
            # Read only selected columns via GCS Range Reads
            table = parquet_file.read_row_group(rg_idx, columns=columns_list, use_threads=True)
            rg_duration = time.perf_counter() - rg_start
            
            # Simulate MaxText JAX tensor batch creation
            batch_dict = {col: table[col].to_numpy() for col in table.column_names}

            now = time.perf_counter()
            if loaded_batches == 0:
                first_batch_time = now - bench_start
                logging.info(f"[MAXTEXT] Time to First Batch (TTFB): {first_batch_time * 1000:.2f} ms")

            num_samples = len(table)
            batch_bytes = sum(arr.nbytes for arr in batch_dict.values())
            
            total_samples += num_samples
            total_feature_bytes += batch_bytes
            loaded_batches += 1

            if loaded_batches % 20 == 0:
                logging.info(f"  [MaxText Batch {loaded_batches}/{max_batches}] Read {num_samples} samples ({batch_bytes / 1e6:.2f} MB)")

    total_duration = time.perf_counter() - bench_start
    data_bytes_read = fs_wrapper.total_bytes_read
    data_range_requests = fs_wrapper.range_read_calls

    throughput_mbs = (data_bytes_read / (1024 * 1024)) / total_duration if total_duration > 0 else 0.0
    throughput_gbps = (throughput_mbs * 8) / 1024
    samples_per_sec = total_samples / total_duration if total_duration > 0 else 0.0

    # Range Read Efficiency: Ratio of useful feature payload bytes vs total bytes read from GCS
    efficiency_pct = (total_feature_bytes / data_bytes_read * 100.0) if data_bytes_read > 0 else 100.0

    avg_range_latency_ms = (np.mean(fs_wrapper.range_read_durations) * 1000) if fs_wrapper.range_read_durations else 0.0
    p95_range_latency_ms = (np.percentile(fs_wrapper.range_read_durations, 95) * 1000) if fs_wrapper.range_read_durations else 0.0

    logging.info("==================================================================================")
    logging.info("                    MAXTEXT PARQUET GCS RANGE READ SUMMARY                        ")
    logging.info("==================================================================================")
    logging.info(f"Access Mode              : {access_mode}")
    logging.info(f"Dataset Path             : {dataset_path}")
    logging.info(f"Target Projected Columns : {columns_to_read}")
    logging.info(f"Total Batches Ingested   : {loaded_batches} batches")
    logging.info(f"Total Samples Processed  : {total_samples} samples")
    logging.info(f"Time to First Batch TTFB : {first_batch_time * 1000:.2f} ms ({first_batch_time:.4f} s)")
    logging.info(f"Footer Metadata Latency  : {footer_duration * 1000:.2f} ms ({footer_read_count} range requests)")
    logging.info(f"Data GCS Range Requests  : {data_range_requests} requests")
    logging.info(f"GCS Bytes Downloaded     : {data_bytes_read / (1024 * 1024):.2f} MB")
    logging.info(f"Useful Feature Payload   : {total_feature_bytes / (1024 * 1024):.2f} MB")
    logging.info(f"Range Read Efficiency    : {efficiency_pct:.2f}%")
    logging.info(f"Read Throughput          : {throughput_mbs:.2f} MB/s ({throughput_gbps:.2f} Gbps)")
    logging.info(f"Ingestion Speed          : {samples_per_sec:.2f} samples/sec")
    logging.info(f"Range Read Latency (Avg) : {avg_range_latency_ms:.2f} ms")
    logging.info(f"Range Read Latency (p95) : {p95_range_latency_ms:.2f} ms")
    logging.info("==================================================================================")


def main():
    args = parse_args()
    logging.info(
        f"Starting MaxText Parquet Range Read Benchmark: path={args.dataset_path}, mode={args.access_mode}, "
        f"columns={args.columns}, batch_size={args.batch_size}, rank={args.rank}/{args.world_size}"
    )

    run_maxtext_parquet_benchmark(
        args.dataset_path,
        args.access_mode,
        args.columns,
        args.batch_size,
        args.max_batches,
        args.num_threads,
        args.rank,
        args.world_size,
    )


if __name__ == "__main__":
    main()
