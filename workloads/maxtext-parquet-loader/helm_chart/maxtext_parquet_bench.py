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
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "arrayrecord"],
        help="Dataset format to benchmark: 'parquet' or 'arrayrecord'",
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
        "--shuffle-mode",
        type=str,
        default="none",
        choices=["none", "two_stage", "global"],
        help="Shuffle mode: 'none' (sequential), 'two_stage' (Grain-style streaming file+buffer shuffle), 'global' (upfront full-index global shuffle)",
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
def run_maxtext_parquet_benchmark(dataset_path, access_mode, columns_to_read, batch_size, max_batches, num_threads, rank, world_size, shuffle_mode="none", data_format="parquet"):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logging.error("pyarrow is required for MaxText Parquet Range Read benchmark. Please install pyarrow (e.g. pip install pyarrow).")
        sys.exit(1)

    try:
        import sys
        sys.path.append(os.path.expanduser("~/.local/lib/python3.13/site-packages"))
        from array_record.python import array_record_module
    except ImportError:
        array_record_module = None

    if data_format == "arrayrecord" and array_record_module is None:
        logging.error("array_record module is required for arrayrecord format benchmark. Please install array-record.")
        sys.exit(1)

    fs_wrapper, clean_path, access_mode = setup_filesystem(dataset_path, access_mode)

    ext = ".array_record" if data_format == "arrayrecord" else ".parquet"
    logging.info(f"[MAXTEXT] Discovering {data_format.upper()} ({ext}) shard files under: {clean_path}")
    
    start_discovery = time.perf_counter()
    if access_mode == "native_gcs":
        selector = pa.fs.FileSelector(clean_path, recursive=True)
        file_infos = fs_wrapper.fs.get_file_info(selector)
        shard_files = [info.path for info in file_infos if info.path.endswith(ext)]
    else:
        shard_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(clean_path)
            for file in files
            if file.endswith(ext)
        ]

    shard_files = sorted(shard_files)
    discovery_duration = time.perf_counter() - start_discovery
    logging.info(f"[MAXTEXT] Found {len(shard_files)} {data_format.upper()} files in {discovery_duration:.4f}s")

    if not shard_files:
        raise FileNotFoundError(f"No {ext} files found in {dataset_path}")

    # Shard files across JAX process ranks (MaxText multi-node data parallelism)
    rank_files = shard_files[rank::world_size]
    logging.info(f"[MAXTEXT] [Rank {rank}/{world_size}] Assigned {len(rank_files)} {data_format.upper()} shards")

    upfront_index_duration = 0.0

    if shuffle_mode == "two_stage":
        import random
        logging.info("[MAXTEXT] [SHUFFLE: TWO_STAGE] Stage 1: Shuffling file shard order deterministically with seed...")
        random.seed(42 + rank)
        random.shuffle(rank_files)
    elif shuffle_mode == "global":
        logging.info(f"[MAXTEXT] [SHUFFLE: GLOBAL] Stage 1: Upfront Scanning ALL {data_format.upper()} footers to build Global Index Map...")
        idx_start = time.perf_counter()
        from concurrent.futures import ThreadPoolExecutor
        def scan_footer(fp):
            if data_format == "parquet":
                m = pq.read_metadata(fp, filesystem=fs_wrapper.fs)
                return fp, m.num_rows
            else:
                r = array_record_module.ArrayRecordReader(fp)
                return fp, r.num_records()
        with ThreadPoolExecutor(max_workers=16) as exec_pool:
            _ = list(exec_pool.map(scan_footer, rank_files))
        upfront_index_duration = time.perf_counter() - idx_start
        logging.info(f"[MAXTEXT] [SHUFFLE: GLOBAL] ⚠️ Upfront Global Index Map Created in {upfront_index_duration:.2f}s ({upfront_index_duration * 1000:.0f} ms penalty!)")

    # Step 1: MaxText Schema Auto-Discovery & Header Reader Setup
    logging.info(f"[MAXTEXT] Step 1: Initializing {data_format.upper()} Schema Discovery & Dataset Shard Reader...")
    footer_start = time.perf_counter()

    if data_format == "parquet":
        first_pq = pq.ParquetFile(rank_files[0], filesystem=fs_wrapper.fs)
        if columns_to_read.strip().lower() in ("auto", "all", ""):
            columns_list = first_pq.schema.names
        else:
            columns_list = [c.strip() for c in columns_to_read.split(",") if c.strip()]
    else:
        columns_list = ["int32_tokens"]

    footer_duration = time.perf_counter() - footer_start

    logging.info(
        f"[MAXTEXT] ✅ Schema Discovery Complete in {footer_duration * 1000:.2f} ms "
        f"({len(columns_list)} columns/fields: {columns_list})"
    )

    # Step 2: MaxText Data Read Benchmark
    logging.info(
        f"[MAXTEXT] Step 2: Executing {data_format.upper()} Data Reads "
        f"(shuffle_mode={shuffle_mode}, batch_size={batch_size}, max_batches={max_batches})..."
    )

    bench_start = time.perf_counter()
    first_batch_time = None

    loaded_batches = 0
    total_samples = 0
    total_feature_bytes = 0
    batch_durations = []
    shuffle_buffer = []

    if data_format == "parquet":
        for fpath in rank_files:
            if loaded_batches >= max_batches:
                break
            parquet_file = pq.ParquetFile(fpath, filesystem=fs_wrapper.fs)
            for rg_idx in range(parquet_file.num_row_groups):
                if loaded_batches >= max_batches:
                    break

                rg_start = time.perf_counter()
                table = parquet_file.read_row_group(rg_idx, columns=columns_list, use_threads=True)
                rg_duration = time.perf_counter() - rg_start
                batch_dict = {col: table[col].to_numpy() for col in table.column_names}

                if shuffle_mode == "two_stage":
                    shuffle_buffer.append((batch_dict, table, rg_duration))
                    if len(shuffle_buffer) < 4 and loaded_batches + len(shuffle_buffer) < max_batches:
                        continue
                    import random
                    idx = random.randint(0, len(shuffle_buffer) - 1)
                    batch_dict, table, rg_duration = shuffle_buffer.pop(idx)

                now = time.perf_counter()
                if loaded_batches == 0:
                    first_batch_time = (now - bench_start) + upfront_index_duration
                    logging.info(f"[MAXTEXT] Time to First Batch (TTFB): {first_batch_time * 1000:.2f} ms ({first_batch_time:.4f} s)")

                num_samples = len(table)
                batch_bytes = sum(arr.nbytes for arr in batch_dict.values())
                total_samples += num_samples
                total_feature_bytes += batch_bytes
                loaded_batches += 1
                batch_durations.append(rg_duration)

                if loaded_batches % 20 == 0:
                    logging.info(f"  [MaxText Batch {loaded_batches}/{max_batches}] Read {num_samples} samples ({batch_bytes / (1024 * 1024):.2f} MB)")
    else:
        import random
        if shuffle_mode in ["two_stage", "global"]:
            random.shuffle(rank_files)

        shuffle_buffer = []

        for fpath in rank_files:
            if loaded_batches >= max_batches:
                break
            
            if access_mode == "native_gcs":
                local_fpath = f"/tmp/{os.path.basename(fpath)}"
                import gcsfs
                gcs_fs = gcsfs.GCSFileSystem()
                clean_fpath = fpath.replace("gs://", "")
                logging.info(f"[MAXTEXT] [ARRAYRECORD] Streaming/Downloading shard {clean_fpath} -> {local_fpath}...")
                gcs_fs.get_file(clean_fpath, local_fpath)
            else:
                local_fpath = fpath

            reader = array_record_module.ArrayRecordReader(local_fpath)
            num_recs = reader.num_records()
            current_batch = []
            
            indices = list(range(num_recs))
            if shuffle_mode == "global":
                random.shuffle(indices)

            for idx in indices:
                if loaded_batches >= max_batches:
                    break

                b_start = time.perf_counter()
                raw_bytes = reader.read([idx])[0]
                tokens = np.frombuffer(raw_bytes, dtype=np.int32)
                b_duration = time.perf_counter() - b_start

                current_batch.append(tokens)

                if len(current_batch) >= batch_size:
                    if shuffle_mode == "two_stage":
                        shuffle_buffer.append((current_batch, b_duration))
                        current_batch = []
                        if len(shuffle_buffer) < 4 and loaded_batches + len(shuffle_buffer) < max_batches:
                            continue
                        pick_idx = random.randint(0, len(shuffle_buffer) - 1)
                        batch_to_emit, b_duration = shuffle_buffer.pop(pick_idx)
                    else:
                        batch_to_emit = current_batch
                        current_batch = []

                    now = time.perf_counter()
                    if loaded_batches == 0:
                        first_batch_time = (now - bench_start) + upfront_index_duration
                        logging.info(f"[MAXTEXT] Time to First Batch (TTFB): {first_batch_time * 1000:.2f} ms ({first_batch_time:.4f} s)")

                    num_samples = len(batch_to_emit)
                    batch_bytes = sum(t.nbytes for t in batch_to_emit)
                    total_samples += num_samples
                    total_feature_bytes += batch_bytes
                    loaded_batches += 1
                    batch_durations.append(b_duration)

                    if loaded_batches % 20 == 0:
                        logging.info(f"  [MaxText Batch {loaded_batches}/{max_batches}] Read {num_samples} samples ({batch_bytes / (1024 * 1024):.2f} MB)")

            reader.close()
            if access_mode == "native_gcs" and os.path.exists(local_fpath):
                os.remove(local_fpath)

        if shuffle_mode == "two_stage" and shuffle_buffer:
            while shuffle_buffer and loaded_batches < max_batches:
                pick_idx = random.randint(0, len(shuffle_buffer) - 1)
                batch_to_emit, b_duration = shuffle_buffer.pop(pick_idx)

                now = time.perf_counter()
                if loaded_batches == 0:
                    first_batch_time = (now - bench_start) + upfront_index_duration
                    logging.info(f"[MAXTEXT] Time to First Batch (TTFB): {first_batch_time * 1000:.2f} ms ({first_batch_time:.4f} s)")

                num_samples = len(batch_to_emit)
                batch_bytes = sum(t.nbytes for t in batch_to_emit)
                total_samples += num_samples
                total_feature_bytes += batch_bytes
                loaded_batches += 1
                batch_durations.append(b_duration)

    total_duration = (time.perf_counter() - bench_start) + upfront_index_duration

    # Effective Feature IO Read Throughput
    throughput_mbs = (total_feature_bytes / (1024 * 1024)) / total_duration if total_duration > 0 else 0.0
    throughput_gbps = (throughput_mbs * 8) / 1024
    samples_per_sec = total_samples / total_duration if total_duration > 0 else 0.0

    avg_batch_ms = (np.mean(batch_durations) * 1000) if batch_durations else 0.0
    p50_batch_ms = (np.percentile(batch_durations, 50) * 1000) if batch_durations else 0.0
    p95_batch_ms = (np.percentile(batch_durations, 95) * 1000) if batch_durations else 0.0
    p99_batch_ms = (np.percentile(batch_durations, 99) * 1000) if batch_durations else 0.0

    logging.info("==================================================================================")
    logging.info(f"               MAXTEXT {data_format.upper()} DATASET READ SUMMARY                 ")
    logging.info("==================================================================================")
    logging.info(f"Data Format              : {data_format.upper()}")
    logging.info(f"Access Mode              : {access_mode}")
    logging.info(f"Shuffle Mode             : {shuffle_mode}")
    logging.info(f"Dataset Path             : {dataset_path}")
    logging.info(f"Total Dataset Shards     : {len(shard_files)} files")
    logging.info(f"Target Projected Fields  : {columns_list}")
    logging.info(f"Total Batches Ingested   : {loaded_batches} batches")
    logging.info(f"Total Samples Ingested   : {total_samples} samples")
    logging.info(f"Payload Data Read Volume : {total_feature_bytes / (1024 * 1024):.2f} MB ({total_feature_bytes / (1024 * 1024 * 1024):.4f} GB)")
    logging.info(f"Time to First Batch TTFB : {first_batch_time * 1000:.2f} ms ({first_batch_time:.4f} s)" if first_batch_time else "N/A")
    if shuffle_mode == "global":
        logging.info(f"Upfront Index Penalty    : {upfront_index_duration * 1000:.2f} ms ({upfront_index_duration:.2f} s)")
    logging.info(f"Schema Discovery Latency : {footer_duration * 1000:.2f} ms")
    logging.info(f"IO Read Throughput       : {throughput_mbs:.2f} MB/s ({throughput_gbps:.2f} Gbps)")
    logging.info(f"Sample Ingestion Speed   : {samples_per_sec:.2f} samples/sec")
    logging.info(f"Batch Load Latency (Avg) : {avg_batch_ms:.2f} ms")
    logging.info(f"Batch Load Latency (p50) : {p50_batch_ms:.2f} ms")
    logging.info(f"Batch Load Latency (p95) : {p95_batch_ms:.2f} ms")
    logging.info(f"Batch Load Latency (p99) : {p99_batch_ms:.2f} ms")
    logging.info("==================================================================================")


def main():
    args = parse_args()
    logging.info(
        f"Starting MaxText Range Read Benchmark: path={args.dataset_path}, format={args.format}, "
        f"mode={args.access_mode}, shuffle_mode={args.shuffle_mode}, columns={args.columns}, "
        f"batch_size={args.batch_size}, rank={args.rank}/{args.world_size}"
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
        args.shuffle_mode,
        args.format,
    )


if __name__ == "__main__":
    main()
