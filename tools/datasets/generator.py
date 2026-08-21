#!/usr/bin/env python3
"""
Unified ML Benchmark Synthetic Dataset Generator.

Generates multi-column Parquet, ArrayRecord, or raw shard datasets for ML storage and data loading benchmarks.
Supports direct local POSIX paths, GCSFuse mounts (/gcs/...), and Native GCS (gs://...).
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [DATASET_GEN] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified ML Synthetic Dataset Generator")
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Target output directory or bucket path (e.g. gs://my-bucket/dataset or /gcs/my-bucket/dataset)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet"],
        help="Dataset format to generate (default: parquet)",
    )
    parser.add_argument(
        "--total-size-mb",
        type=float,
        default=1024.0,
        help="Target total dataset size in Megabytes",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="Number of shard files to generate",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Tokens / Sequence length per sample (default: 2048 tokens)",
    )
    parser.add_argument(
        "--metadata-bytes-per-row",
        type=int,
        default=4096,
        help="Bytes of extra metadata payload per row (used to test range read column projection)",
    )
    return parser.parse_args()


def generate_parquet_dataset(output_path, total_size_mb, num_files, sequence_length, metadata_bytes_per_row):
    target_bytes_per_file = (total_size_mb * 1024 * 1024) / num_files
    bytes_per_row = max(512, sequence_length * 17 + metadata_bytes_per_row)
    rows_per_file = int(math.ceil(target_bytes_per_file / bytes_per_row))

    logging.info(
        f"Generating {num_files} Parquet files with ~{rows_per_file} rows/file "
        f"(seq_len={sequence_length}, metadata_bytes={metadata_bytes_per_row} B/row)..."
    )

    import tempfile
    import shutil

    is_native_gcs = output_path.startswith("gs://")
    if is_native_gcs:
        staging_dir = tempfile.mkdtemp(prefix="parquet_gen_")
        local_dir = staging_dir
    else:
        staging_dir = None
        local_dir = output_path

    os.makedirs(local_dir, exist_ok=True)

    start_time = time.perf_counter()
    total_written_bytes = 0
    generated_files = []

    try:
        for i in range(num_files):
            file_start = time.perf_counter()

            sample_ids = np.arange(i * rows_per_file, (i + 1) * rows_per_file, dtype=np.int64)
            input_ids = np.random.randint(0, 32000, size=(rows_per_file, sequence_length), dtype=np.int64)
            attention_mask = np.ones((rows_per_file, sequence_length), dtype=np.int8)
            labels = np.random.randint(0, 32000, size=(rows_per_file, sequence_length), dtype=np.int64)

            dummy_payload = b"X" * metadata_bytes_per_row
            extra_metadata = [dummy_payload] * rows_per_file

            table = pa.Table.from_arrays(
                [
                    pa.array(sample_ids),
                    pa.FixedSizeListArray.from_arrays(pa.array(input_ids.ravel()), sequence_length),
                    pa.FixedSizeListArray.from_arrays(pa.array(attention_mask.ravel()), sequence_length),
                    pa.FixedSizeListArray.from_arrays(pa.array(labels.ravel()), sequence_length),
                    pa.array(extra_metadata),
                ],
                names=["sample_id", "input_ids", "attention_mask", "label", "extra_metadata_bytes"],
            )

            local_file_path = f"{local_dir}/maxtext_shard_{i:05d}.parquet"
            pq.write_table(table, local_file_path, compression="snappy")
            written_bytes = os.path.getsize(local_file_path)
            total_written_bytes += written_bytes
            generated_files.append((local_file_path, f"maxtext_shard_{i:05d}.parquet", written_bytes))
            elapsed = time.perf_counter() - file_start
            logging.info(
                f"  [Gen Shard {i+1}/{num_files}] Wrote {local_file_path} ({written_bytes / 1e6:.2f} MB) in {elapsed:.2f}s"
            )

        if is_native_gcs:
            logging.info(f"Uploading {num_files} shards to {output_path} in parallel...")
            up_start = time.perf_counter()
            from concurrent.futures import ThreadPoolExecutor
            from google.cloud import storage
            import google.cloud.storage.blob

            google.cloud.storage.blob._MAX_MULTIPART_SIZE = 1000 * 1024 * 1024  # Force single-shot upload for Zonal RAPID buckets

            clean_uri = output_path.replace("gs://", "")
            parts = clean_uri.split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1].strip("/") if len(parts) > 1 else ""

            client = storage.Client()
            bucket = client.bucket(bucket_name)

            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            credentials, _ = google.auth.default()
            session = AuthorizedSession(credentials)

            def upload_file(item):
                lpath, fname, size = item
                blob_path = f"{prefix}/{fname}" if prefix else fname
                url = f"https://storage.googleapis.com/{bucket_name}/{blob_path}"
                headers = {
                    "Content-Type": "application/octet-stream",
                    "x-goog-gcs-appendable": "true",
                }
                uploaded = False
                try:
                    with open(lpath, "rb") as f:
                        resp = session.put(url, data=f, headers=headers)
                    if resp.status_code in (200, 201):
                        uploaded = True
                except Exception:
                    pass

                if not uploaded:
                    # Standard GCS client fallback for standard buckets
                    b = bucket.blob(blob_path)
                    b.upload_from_filename(lpath)

                if os.path.exists(lpath):
                    os.remove(lpath)
                return fname, size

            with ThreadPoolExecutor(max_workers=16) as pool:
                results = list(pool.map(upload_file, generated_files))

            logging.info(f"Parallel GCS Upload complete in {time.perf_counter() - up_start:.2f}s")
    finally:
        if staging_dir and os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)

    total_duration = time.perf_counter() - start_time
    total_mb = total_written_bytes / (1024 * 1024)
    logging.info(
        f"✅ Dataset generation complete: {total_mb:.2f} MB generated in {total_duration:.2f}s ({total_mb / total_duration:.2f} MB/s)"
    )


def main():
    args = parse_args()
    output_path = args.output_path.rstrip("/")
    if args.format == "parquet":
        generate_parquet_dataset(
            output_path, args.total_size_mb, args.num_files, args.sequence_length, args.metadata_bytes_per_row
        )


if __name__ == "__main__":
    main()
