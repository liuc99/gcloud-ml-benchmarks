#!/usr/bin/env python3
"""
MaxText Parquet Dataset Generator for GCS Range Read Benchmarks.

Generates multi-column Parquet datasets containing:
  - input_ids (token sequence)
  - attention_mask (mask sequence)
  - labels (target labels)
  - extra_metadata_bytes (large unused binary payload)

Demonstrates how Column Projection & Range Reads allow reading only input_ids/labels
without transferring extra_metadata_bytes from GCS.
"""

import argparse
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
    parser = argparse.ArgumentParser(description="MaxText Parquet Dataset Generator")
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Target output directory or bucket path (e.g. gs://my-bucket/maxtext_parquet or /gcs/my-bucket/maxtext_parquet)",
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
        help="Number of Parquet shard files to generate",
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


def generate_maxtext_parquet_dataset(output_path, total_size_mb, num_files, sequence_length, metadata_bytes_per_row):
    target_bytes_per_file = (total_size_mb * 1024 * 1024) / num_files
    # Bytes per row: input_ids (8B * seq) + attention_mask (1B * seq) + labels (8B * seq) + metadata_bytes
    bytes_per_row = max(512, sequence_length * 17 + metadata_bytes_per_row)
    rows_per_file = int(math.ceil(target_bytes_per_file / bytes_per_row))

    logging.info(
        f"Generating {num_files} MaxText Parquet files with ~{rows_per_file} rows/file "
        f"(seq_len={sequence_length}, metadata_bytes={metadata_bytes_per_row} B/row)..."
    )

    is_native_gcs = output_path.startswith("gs://")
    if is_native_gcs:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
    else:
        os.makedirs(output_path, exist_ok=True)

    start_time = time.perf_counter()
    total_written_bytes = 0

    for i in range(num_files):
        file_start = time.perf_counter()
        
        sample_ids = np.arange(i * rows_per_file, (i + 1) * rows_per_file, dtype=np.int64)
        input_ids = np.random.randint(0, 32000, size=(rows_per_file, sequence_length), dtype=np.int64)
        attention_mask = np.ones((rows_per_file, sequence_length), dtype=np.int8)
        labels = np.random.randint(0, 32000, size=(rows_per_file, sequence_length), dtype=np.int64)

        # Generate extra metadata payload to simulate unread columns
        dummy_payload = b"X" * metadata_bytes_per_row
        extra_metadata = [dummy_payload] * rows_per_file

        table = pa.Table.from_arrays(
            [
                pa.array(sample_ids),
                pa.array(input_ids.tolist()),
                pa.array(attention_mask.tolist()),
                pa.array(labels.tolist()),
                pa.array(extra_metadata),
            ],
            names=["sample_id", "input_ids", "attention_mask", "label", "extra_metadata_bytes"],
        )

        file_path = f"{output_path}/maxtext_shard_{i:05d}.parquet"
        if is_native_gcs:
            with fs.open(file_path, "wb") as f:
                pq.write_table(table, f, compression="snappy")
            written_bytes = fs.du(file_path)
        else:
            pq.write_table(file_path, table, compression="snappy")
            written_bytes = os.path.getsize(file_path)

        total_written_bytes += written_bytes
        elapsed = time.perf_counter() - file_start
        logging.info(
            f"  [Shard {i+1}/{num_files}] Wrote {file_path} ({written_bytes / 1e6:.2f} MB) in {elapsed:.2f}s"
        )

    total_duration = time.perf_counter() - start_time
    total_mb = total_written_bytes / (1024 * 1024)
    logging.info(
        f"✅ MaxText Parquet dataset generation complete: {total_mb:.2f} MB generated in {total_duration:.2f}s ({total_mb / total_duration:.2f} MB/s)"
    )


def main():
    args = parse_args()
    output_path = args.output_path.rstrip("/")
    generate_maxtext_parquet_dataset(
        output_path, args.total_size_mb, args.num_files, args.sequence_length, args.metadata_bytes_per_row
    )


if __name__ == "__main__":
    main()
