#!/usr/bin/env python3
"""
Parquet to ArrayRecord Converter for MaxText Dataset Loading Benchmarks.

Reads Parquet text dataset shards, tokenizes text into int32 token arrays,
and writes out pre-tokenized ArrayRecord binary shards (.array_record) to GCS/POSIX.
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

try:
    import sys
    sys.path.append(os.path.expanduser("~/.local/lib/python3.13/site-packages"))
    from array_record.python import array_record_module
except ImportError:
    array_record_module = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PARQUET2ARRAYRECORD] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Parquet to ArrayRecord Converter")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Input Parquet dataset path (e.g. gs://my-bucket/parquet_dataset)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output ArrayRecord dataset path (e.g. gs://my-bucket/arrayrecord_dataset)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing raw text (default: text)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Target sequence length for tokenized arrays (default: 2048)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum Parquet files to convert (0 for all files)",
    )
    return parser.parse_args()


def convert_parquet_to_arrayrecord(input_path, output_path, text_column="text", sequence_length=2048, max_files=0):
    if pq is None:
        logging.error("pyarrow is required. Please install pyarrow.")
        sys.exit(1)
    if array_record_module is None:
        logging.error("array_record is required. Please install array-record.")
        sys.exit(1)

    clean_input = input_path.replace("gs://", "").rstrip("/")
    clean_output = output_path.replace("gs://", "").rstrip("/")

    is_native_gcs = input_path.startswith("gs://")

    if is_native_gcs:
        import pyarrow.fs as pafs
        if hasattr(pafs, "GcsFileSystem"):
            fs = pafs.GcsFileSystem()
        elif hasattr(pafs, "GCSFileSystem"):
            fs = pafs.GCSFileSystem()
        else:
            import gcsfs
            fs = gcsfs.GCSFileSystem()

        selector = pafs.FileSelector(clean_input, recursive=True)
        file_infos = fs.get_file_info(selector)
        parquet_files = [info.path for info in file_infos if info.path.endswith(".parquet")]
    else:
        fs = None
        parquet_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(clean_input)
            for file in files
            if file.endswith(".parquet")
        ]
        os.makedirs(clean_output, exist_ok=True)

    parquet_files = sorted(parquet_files)
    if max_files > 0:
        parquet_files = parquet_files[:max_files]

    logging.info(f"Found {len(parquet_files)} Parquet shards to convert from '{input_path}' to '{output_path}'")

    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        tokenizer = None

    start_conv = time.perf_counter()
    total_records = 0
    total_bytes = 0

    for idx, fpath in enumerate(parquet_files):
        shard_start = time.perf_counter()
        shard_filename = f"shard-{idx:05d}-of-{len(parquet_files):05d}.array_record"
        
        if is_native_gcs:
            local_tmp_path = os.path.join("/tmp", shard_filename)
            out_gcs_file = f"{clean_output}/{shard_filename}"
        else:
            local_tmp_path = os.path.join(clean_output, shard_filename)
            out_gcs_file = None

        parquet_file = pq.ParquetFile(fpath, filesystem=fs if is_native_gcs else None)
        writer = array_record_module.ArrayRecordWriter(local_tmp_path, options="group_size:1")

        shard_records = 0
        for rg_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(rg_idx, columns=[text_column] if text_column in parquet_file.schema.names else parquet_file.schema.names[:1])
            col_data = table[table.column_names[0]].to_pylist()

            for item in col_data:
                if isinstance(item, str):
                    if tokenizer:
                        tokens = np.array(tokenizer.encode(item[:sequence_length * 4])[:sequence_length], dtype=np.int32)
                    else:
                        tokens = np.frombuffer(item[:sequence_length * 4].encode("utf-8"), dtype=np.uint8).astype(np.int32)
                elif isinstance(item, (bytes, bytearray)):
                    tokens = np.frombuffer(item, dtype=np.int32)
                else:
                    tokens = np.random.randint(0, 32000, size=sequence_length, dtype=np.int32)

                raw_bytes = tokens.tobytes()
                writer.write(raw_bytes)
                shard_records += 1
                total_bytes += len(raw_bytes)

        writer.close()

        if is_native_gcs and out_gcs_file:
            import google.auth
            from google.auth.transport.requests import Request
            import requests

            gcs_path_no_prefix = out_gcs_file.replace("gs://", "")
            bucket_name, blob_name = gcs_path_no_prefix.split("/", 1)
            credentials, _ = google.auth.default()
            credentials.refresh(Request())
            headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/octet-stream",
            }
            url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={blob_name}"
            with open(local_tmp_path, "rb") as f:
                resp = requests.post(url, headers=headers, data=f)
                resp.raise_for_status()
            os.remove(local_tmp_path)

        shard_duration = time.perf_counter() - shard_start
        total_records += shard_records
        logging.info(f"  [Shard {idx+1}/{len(parquet_files)}] Converted {shard_records} records in {shard_duration:.2f}s -> {output_path}/{shard_filename}")

    total_duration = time.perf_counter() - start_conv
    throughput_mbs = (total_bytes / (1024 * 1024)) / total_duration if total_duration > 0 else 0.0
    rec_per_sec = total_records / total_duration if total_duration > 0 else 0.0

    logging.info("==================================================================================")
    logging.info("                PARQUET TO ARRAYRECORD CONVERSION COMPLETE                        ")
    logging.info("==================================================================================")
    logging.info(f"Input Shards      : {len(parquet_files)} files")
    logging.info(f"Output Shards     : {output_path}")
    logging.info(f"Total Records     : {total_records} records")
    logging.info(f"Total Payload Size: {total_bytes / (1024 * 1024):.2f} MB")
    logging.info(f"Total Time Taken  : {total_duration:.2f} s")
    logging.info(f"Conversion Speed  : {throughput_mbs:.2f} MB/s ({rec_per_sec:.2f} records/sec)")
    logging.info("==================================================================================")


def main():
    args = parse_args()
    convert_parquet_to_arrayrecord(
        args.input_path,
        args.output_path,
        args.text_column,
        args.sequence_length,
        args.max_files,
    )


if __name__ == "__main__":
    main()
