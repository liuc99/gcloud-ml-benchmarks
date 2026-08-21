#!/usr/bin/env python3
"""
Parquet to ArrayRecord Multi-Process Converter for ML Dataset Loading Benchmarks.

Reads Parquet text dataset shards, tokenizes text into int32 token arrays,
and writes out pre-tokenized ArrayRecord binary shards (.array_record) to GCS/POSIX.
Supports true multi-process parallelism bypassing Python GIL for maximum CPU core utilization.
Automatically produces a metadata manifest.json to prevent cluster startup Metadata Listing Storms.
"""

import argparse
import concurrent.futures
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

try:
    from array_record.python import array_record_module
except ImportError:
    array_record_module = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PARQUET2ARRAYRECORD] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Parquet to ArrayRecord Multi-Process Converter")
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=64,
        help="Number of concurrent worker processes for conversion & upload (default: 64)",
    )
    return parser.parse_args()


def convert_single_shard_process(task_args):
    """
    Child process worker to bypass Python GIL and utilize individual CPU cores.
    """
    idx, fpath, total_files, clean_output, is_native_gcs, text_column, sequence_length = task_args
    shard_start = time.perf_counter()
    shard_filename = f"shard-{idx:05d}-of-{total_files:05d}.array_record"

    import pyarrow.parquet as pq_proc
    from array_record.python import array_record_module as ar_proc

    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        tokenizer = None

    if is_native_gcs:
        import pyarrow.fs as pafs
        if hasattr(pafs, "GcsFileSystem"):
            proc_fs = pafs.GcsFileSystem()
        elif hasattr(pafs, "GCSFileSystem"):
            proc_fs = pafs.GCSFileSystem()
        else:
            import gcsfs
            proc_fs = gcsfs.GCSFileSystem()
    else:
        proc_fs = None

    import tempfile
    temp_local_out = os.path.join(tempfile.gettempdir(), shard_filename)

    try:
        if proc_fs:
            table = pq_proc.read_table(fpath, filesystem=proc_fs)
        else:
            table = pq_proc.read_table(fpath)

        if text_column in table.column_names:
            texts = table[text_column].to_pylist()
        else:
            first_col = table.column_names[0]
            texts = table[first_col].to_pylist()

        writer = ar_proc.ArrayRecordWriter(temp_local_out, options="group_size:1")
        records_written = 0
        total_shard_bytes = 0

        for text in texts:
            if isinstance(text, str):
                if tokenizer:
                    tokens = np.array(
                        tokenizer.encode(
                            text[:sequence_length * 4],
                            disallowed_special=(),
                            allowed_special="all",
                        )[:sequence_length],
                        dtype=np.int32,
                    )
                else:
                    tokens = np.frombuffer(text[:sequence_length * 4].encode("utf-8"), dtype=np.uint8).astype(np.int32)
            elif isinstance(text, (bytes, bytearray)):
                tokens = np.frombuffer(text, dtype=np.int32)
            elif isinstance(text, (list, np.ndarray)):
                tokens = np.array(text[:sequence_length], dtype=np.int32)
            else:
                tokens = np.random.randint(0, 32000, size=sequence_length, dtype=np.int32)

            if len(tokens) < sequence_length:
                tokens = np.pad(tokens, (0, sequence_length - len(tokens)), constant_values=0)
            elif len(tokens) > sequence_length:
                tokens = tokens[:sequence_length]

            record_bytes = tokens.tobytes()
            writer.write(record_bytes)
            records_written += 1
            total_shard_bytes += len(record_bytes)

        writer.close()

        if is_native_gcs:
            from google.cloud import storage
            storage_client = storage.Client()
            bucket_name = clean_output.split("/")[0]
            blob_prefix = "/".join(clean_output.split("/")[1:])
            blob_path = f"{blob_prefix}/{shard_filename}".lstrip("/")
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(temp_local_out)
            final_dst = f"gs://{bucket_name}/{blob_path}"
        else:
            final_dst = os.path.join(clean_output, shard_filename)
            if temp_local_out != final_dst:
                import shutil
                shutil.move(temp_local_out, final_dst)

        if os.path.exists(temp_local_out):
            os.remove(temp_local_out)

        dur = time.perf_counter() - shard_start
        return idx, records_written, total_shard_bytes, dur, None, shard_filename

    except Exception as e:
        if os.path.exists(temp_local_out):
            os.remove(temp_local_out)
        return idx, 0, 0, 0, str(e), shard_filename


def convert_parquet_to_arrayrecord(
    input_path: str,
    output_path: str,
    text_column: str = "text",
    sequence_length: int = 2048,
    max_files: int = 0,
    num_workers: int = 64,
):
    """
    Main conversion orchestrator using ProcessPoolExecutor for true multi-core utilization.
    """
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

    total_files = len(parquet_files)
    logging.info(f"Starting Multi-Process conversion of {total_files} Parquet shards -> '{output_path}' ({num_workers} processes, GIL-free)")

    tasks = [
        (idx, fpath, total_files, clean_output, is_native_gcs, text_column, sequence_length)
        for idx, fpath in enumerate(parquet_files)
    ]

    start_conv = time.perf_counter()
    total_records = 0
    total_bytes = 0
    completed = 0
    shard_metadata_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(convert_single_shard_process, task): task[0] for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            idx, shard_recs, shard_b, dur, err, s_name = future.result()
            completed += 1
            if err:
                logging.error(f"  [Shard {idx+1}/{total_files}] Failed with error: {err}")
            else:
                total_records += shard_recs
                total_bytes += shard_b
                shard_metadata_list.append({
                    "shard_index": idx,
                    "filename": s_name,
                    "records": shard_recs,
                    "bytes": shard_b,
                })
                if completed % 25 == 0 or completed == total_files:
                    elapsed = time.perf_counter() - start_conv
                    cur_speed = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    rec_rate = total_records / elapsed if elapsed > 0 else 0
                    pct = (completed / total_files) * 100
                    logging.info(
                        f"  [Progress {completed}/{total_files} ({pct:.1f}%)] "
                        f"Converted {total_records} records ({total_bytes / (1024*1024):.1f} MB) in {elapsed:.1f}s "
                        f"[{cur_speed:.2f} MB/s, {rec_rate:.1f} rec/s]"
                    )

    total_duration = time.perf_counter() - start_conv
    throughput_mbs = (total_bytes / (1024 * 1024)) / total_duration if total_duration > 0 else 0.0
    rec_per_sec = total_records / total_duration if total_duration > 0 else 0.0

    # Sort manifest entries by shard index
    shard_metadata_list.sort(key=lambda x: x["shard_index"])
    manifest_data = {
        "dataset_format": "arrayrecord",
        "num_shards": total_files,
        "total_records": total_records,
        "total_bytes": total_bytes,
        "sequence_length": sequence_length,
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "shards": [s["filename"] for s in shard_metadata_list],
    }

    manifest_json_str = json.dumps(manifest_data, indent=2)
    import tempfile
    manifest_local_path = os.path.join(tempfile.gettempdir(), "manifest.json")
    with open(manifest_local_path, "w", encoding="utf-8") as mf:
        mf.write(manifest_json_str)

    if is_native_gcs:
        from google.cloud import storage
        storage_client = storage.Client()
        bucket_name = clean_output.split("/")[0]
        blob_prefix = "/".join(clean_output.split("/")[1:])
        manifest_blob_path = f"{blob_prefix}/manifest.json".lstrip("/")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(manifest_blob_path)
        blob.upload_from_filename(manifest_local_path)
        logging.info(f"✅ Generated dataset manifest: gs://{bucket_name}/{manifest_blob_path}")
    else:
        manifest_dest = os.path.join(clean_output, "manifest.json")
        import shutil
        shutil.copyfile(manifest_local_path, manifest_dest)
        logging.info(f"✅ Generated dataset manifest: {manifest_dest}")

    if os.path.exists(manifest_local_path):
        os.remove(manifest_local_path)

    logging.info("==================================================================================")
    logging.info("        PARQUET TO ARRAYRECORD MULTI-PROCESS CONVERSION COMPLETE                  ")
    logging.info("==================================================================================")
    logging.info(f"Input Shards          : {total_files} files")
    logging.info(f"Output Destination    : {output_path}")
    logging.info(f"Total Records Emitted : {total_records} records")
    logging.info(f"Total Converted Volume: {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024 * 1024 * 1024):.4f} GB)")
    logging.info(f"Total Conversion Time : {total_duration:.2f} s ({total_duration / 60:.2f} min)")
    logging.info(f"Multi-Process Speed   : {throughput_mbs:.2f} MB/s ({rec_per_sec:.2f} records/sec)")
    logging.info("==================================================================================")


def main():
    args = parse_args()
    convert_parquet_to_arrayrecord(
        args.input_path,
        args.output_path,
        args.text_column,
        args.sequence_length,
        args.max_files,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
