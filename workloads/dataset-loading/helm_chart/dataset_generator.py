#!/usr/bin/env python3
"""
Synthetic Dataset Generator for ML Storage Benchmarks.

Generates synthetic ML datasets (Parquet, WebDataset TAR, Zarr/TensorStore, PyTorch PT)
directly to local storage, GCSFuse mounts (/gcs/...), native GCS (gs://...), or Lustre mounts (/lustre/...).
"""

import argparse
import io
import json
import logging
import math
import os
import sys
import tarfile
import time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [DATASET_GEN] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic ML Dataset Generator")
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Target output directory or bucket path (e.g., /gcs/my-bucket/dataset, gs://my-bucket/dataset, /lustre/dataset)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "webdataset", "zarr", "pytorch_pt", "jsonl"],
        help="Dataset storage format",
    )
    parser.add_argument(
        "--total-size-mb",
        type=float,
        default=1024.0,
        help="Total target dataset size in Megabytes (default: 1024 MB = 1 GB)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="Number of shard files / partitions to generate",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Tokens / Sequence length per sample",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension per sample feature (if vector-based)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing files in target path",
    )
    return parser.parse_args()


def generate_parquet_shards(output_path, total_size_mb, num_files, sequence_length):
    import pyarrow as pa
    import pyarrow.parquet as pq

    target_bytes_per_file = (total_size_mb * 1024 * 1024) / num_files
    bytes_per_row = max(128, sequence_length * 8 + 64)
    rows_per_file = int(math.ceil(target_bytes_per_file / bytes_per_row))

    logging.info(
        f"Generating {num_files} Parquet files with ~{rows_per_file} rows/file (target ~{target_bytes_per_file / 1e6:.2f} MB each)..."
    )

    is_gcsfs = output_path.startswith("gs://")
    fs = None
    if is_gcsfs:
        import gcsfs

        fs = gcsfs.GCSFileSystem()

    if not is_gcsfs:
        os.makedirs(output_path, exist_ok=True)

    start_time = time.perf_counter()
    total_written_bytes = 0

    for i in range(num_files):
        file_start = time.perf_counter()
        token_ids = np.random.randint(0, 32000, size=(rows_per_file, sequence_length), dtype=np.int64)
        labels = np.random.randint(0, 1000, size=rows_per_file, dtype=np.int32)
        sample_ids = np.arange(i * rows_per_file, (i + 1) * rows_per_file, dtype=np.int64)

        table = pa.Table.from_arrays(
            [
                pa.array(sample_ids),
                pa.array(token_ids.tolist()),
                pa.array(labels),
            ],
            names=["sample_id", "input_ids", "label"],
        )

        file_path = f"{output_path}/shard_{i:05d}.parquet"
        if is_gcsfs:
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
        f"✅ Parquet dataset generation complete: {total_mb:.2f} MB generated in {total_duration:.2f}s ({total_mb / total_duration:.2f} MB/s)"
    )


def generate_webdataset_shards(output_path, total_size_mb, num_files, sequence_length, embedding_dim):
    target_bytes_per_file = (total_size_mb * 1024 * 1024) / num_files
    bytes_per_sample = max(256, embedding_dim * 4 + 200)
    samples_per_file = int(math.ceil(target_bytes_per_file / bytes_per_sample))

    logging.info(
        f"Generating {num_files} WebDataset TAR shards with ~{samples_per_file} samples/shard..."
    )

    is_gcsfs = output_path.startswith("gs://")
    if is_gcsfs:
        raise ValueError("WebDataset tar generator requires a local or mounted directory path (e.g., /gcs/... or /lustre/...)")

    os.makedirs(output_path, exist_ok=True)
    start_time = time.perf_counter()
    total_written_bytes = 0

    for i in range(num_files):
        tar_filename = os.path.join(output_path, f"shard_{i:05d}.tar")
        file_start = time.perf_counter()
        
        with tarfile.open(tar_filename, "w") as tar:
            for s in range(samples_per_file):
                key = f"sample_{i:05d}_{s:06d}"
                
                # 1. Feature tensor payload (.npy bytes)
                feat_array = np.random.randn(sequence_length, embedding_dim).astype(np.float32)
                feat_bytes = feat_array.tobytes()
                
                ti_feat = tarfile.TarInfo(name=f"{key}.npy")
                ti_feat.size = len(feat_bytes)
                tar.addfile(ti_feat, io.BytesIO(feat_bytes))
                
                # 2. Metadata (.json)
                meta = {"key": key, "shard": i, "sample_idx": s, "label": s % 100}
                meta_bytes = json.dumps(meta).encode("utf-8")
                ti_meta = tarfile.TarInfo(name=f"{key}.json")
                ti_meta.size = len(meta_bytes)
                tar.addfile(ti_meta, io.BytesIO(meta_bytes))

        written_bytes = os.path.getsize(tar_filename)
        total_written_bytes += written_bytes
        elapsed = time.perf_counter() - file_start
        logging.info(
            f"  [Shard {i+1}/{num_files}] Wrote {tar_filename} ({written_bytes / 1e6:.2f} MB) in {elapsed:.2f}s"
        )

    total_duration = time.perf_counter() - start_time
    total_mb = total_written_bytes / (1024 * 1024)
    logging.info(
        f"✅ WebDataset generation complete: {total_mb:.2f} MB in {total_duration:.2f}s ({total_mb / total_duration:.2f} MB/s)"
    )


def generate_zarr_shards(output_path, total_size_mb, num_files, sequence_length, embedding_dim):
    import tensorstore as ts

    target_bytes = total_size_mb * 1024 * 1024
    num_samples = int(target_bytes / (sequence_length * embedding_dim * 4))
    num_samples = max(num_samples, num_files * 10)

    shape = [num_samples, sequence_length, embedding_dim]
    chunk_shape = [max(1, num_samples // num_files), sequence_length, embedding_dim]

    logging.info(
        f"Generating Zarr/TensorStore array dataset at {output_path} with shape {shape}, chunk shape {chunk_shape}..."
    )

    spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "gcs" if output_path.startswith("gs://") else "file",
            "path": output_path.replace("gs://", ""),
        },
        "metadata": {
            "dtype": "<f4",
            "shape": shape,
            "chunks": chunk_shape,
        },
        "create": True,
        "delete_existing": True,
    }

    start_time = time.perf_counter()
    dataset = ts.open(spec).result()

    chunk_size_samples = chunk_shape[0]
    for c in range(num_files):
        start_idx = c * chunk_size_samples
        end_idx = min(num_samples, (c + 1) * chunk_size_samples)
        if start_idx >= num_samples:
            break
        data_chunk = np.random.randn(end_idx - start_idx, sequence_length, embedding_dim).astype(np.float32)
        dataset[start_idx:end_idx].write(data_chunk).result()
        logging.info(f"  [Chunk {c+1}/{num_files}] Wrote rows [{start_idx}:{end_idx}]")

    duration = time.perf_counter() - start_time
    logging.info(f"✅ Zarr/TensorStore dataset generation complete in {duration:.2f}s")


def generate_pytorch_pt_shards(output_path, total_size_mb, num_files, sequence_length, embedding_dim):
    import torch

    target_bytes_per_file = (total_size_mb * 1024 * 1024) / num_files
    bytes_per_sample = sequence_length * embedding_dim * 4
    samples_per_file = max(1, int(math.ceil(target_bytes_per_file / bytes_per_sample)))

    logging.info(
        f"Generating {num_files} PyTorch .pt shard files with {samples_per_file} samples each..."
    )

    is_gcsfs = output_path.startswith("gs://")
    if is_gcsfs:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
    else:
        os.makedirs(output_path, exist_ok=True)

    start_time = time.perf_counter()
    total_written_bytes = 0

    for i in range(num_files):
        file_start = time.perf_counter()
        tensors = {
            "inputs": torch.randn(samples_per_file, sequence_length, embedding_dim, dtype=torch.float32),
            "labels": torch.randint(0, 1000, (samples_per_file,)),
        }
        file_path = f"{output_path}/shard_{i:05d}.pt"
        if is_gcsfs:
            with fs.open(file_path, "wb") as f:
                torch.save(tensors, f)
            written_bytes = fs.du(file_path)
        else:
            torch.save(tensors, file_path)
            written_bytes = os.path.getsize(file_path)

        total_written_bytes += written_bytes
        elapsed = time.perf_counter() - file_start
        logging.info(
            f"  [Shard {i+1}/{num_files}] Wrote {file_path} ({written_bytes / 1e6:.2f} MB) in {elapsed:.2f}s"
        )

    total_duration = time.perf_counter() - start_time
    total_mb = total_written_bytes / (1024 * 1024)
    logging.info(
        f"✅ PyTorch PT dataset generation complete: {total_mb:.2f} MB in {total_duration:.2f}s ({total_mb / total_duration:.2f} MB/s)"
    )


def generate_jsonl_shards(output_path, total_size_mb, num_files, sequence_length):
    target_bytes_per_file = (total_size_mb * 1024 * 1024) / num_files
    # Create sample string text
    words = ["llama", "benchmark", "gcs", "gcsfuse", "lustre", "dataset", "throughput", "training", "cloud", "google"]
    sample_text = " ".join(np.random.choice(words, size=sequence_length))
    sample_obj = {"id": 0, "text": sample_text, "label": 1}
    sample_json = json.dumps(sample_obj) + "\n"
    bytes_per_row = len(sample_json.encode("utf-8"))
    rows_per_file = int(math.ceil(target_bytes_per_file / bytes_per_row))

    logging.info(
        f"Generating {num_files} JSONL files with ~{rows_per_file} rows/file..."
    )

    is_gcsfs = output_path.startswith("gs://")
    if is_gcsfs:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
    else:
        os.makedirs(output_path, exist_ok=True)

    start_time = time.perf_counter()
    total_written_bytes = 0

    for i in range(num_files):
        file_start = time.perf_counter()
        file_path = f"{output_path}/shard_{i:05d}.jsonl"
        
        lines = []
        for s in range(rows_per_file):
            lines.append(json.dumps({"id": i * rows_per_file + s, "text": sample_text, "label": s % 100}) + "\n")
        
        content = "".join(lines).encode("utf-8")
        if is_gcsfs:
            with fs.open(file_path, "wb") as f:
                f.write(content)
            written_bytes = len(content)
        else:
            with open(file_path, "wb") as f:
                f.write(content)
            written_bytes = len(content)

        total_written_bytes += written_bytes
        elapsed = time.perf_counter() - file_start
        logging.info(
            f"  [Shard {i+1}/{num_files}] Wrote {file_path} ({written_bytes / 1e6:.2f} MB) in {elapsed:.2f}s"
        )

    total_duration = time.perf_counter() - start_time
    total_mb = total_written_bytes / (1024 * 1024)
    logging.info(
        f"✅ JSONL dataset generation complete: {total_mb:.2f} MB in {total_duration:.2f}s ({total_mb / total_duration:.2f} MB/s)"
    )


def main():
    args = parse_args()
    logging.info(
        f"Starting dataset generation: format={args.format}, output={args.output_path}, target_size={args.total_size_mb}MB"
    )

    output_path = args.output_path.rstrip("/")

    if args.format == "parquet":
        generate_parquet_shards(output_path, args.total_size_mb, args.num_files, args.sequence_length)
    elif args.format == "webdataset":
        generate_webdataset_shards(
            output_path, args.total_size_mb, args.num_files, args.sequence_length, args.embedding_dim
        )
    elif args.format == "zarr":
        generate_zarr_shards(
            output_path, args.total_size_mb, args.num_files, args.sequence_length, args.embedding_dim
        )
    elif args.format == "pytorch_pt":
        generate_pytorch_pt_shards(
            output_path, args.total_size_mb, args.num_files, args.sequence_length, args.embedding_dim
        )
    elif args.format == "jsonl":
        generate_jsonl_shards(
            output_path, args.total_size_mb, args.num_files, args.sequence_length
        )
    else:
        raise ValueError(f"Unsupported format: {args.format}")


if __name__ == "__main__":
    main()
