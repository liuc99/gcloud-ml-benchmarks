#!/usr/bin/env python3
"""
Comprehensive Dataset Loading Benchmark Script.

Measures dataset streaming and batch ingestion performance across GCSFuse,
native GCS (gcsfs), Managed Lustre, and local filesystems.
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import numpy as np
try:
    import torch
    from torch.utils.data import DataLoader, IterableDataset
except ImportError:
    torch = None
    DataLoader = None
    IterableDataset = object

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [BENCHMARK] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="ML Dataset Loading Benchmark")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset directory (e.g. /gcs/my-bucket/dataset, gs://my-bucket/dataset, /lustre/dataset)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "webdataset", "zarr", "pytorch_pt", "jsonl"],
        help="Dataset format to benchmark",
    )
    parser.add_argument(
        "--reader",
        type=str,
        default="hf_datasets",
        choices=["hf_datasets", "pure_hf", "webdataset", "tensorstore", "pytorch_loader", "pyarrow", "python_jsonl"],
        help="Dataset reader framework / library",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per step",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker sub-processes per rank",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches prefetched per worker",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Maximum number of batches to read before stopping benchmark",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epoch iterations",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=int(os.environ.get("RANK", "0")),
        help="DDP rank index",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", "1")),
        help="Total DDP ranks across cluster",
    )
    parser.add_argument(
        "--use-manifest",
        action="store_true",
        default=False,
        help="Use manifest.json for explicit shard list instead of globbing",
    )
    parser.add_argument(
        "--shuffle-strategy",
        type=str,
        default="none",
        choices=["none", "two_stage", "global"],
        help="Dataset shuffle strategy: 'none' (sequential), 'two_stage' (shard permutation + multi-worker streaming buffer), or 'global' (native for indexed formats like ArrayRecord; Parquet approximates via 2-stage due to row-group compression constraints)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10000,
        help="In-memory shuffle buffer size for streaming readers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
def hf_collate_fn(batch):
    if not batch:
        return {}
    first = batch[0]
    if isinstance(first, dict):
        res = {}
        for k in first.keys():
            vals = [item[k] for item in batch]
            try:
                res[k] = torch.tensor(vals)
            except Exception:
                res[k] = vals
        return res
    return batch


# -----------------------------------------------------------------------------
# 1. HuggingFace Parquet Reader
# -----------------------------------------------------------------------------
def benchmark_hf_parquet(dataset_path, batch_size, num_workers, prefetch_factor, max_batches, rank, world_size, use_manifest=False, shuffle_strategy="none", buffer_size=10000, seed=42):
    import datasets
    import datasets.distributed

    dataset_path = dataset_path.rstrip("/")
    is_direct_gcs = dataset_path.startswith("gs://")

    if is_direct_gcs:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        manifest_uri = f"{dataset_path}/manifest.json"
        
        if use_manifest and fs.exists(manifest_uri):
            with fs.open(manifest_uri, "r") as f:
                manifest_data = json.load(f)
            shard_names = manifest_data.get("shards", [])
            data_files = [f"{dataset_path}/{s}" for s in shard_names]
            logging.info(f"Loading HF streaming Parquet dataset from DIRECT GCS MANIFEST ({len(data_files)} shards from {manifest_uri}, shuffle={shuffle_strategy})")
        else:
            pattern = f"{dataset_path}/*.parquet"
            logging.info(f"Resolving shard list dynamically via Direct GCS fs.glob: {pattern} (shuffle={shuffle_strategy})...")
            glob_start = time.perf_counter()
            raw_shards = fs.glob(pattern)
            data_files = sorted([f if f.startswith("gs://") else f"gs://{f}" for f in raw_shards])
            glob_duration = time.perf_counter() - glob_start
            logging.info(f"Discovered {len(data_files)} shards via Direct GCS glob in {glob_duration:.4f}s")
    else:
        manifest_file = os.path.join(dataset_path, "manifest.json")
        if not os.path.exists(manifest_file):
            manifest_file = "/tmp/manifest.json"

        if use_manifest:
            if not os.path.exists(manifest_file):
                import glob
                logging.info(f"Manifest not found in dataset path, generating local manifest at {manifest_file}...")
                shards = sorted([f.replace(f"{dataset_path}/", "") for f in glob.glob(f"{dataset_path}/*.parquet")])
                with open(manifest_file, "w") as f:
                    json.dump({"shards": shards}, f)

            with open(manifest_file, "r") as f:
                manifest_data = json.load(f)
            shard_names = manifest_data.get("shards", [])
            data_files = [os.path.join(dataset_path, s) for s in shard_names]
            logging.info(f"Loading HF streaming Parquet dataset from POSIX MANIFEST ({len(data_files)} shards from {manifest_file}, shuffle={shuffle_strategy})")
        else:
            import glob
            pattern = f"{dataset_path}/*.parquet"
            logging.info(f"Resolving shard list dynamically via runtime POSIX glob: {pattern} (shuffle={shuffle_strategy})...")
            glob_start = time.perf_counter()
            data_files = sorted(glob.glob(pattern))
            glob_duration = time.perf_counter() - glob_start
            logging.info(f"Discovered {len(data_files)} shards via dynamic POSIX glob in {glob_duration:.4f}s")

    if shuffle_strategy == "global":
        logging.warning(
            "[FORMAT_LIMITATION] True single-sample Global Shuffle is physically unsupported in columnar Parquet formats "
            "due to Row-Group compression (arbitrary row seeks incur 5,000x-50,000x decompression & I/O amplification). "
            "Approximating global shuffle via 2-Stage streaming (Full shard permutation + large sliding sample buffer). "
            "For deterministic True Global Shuffle with zero duplication and single-sample index seeking, use ArrayRecord + Grain."
        )
        import random
        rng = random.Random(seed)
        rng.shuffle(data_files)
        logging.info(f"Applying Global Full Shard Permutation ({len(data_files)} shards, seed={seed})")

    prep_start = time.perf_counter()
    ds = datasets.load_dataset("parquet", data_files=data_files, split="train", streaming=True)

    if shuffle_strategy in ("two_stage", "global"):
        logging.info(f"Applying Streaming Shuffle Buffer with buffer_size={buffer_size}, seed={seed} (strategy={shuffle_strategy})")
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed)

    if world_size > 1:
        ds = datasets.distributed.split_dataset_by_node(ds, rank=rank, world_size=world_size)

    mp_context = None
    if num_workers > 0 and dataset_path.startswith("gs://"):
        import torch.multiprocessing as mp
        mp_context = mp.get_context("spawn")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=hf_collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context=mp_context,
    )

    prep_duration = time.perf_counter() - prep_start
    logging.info(f"HF dataloader prepared in {prep_duration:.4f}s")

    return loader


def benchmark_pure_hf_parquet(dataset_path, batch_size, max_batches, rank, world_size, use_manifest=False):
    """Pure HuggingFace streaming reader without PyTorch DataLoader (using ds.iter(batch_size))."""
    import datasets
    import datasets.distributed

    dataset_path = dataset_path.rstrip("/")
    manifest_file = os.path.join(dataset_path, "manifest.json")
    if not os.path.exists(manifest_file):
        manifest_file = "/tmp/manifest.json"

    if use_manifest:
        if not os.path.exists(manifest_file):
            import glob
            logging.info(f"Generating local manifest at {manifest_file}...")
            shards = sorted([os.path.basename(f) for f in glob.glob(f"{dataset_path}/*.parquet")])
            with open(manifest_file, "w") as f:
                json.dump({"shards": shards}, f)

        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)
        shard_names = manifest_data.get("shards", [])
        data_files = [os.path.join(dataset_path, s) for s in shard_names]
        logging.info(f"Loading Pure HF streaming Parquet dataset from MANIFEST ({len(data_files)} shards)")
        prep_start = time.perf_counter()
        ds = datasets.load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    else:
        file_pattern = f"{dataset_path}/*.parquet"
        logging.info(f"Loading Pure HF streaming Parquet dataset from GLOB PATTERN: {file_pattern}")
        prep_start = time.perf_counter()
        ds = datasets.load_dataset("parquet", data_files=file_pattern, split="train", streaming=True)

    if world_size > 1:
        ds = datasets.distributed.split_dataset_by_node(ds, rank=rank, world_size=world_size)

    loader = ds.iter(batch_size=batch_size)

    prep_duration = time.perf_counter() - prep_start
    logging.info(f"Pure HF streaming iterator prepared in {prep_duration:.4f}s")

    return loader


# -----------------------------------------------------------------------------
# 2. WebDataset Reader
# -----------------------------------------------------------------------------
def benchmark_webdataset(dataset_path, batch_size, num_workers, prefetch_factor, max_batches, rank, world_size):
    import webdataset as wds

    dataset_path = dataset_path.rstrip("/")
    urls = f"{dataset_path}/shard_*.tar"
    logging.info(f"Loading WebDataset TAR shards from: {urls}")
    prep_start = time.perf_counter()

    ds = wds.WebDataset(urls, shardshuffle=False, nodesplitter=wds.split_by_node if world_size > 1 else None)
    ds = ds.decode().to_tuple("npy", "json")
    
    loader = wds.WebLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    prep_duration = time.perf_counter() - prep_start
    logging.info(f"WebDataset loader prepared in {prep_duration:.4f}s")
    return loader


def benchmark_pyarrow(dataset_path, batch_size, num_workers, prefetch_factor, max_batches, rank, world_size):
    """Pure C++ PyArrow Dataset Scanner without Python row iteration (zero-copy RecordBatch streaming)."""
    import glob
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    dataset_path = dataset_path.rstrip("/")
    logging.info(f"Loading PyArrow C++ Dataset Scanner from: {dataset_path}")
    prep_start = time.perf_counter()

    if dataset_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        all_files = sorted(fs.glob(f"{dataset_path}/*.parquet"))
        gcs_fs = pafs.GcsFileSystem()
        # pyarrow fs expects paths without leading gs:// or slash
        files = [f.replace("gs://", "").lstrip("/") for f in all_files]
        if world_size > 1:
            files = files[rank::world_size]
        dataset = ds.dataset(files, filesystem=gcs_fs, format="parquet")
    else:
        files = sorted(glob.glob(f"{dataset_path}/*.parquet"))
        if world_size > 1:
            files = files[rank::world_size]
        dataset = ds.dataset(files, format="parquet")

    # C++ Multi-threaded Scanner (bypasses Python GIL, decodes Arrow RecordBatches directly in C++)
    scanner = dataset.scanner(batch_size=batch_size, use_threads=True)
    loader = scanner.to_batches()

    prep_duration = time.perf_counter() - prep_start
    logging.info(f"PyArrow C++ Scanner prepared in {prep_duration:.4f}s with {len(files)} parquet files")
    return loader


# -----------------------------------------------------------------------------
# 3. TensorStore / Zarr Reader
# -----------------------------------------------------------------------------
class TensorStoreDataset(IterableDataset):
    def __init__(self, dataset_path, batch_size, rank, world_size):
        import tensorstore as ts

        spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "gcs" if dataset_path.startswith("gs://") else "file",
                "path": dataset_path.replace("gs://", "").rstrip("/"),
            },
        }
        self.ts_dataset = ts.open(spec).result()
        self.total_samples = self.ts_dataset.shape[0]
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        samples_per_rank = self.total_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank

        for i in range(start_idx, end_idx, self.batch_size):
            b_end = min(end_idx, i + self.batch_size)
            batch_data = self.ts_dataset[i:b_end].read().result()
            yield torch.from_numpy(batch_data)


def benchmark_tensorstore(dataset_path, batch_size, num_workers, prefetch_factor, max_batches, rank, world_size):
    prep_start = time.perf_counter()
    ds = TensorStoreDataset(dataset_path, batch_size, rank, world_size)
    loader = DataLoader(ds, batch_size=None, num_workers=0)
    prep_duration = time.perf_counter() - prep_start
    logging.info(f"TensorStore dataloader prepared in {prep_duration:.4f}s")
    return loader


# -----------------------------------------------------------------------------
# 4. PyTorch PT Files Reader
# -----------------------------------------------------------------------------
class PyTorchPtDataset(IterableDataset):
    def __init__(self, dataset_path, rank, world_size):
        self.dataset_path = dataset_path.rstrip("/")
        self.is_gcsfs = self.dataset_path.startswith("gs://")
        if self.is_gcsfs:
            import gcsfs
            self.fs = gcsfs.GCSFileSystem()
            all_files = sorted(self.fs.glob(f"{self.dataset_path}/*.pt"))
            self.files = [f"gs://{f}" for f in all_files]
        else:
            all_files = sorted([os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith(".pt")])
            self.files = all_files

        # Rank sharding
        self.files = self.files[rank::world_size]

    def __iter__(self):
        for fpath in self.files:
            if self.is_gcsfs:
                with self.fs.open(fpath, "rb") as f:
                    data = torch.load(f)
            else:
                data = torch.load(fpath)
            
            inputs = data["inputs"]
            labels = data["labels"]
            for i in range(inputs.shape[0]):
                yield {"input_ids": inputs[i], "label": labels[i]}


def benchmark_pytorch_pt(dataset_path, batch_size, num_workers, prefetch_factor, max_batches, rank, world_size):
    prep_start = time.perf_counter()
    ds = PyTorchPtDataset(dataset_path, rank, world_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    prep_duration = time.perf_counter() - prep_start
    logging.info(f"PyTorch PT dataloader prepared in {prep_duration:.4f}s")
    return loader


class JsonlDataset(IterableDataset):
    def __init__(self, dataset_path, rank, world_size):
        self.dataset_path = dataset_path.rstrip("/")
        self.is_gcsfs = self.dataset_path.startswith("gs://")
        if self.is_gcsfs:
            import gcsfs
            self.fs = gcsfs.GCSFileSystem()
            all_files = sorted(self.fs.glob(f"{self.dataset_path}/*.jsonl"))
            self.files = [f"gs://{f}" for f in all_files]
        else:
            all_files = sorted([os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith(".jsonl")])
            self.files = all_files

        # Rank sharding
        self.files = self.files[rank::world_size]

    def __iter__(self):
        for fpath in self.files:
            if self.is_gcsfs:
                with self.fs.open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        yield json.loads(line)
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        yield json.loads(line)


def benchmark_jsonl(dataset_path, batch_size, num_workers, prefetch_factor, max_batches, rank, world_size):
    prep_start = time.perf_counter()
    ds = JsonlDataset(dataset_path, rank, world_size)
    if DataLoader is not None:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
    else:
        # Fallback pure-python batching when PyTorch is not installed
        def pure_python_loader():
            batch = []
            for item in ds:
                batch.append(item)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        loader = pure_python_loader()

    prep_duration = time.perf_counter() - prep_start
    logging.info(f"JSONL dataloader prepared in {prep_duration:.4f}s")
    return loader


# -----------------------------------------------------------------------------
# Execution & Metrics Loop
# -----------------------------------------------------------------------------
def run_benchmark(loader, max_batches):
    logging.info("Starting dataset read benchmarking loop...")
    batch_latencies = []
    total_samples = 0
    total_bytes = 0

    bench_start = time.perf_counter()
    first_batch_time = None

    for step, batch in enumerate(loader):
        now = time.perf_counter()
        if step == 0:
            first_batch_time = now - bench_start
            logging.info(f"Time to First Batch (TTFB): {first_batch_time * 1000:.2f} ms ({first_batch_time:.4f} s)")

        step_start = time.perf_counter()

        # Extract sample count and size estimation
        if hasattr(batch, "num_rows") and hasattr(batch, "nbytes"):
            num_samples = batch.num_rows
            batch_bytes = batch.nbytes
        elif isinstance(batch, (list, tuple)):
            num_samples = len(batch)
            batch_bytes = sum(len(str(item).encode("utf-8")) for item in batch)
        elif isinstance(batch, dict):
            first_val = next(iter(batch.values()))
            num_samples = len(first_val) if hasattr(first_val, "__len__") else 1
            batch_bytes = sum(
                val.element_size() * val.nelement()
                if torch is not None and isinstance(val, torch.Tensor)
                else len(str(val).encode("utf-8"))
                for val in batch.values()
            )
        elif torch is not None and isinstance(batch, torch.Tensor):
            num_samples = batch.shape[0]
            batch_bytes = batch.element_size() * batch.nelement()
        else:
            num_samples = 64
            batch_bytes = num_samples * 1024

        total_samples += num_samples
        total_bytes += batch_bytes

        step_duration = time.perf_counter() - step_start
        batch_latencies.append(step_duration)

        if (step + 1) % 20 == 0:
            avg_lat = np.mean(batch_latencies[-20:]) * 1000
            logging.info(f"  [Step {step+1}/{max_batches}] Recent 20-batch avg latency: {avg_lat:.2f} ms")

        if step + 1 >= max_batches:
            break

    total_duration = time.perf_counter() - bench_start
    
    # Calculate summary metrics
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024
    throughput_mbs = total_mb / total_duration if total_duration > 0 else 0.0
    throughput_gbps = (throughput_mbs * 8) / 1024
    samples_per_sec = total_samples / total_duration if total_duration > 0 else 0.0

    p50_ms = np.percentile(batch_latencies, 50) * 1000 if batch_latencies else 0.0
    p95_ms = np.percentile(batch_latencies, 95) * 1000 if batch_latencies else 0.0
    p99_ms = np.percentile(batch_latencies, 99) * 1000 if batch_latencies else 0.0

    logging.info("==================================================================================")
    logging.info("                             DATASET LOADING SUMMARY                              ")
    logging.info("==================================================================================")
    logging.info(f"Total Batches Read       : {len(batch_latencies)}")
    logging.info(f"Total Samples Ingested   : {total_samples} samples")
    logging.info(f"Total Data Volume        : {total_mb:.2f} MB ({total_gb:.4f} GB)")
    logging.info(f"Total Ingestion Duration : {total_duration:.4f} seconds")
    logging.info(f"Time to First Batch TTFB : {first_batch_time * 1000:.2f} ms ({first_batch_time:.4f} s)")
    logging.info(f"Read Throughput          : {throughput_mbs:.2f} MB/s ({throughput_gbps:.2f} Gbps)")
    logging.info(f"Ingestion Speed          : {samples_per_sec:.2f} samples/sec")
    logging.info(f"Batch Latency p50 / p95  : {p50_ms:.2f} ms / {p95_ms:.2f} ms")
    logging.info(f"Batch Latency p99        : {p99_ms:.2f} ms")
    logging.info("==================================================================================")


def main():
    args = parse_args()
    logging.info(
        f"Starting Dataset Loading Benchmark: path={args.dataset_path}, format={args.format}, reader={args.reader}, "
        f"batch_size={args.batch_size}, num_workers={args.num_workers}, rank={args.rank}/{args.world_size}"
    )

    if args.reader == "hf_datasets":
        loader = benchmark_hf_parquet(
            args.dataset_path, args.batch_size, args.num_workers, args.prefetch_factor, args.max_batches, args.rank, args.world_size,
            use_manifest=args.use_manifest, shuffle_strategy=args.shuffle_strategy, buffer_size=args.buffer_size, seed=args.seed
        )
    elif args.reader == "pure_hf":
        loader = benchmark_pure_hf_parquet(
            args.dataset_path, args.batch_size, args.max_batches, args.rank, args.world_size, use_manifest=args.use_manifest
        )
    elif args.reader == "pyarrow":
        loader = benchmark_pyarrow(
            args.dataset_path, args.batch_size, args.num_workers, args.prefetch_factor, args.max_batches, args.rank, args.world_size
        )
    elif args.reader == "webdataset":
        loader = benchmark_webdataset(
            args.dataset_path, args.batch_size, args.num_workers, args.prefetch_factor, args.max_batches, args.rank, args.world_size
        )
    elif args.reader == "tensorstore":
        loader = benchmark_tensorstore(
            args.dataset_path, args.batch_size, args.num_workers, args.prefetch_factor, args.max_batches, args.rank, args.world_size
        )
    elif args.reader == "pytorch_loader":
        loader = benchmark_pytorch_pt(
            args.dataset_path, args.batch_size, args.num_workers, args.prefetch_factor, args.max_batches, args.rank, args.world_size
        )
    elif args.reader == "python_jsonl":
        loader = benchmark_jsonl(
            args.dataset_path, args.batch_size, args.num_workers, args.prefetch_factor, args.max_batches, args.rank, args.world_size
        )
    else:
        raise ValueError(f"Unsupported reader: {args.reader}")

    for epoch in range(args.epochs):
        logging.info(f"--- Starting Epoch {epoch+1}/{args.epochs} ---")
        run_benchmark(loader, args.max_batches)


if __name__ == "__main__":
    main()
