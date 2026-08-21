#!/usr/bin/env python3
"""
MaxText In-Tree Standalone DataLoader Benchmark.

Wraps and executes MaxText's native data loading pipeline (Grain, TFDS, ArrayRecord, Parquet)
measuring end-to-end data ingestion speed, TTFB, per-step batch latency, and Tokenizer CPU overhead.
Supports full Two-Stage Shuffle (4-stream interleaved shard streams + sliding window shuffle buffer).
Supports execution on TPU, GPU, or CPU.
"""

import argparse
import datetime
import json
import logging
import math
import os
import queue
import sys
import threading
import time
import bisect
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MAXTEXT_IN_TREE] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="MaxText In-Tree Standalone DataLoader Benchmark")
    parser.add_argument(
        "--config-path",
        type=str,
        default="src/maxtext/configs/base.yml",
        help="Path to MaxText base YAML config",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="maxtext_dataloader_bench",
        help="MaxText run name identifier",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="grain_array_record",
        help="Dataset type: 'grain_array_record', 'grain_parquet', 'synthetic', 'tfds', 'c4_mlperf'",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Dataset directory or GCS bucket path (e.g. gs://bucket/data or /gcs/bucket/data)",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="auto",
        choices=["auto", "parquet", "arrayrecord"],
        help="Dataset format to benchmark ('parquet' or 'arrayrecord')",
    )
    parser.add_argument(
        "--use-manifest",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to prioritize manifest.json over dynamic directory globbing ('true' or 'false')",
    )
    parser.add_argument(
        "--shuffle-mode",
        type=str,
        default="two_stage",
        choices=["none", "two_stage", "global"],
        help="Shuffle mode ('none', 'two_stage', 'global')",
    )
    parser.add_argument(
        "--num-streams",
        type=int,
        default=4,
        help="Number of concurrent shard streams for multi-stream interleaving",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=1024,
        help="In-memory sliding window shuffle buffer capacity (samples)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of batches/steps to load",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=128,
        help="Batch size per device",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Model sequence length",
    )
    parser.add_argument(
        "--chunk-records",
        type=int,
        default=1,
        help="Number of records per Range Read chunk (default: 1 = 8 KB point seek)",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="cpu",
        choices=["cpu", "tpu", "gpu"],
        help="Target hardware platform",
    )
    parser.add_argument(
        "--extra-args",
        nargs="*",
        default=[],
        help="Additional key=value config overrides passed to MaxText pyconfig",
    )
    return parser.parse_known_args()


def run_standalone_dataloader(config_args, extra_args):
    """
    Executes MaxText DataLoader with step-by-step performance instrumentation.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    if config_args.hardware == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    try:
        import jax
        from maxtext.configs import pyconfig
        from maxtext.trainers.pre_train.train import get_first_step
        from maxtext.common.data_loader import DataLoader
        from maxtext.utils import max_logging
        from maxtext.utils.train_utils import validate_train_config, setup_train_loop
        native_maxtext_available = True
    except ImportError as e:
        native_maxtext_available = False

    if not native_maxtext_available or config_args.dataset_format != "auto" or config_args.dataset_path:
        return run_end_to_end_dataloader_bench(config_args)

    argv = [
        sys.argv[0],
        config_args.config_path,
        f"run_name={config_args.run_name}",
        f"dataset_type={config_args.dataset_type}",
        f"steps={config_args.steps}",
        f"per_device_batch_size={config_args.per_device_batch_size}",
    ]
    if config_args.dataset_path:
        argv.append(f"dataset_path={config_args.dataset_path}")
    if config_args.hardware == "cpu":
        argv.extend([
            "ici_data_parallelism=1",
            "ici_fsdp_parallelism=1",
            "ici_tensor_parallelism=1",
        ])
    argv.extend(extra_args)

    logging.info(f"Initializing MaxText pyconfig with argv: {argv[1:]}")
    config = pyconfig.initialize(argv)
    validate_train_config(config)

    device_count = jax.device_count()
    process_count = jax.process_count()
    process_index = jax.process_index()

    if process_index == 0:
        logging.info(f"JAX Topology: {device_count} devices across {process_count} processes. Devices: {jax.devices()}")

    if config.dataset_type in ("tfds", "c4_mlperf") and config.dataset_path:
        os.environ["TFDS_DATA_DIR"] = config.dataset_path

    logging.info("Calling setup_train_loop to initialize mesh and native data_iterator...")
    _, _, _, model, mesh, _, data_iterator, _, _, _, state = setup_train_loop(config, recorder=None)
    data_loader = DataLoader(config, mesh, data_iterator, None)

    step_latencies_ms = []
    bench_start = time.perf_counter()

    t0 = time.perf_counter()
    example_batch = data_loader.load_next_batch()
    jax.block_until_ready(example_batch)
    first_batch_dur_sec = time.perf_counter() - t0
    first_batch_dur_ms = first_batch_dur_sec * 1000.0
    step_latencies_ms.append(first_batch_dur_ms)

    if process_index == 0:
        logging.info(f"STANDALONE DATALOADER : First step completed in {first_batch_dur_sec:.4f} seconds ({first_batch_dur_ms:.2f} ms), on host 0")

    start_step = get_first_step(model, state)
    total_steps = config.steps
    total_samples = config.per_device_batch_size * max(1, device_count)

    for step in np.arange(start_step + 1, total_steps):
        s_start = time.perf_counter()
        example_batch = data_loader.load_next_batch()
        jax.block_until_ready(example_batch)
        s_dur_ms = (time.perf_counter() - s_start) * 1000.0
        step_latencies_ms.append(s_dur_ms)
        total_samples += config.per_device_batch_size * max(1, device_count)

        if (step + 1) % 20 == 0 and process_index == 0:
            logging.info(f"  [MaxText Step {step + 1}/{total_steps}] Batch latency: {s_dur_ms:.2f} ms")

    total_duration_sec = time.perf_counter() - bench_start

    if process_index == 0:
        avg_batch_ms = np.mean(step_latencies_ms)
        p50_batch_ms = np.percentile(step_latencies_ms, 50)
        p95_batch_ms = np.percentile(step_latencies_ms, 95)
        p99_batch_ms = np.percentile(step_latencies_ms, 99)
        samples_per_sec = total_samples / total_duration_sec if total_duration_sec > 0 else 0.0

        logging.info(f"STANDALONE DATALOADER : {total_steps} batches loaded in {total_duration_sec:.4f} seconds, on host 0")
        logging.info("==================================================================================")
        logging.info("                 MAXTEXT IN-TREE LOADER BENCHMARK SUMMARY                         ")
        logging.info("==================================================================================")
        logging.info(f"Dataset Type             : {config.dataset_type}")
        logging.info(f"Dataset Path             : {config.dataset_path or 'N/A'}")
        logging.info(f"Hardware Platform        : {config_args.hardware.upper()} ({device_count} devices)")
        logging.info(f"Total Batches Ingested   : {total_steps} batches")
        logging.info(f"Total Samples Ingested   : {total_samples} samples")
        logging.info(f"Time to First Batch TTFB : {first_batch_dur_ms:.2f} ms ({first_batch_dur_sec:.4f} s)")
        logging.info(f"Sample Ingestion Speed   : {samples_per_sec:.2f} samples/sec")
        logging.info(f"Batch Load Latency (Avg) : {avg_batch_ms:.2f} ms")
        logging.info(f"Batch Load Latency (p50) : {p50_batch_ms:.2f} ms")
        logging.info(f"Batch Load Latency (p95) : {p95_batch_ms:.2f} ms")
        logging.info(f"Batch Load Latency (p99) : {p99_batch_ms:.2f} ms")
        logging.info("==================================================================================")


def run_end_to_end_dataloader_bench(config_args):
    """
    Executes a high-fidelity MaxText/Grain end-to-end data pipeline with full Two-Stage Shuffle:
      1. Parquet with runtime CPU BPE Tokenization + Interleaved Multi-Stream + Window Shuffle Buffer
      2. ArrayRecord with Pre-tokenized Zero-CPU + Interleaved Multi-Stream + Window Shuffle Buffer
    """
    dpath = config_args.dataset_path
    batch_size = config_args.per_device_batch_size
    total_steps = config_args.steps
    seq_len = config_args.sequence_length
    num_streams = max(1, config_args.num_streams)
    shuffle_mode = config_args.shuffle_mode
    buf_size = max(batch_size * 2, config_args.shuffle_buffer_size)

    # Determine format
    is_arrayrecord = False
    if config_args.dataset_format == "arrayrecord" or "arrayrecord" in dpath.lower():
        is_arrayrecord = True
    elif config_args.dataset_format == "parquet" or "parquet" in dpath.lower():
        is_arrayrecord = False
    else:
        if os.path.exists(dpath):
            files = os.listdir(dpath)
            is_arrayrecord = any(f.endswith(".array_record") for f in files)

    fmt_name = "ARRAYRECORD (Pre-Tokenized Zero-CPU)" if is_arrayrecord else "PARQUET (With Runtime CPU Tokenizer)"
    logging.info(f"Starting MaxText Standalone DataLoader Benchmark: format={fmt_name}, path={dpath}, steps={total_steps}, batch_size={batch_size}, shuffle={shuffle_mode}, streams={num_streams}")

    files = []
    used_manifest = False
    pipeline_init_start = time.perf_counter()
    discovery_start = pipeline_init_start
    idx_dur_ms = 0.0

    if config_args.use_manifest.lower() in ("true", "1", "yes"):
        manifest_path = os.path.join(dpath, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    mdata = json.load(mf)
                    if "shards" in mdata:
                        files = [os.path.join(dpath, s) for s in mdata["shards"]]
                        used_manifest = True
                        logging.info(f"✅ Discovered {len(files)} dataset shards instantly via manifest.json (Zero GCS Listing Overhead)")
            except Exception as e:
                logging.warning(f"Failed to read manifest.json: {e}, falling back to directory scanning")

    if not files:
        import glob
        if os.path.isdir(dpath):
            pattern = os.path.join(dpath, "*.array_record" if is_arrayrecord else "*.parquet")
            files = sorted(glob.glob(pattern))
            if not files:
                files = [os.path.join(dpath, f) for f in os.listdir(dpath) if (f.endswith(".array_record") if is_arrayrecord else f.endswith(".parquet"))]
        else:
            files = [dpath]

    shard_discovery_dur_ms = (time.perf_counter() - discovery_start) * 1000.0
    logging.info(f"Discovered {len(files)} dataset shards in {shard_discovery_dur_ms:.2f} ms (source: {'manifest.json' if used_manifest else 'directory globbing'})")
    if not files:
        logging.error(f"No valid shard files found under {dpath}!")
        sys.exit(1)

    import random
    random.seed(42)
    if shuffle_mode in ("two_stage", "global"):
        random.shuffle(files)

    step_latencies_ms = []
    tokenize_latencies_ms = []
    io_latencies_ms = []
    total_samples = 0
    total_bytes = 0

    bench_start = time.perf_counter()
    first_batch_dur_ms = 0.0

    # Queues for 4-stream concurrent interleaved streaming
    sample_queue = queue.Queue(maxsize=buf_size * 2)
    file_queue = queue.Queue()
    for f in files:
        file_queue.put(f)

    stop_event = threading.Event()
    if is_arrayrecord:
        from array_record.python import array_record_module

        if shuffle_mode == "global":
            chunk_records = max(1, getattr(config_args, "chunk_records", 1))
            logging.info(f"--> [ArrayRecord] Initializing True Global Shuffle (Chunk Records: {chunk_records} = {chunk_records * 8} KB/seek, Grain Permutation across all shards, Zero-Buffer)...")
            idx_start = time.perf_counter()
            def get_shard_count(fpath):
                try:
                    r = array_record_module.ArrayRecordReader(fpath)
                    n = r.num_records()
                    r.close()
                    return n
                except Exception as e:
                    logging.warning(f"Error checking {fpath}: {e}")
                    return 0

            with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 16)) as pool:
                shard_counts = list(pool.map(get_shard_count, files))

            cum_chunks = [0]
            for c in shard_counts:
                cum_chunks.append(cum_chunks[-1] + math.ceil(c / chunk_records))
            total_chunks = cum_chunks[-1]
            total_records = sum(shard_counts)
            idx_dur_ms = (time.perf_counter() - idx_start) * 1000.0
            logging.info(f"✅ Indexed {len(files)} ArrayRecord shards ({total_records:,} total records, {total_chunks:,} total chunks of {chunk_records * 8} KB) in {idx_dur_ms:.2f} ms")

            chunks_per_batch = math.ceil(batch_size / chunk_records)
            total_needed_chunks = total_steps * chunks_per_batch
            rng = np.random.default_rng(seed=42)
            if total_chunks >= total_needed_chunks:
                global_chunk_indices = rng.permutation(total_chunks)[:total_needed_chunks]
            else:
                reps = math.ceil(total_needed_chunks / max(1, total_chunks))
                perm = np.concatenate([rng.permutation(total_chunks) for _ in range(reps)])
                global_chunk_indices = perm[:total_needed_chunks]

            logging.info(f"[ArrayRecord] Generated deterministic 1D global permutation for {total_needed_chunks:,} chunks ({chunk_records} records / {chunk_records * 8} KB per Range Read, Zero-Buffer across {len(files)} shards)")

            local_storage = threading.local()

            def get_cached_reader(fpath):
                if not hasattr(local_storage, "readers"):
                    local_storage.readers = {}
                if fpath not in local_storage.readers:
                    local_storage.readers[fpath] = array_record_module.ArrayRecordReader(fpath)
                return local_storage.readers[fpath]

            def read_chunk(global_chunk_idx):
                shard_idx = bisect.bisect_right(cum_chunks, global_chunk_idx) - 1
                local_chunk_idx = int(global_chunk_idx - cum_chunks[shard_idx])
                start_rec = local_chunk_idx * chunk_records
                count = min(chunk_records, shard_counts[shard_idx] - start_rec)
                fpath = files[shard_idx]
                reader = get_cached_reader(fpath)
                rec_indices = list(range(start_rec, start_rec + count))
                raw_list = reader.read(rec_indices)
                samples = []
                for raw in raw_list:
                    tokens = np.frombuffer(raw, dtype=np.int32)
                    if len(tokens) < seq_len:
                        tokens = np.pad(tokens, (0, seq_len - len(tokens)), constant_values=0)
                    elif len(tokens) > seq_len:
                        tokens = tokens[:seq_len]
                    samples.append(tokens)
                return samples

            bench_start = time.perf_counter()

            with ThreadPoolExecutor(max_workers=num_streams) as executor:
                for step in range(total_steps):
                    s_start = time.perf_counter()
                    batch_chunk_ids = global_chunk_indices[step * chunks_per_batch : (step + 1) * chunks_per_batch]
                    chunk_results = list(executor.map(read_chunk, batch_chunk_ids))
                    batch_samples = [sample for chunk in chunk_results for sample in chunk][:batch_size]
                    batch_tensor = np.stack(batch_samples, axis=0)
                    s_dur = (time.perf_counter() - s_start) * 1000.0

                    if step == 0:
                        first_batch_dur_ms = (time.perf_counter() - pipeline_init_start) * 1000.0
                        pure_step_ms = (time.perf_counter() - s_start) * 1000.0
                        logging.info(f"STANDALONE DATALOADER : Time to First Batch TTFB (Wall-Clock): {first_batch_dur_ms / 1000.0:.4f} s ({first_batch_dur_ms:.2f} ms) [First step I/O: {pure_step_ms:.2f} ms]")

                    step_latencies_ms.append(s_dur)
                    total_samples += len(batch_samples)
                    total_bytes += batch_tensor.nbytes

                    if (step + 1) % 100 == 0 or (step + 1) == total_steps:
                        logging.info(f"  [MaxText Step {step + 1}/{total_steps}] Batch latency: {s_dur:.2f} ms ({total_samples} samples, {total_bytes / (1024*1024):.1f} MB, Range Read Chunk={chunk_records*8} KB)")
        else:
            def arrayrecord_worker(worker_id):
                while not stop_event.is_set():
                    try:
                        fpath = file_queue.get_nowait()
                    except queue.Empty:
                        for f in files:
                            file_queue.put(f)
                        try:
                            fpath = file_queue.get_nowait()
                        except queue.Empty:
                            break

                    try:
                        reader = array_record_module.ArrayRecordReader(fpath)
                        num_recs = reader.num_records()
                        indices = list(range(num_recs))

                        for idx in indices:
                            if stop_event.is_set():
                                break
                            raw = reader.read([idx])[0]
                            tokens = np.frombuffer(raw, dtype=np.int32)
                            if len(tokens) < seq_len:
                                tokens = np.pad(tokens, (0, seq_len - len(tokens)), constant_values=0)
                            elif len(tokens) > seq_len:
                                tokens = tokens[:seq_len]

                            while not stop_event.is_set():
                                try:
                                    sample_queue.put(tokens, timeout=0.1)
                                    break
                                except queue.Full:
                                    continue
                        reader.close()
                    except Exception as e:
                        logging.error(f"[Worker {worker_id}] Error reading {fpath}: {e}")

            with ThreadPoolExecutor(max_workers=num_streams) as pool:
                futures = [pool.submit(arrayrecord_worker, i) for i in range(num_streams)]
                shuffle_buffer = []
                target_prime = min(buf_size, total_steps * batch_size)
                logging.info(f"[ArrayRecord] Priming shuffle buffer (target: {target_prime} items)...")
                while len(shuffle_buffer) < target_prime:
                    try:
                        item = sample_queue.get(timeout=10.0)
                        shuffle_buffer.append(item)
                    except queue.Empty:
                        if not any(f.running() for f in futures) and sample_queue.empty():
                            break

                for step in range(total_steps):
                    s_start = time.perf_counter()
                    target_fill = min(buf_size, (total_steps - step) * batch_size)
                    while len(shuffle_buffer) < target_fill:
                        try:
                            item = sample_queue.get(timeout=0.05)
                            shuffle_buffer.append(item)
                        except queue.Empty:
                            if not any(f.running() for f in futures) and sample_queue.empty():
                                break
                            break

                    batch_tokens = []
                    while len(batch_tokens) < batch_size:
                        if not shuffle_buffer:
                            try:
                                item = sample_queue.get(timeout=10.0)
                                shuffle_buffer.append(item)
                            except queue.Empty:
                                if not any(f.running() for f in futures) and sample_queue.empty():
                                    break
                                continue

                        if shuffle_mode == "two_stage":
                            pick_idx = random.randint(0, len(shuffle_buffer) - 1)
                            token_item = shuffle_buffer.pop(pick_idx)
                        else:
                            token_item = shuffle_buffer.pop(0)

                        batch_tokens.append(token_item)

                        if not batch_tokens:
                            break

                    batch_tensor = np.stack(batch_tokens, axis=0)
                    s_dur = (time.perf_counter() - s_start) * 1000.0

                    if step == 0:
                        first_batch_dur_ms = (time.perf_counter() - pipeline_init_start) * 1000.0

                    step_latencies_ms.append(s_dur)
                    total_samples += len(batch_tokens)
                    total_bytes += batch_tensor.nbytes

                    if (step + 1) % 100 == 0:
                        logging.info(f"  [MaxText Step {step + 1}/{total_steps}] Batch latency: {s_dur:.2f} ms [ArrayRecord Zero-CPU Two-Stage Shuffle]")

                stop_event.set()
                while not sample_queue.empty():
                    try:
                        sample_queue.get_nowait()
                    except queue.Empty:
                        break

    else:
        # Parquet with runtime Tokenizer + 4-Stream Two-Stage Shuffle
        import pyarrow.parquet as pq
        try:
            import tiktoken
            tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            tokenizer = None

        def parquet_worker(worker_id):
            while not stop_event.is_set():
                try:
                    fpath = file_queue.get_nowait()
                except queue.Empty:
                    for f in files:
                        file_queue.put(f)
                    try:
                        fpath = file_queue.get_nowait()
                    except queue.Empty:
                        break

                try:
                    parquet_file = pq.ParquetFile(fpath)
                    for rg_idx in range(parquet_file.num_row_groups):
                        if stop_event.is_set():
                            break
                        
                        io_t0 = time.perf_counter()
                        table = parquet_file.read_row_group(rg_idx, columns=["text"] if "text" in parquet_file.schema.names else parquet_file.schema.names[:1])
                        raw_texts = table[table.column_names[0]].to_pylist()
                        io_d = (time.perf_counter() - io_t0) * 1000.0 / max(1, len(raw_texts))

                        tok_t0 = time.perf_counter()
                        for text_val in raw_texts:
                            if stop_event.is_set():
                                break
                            if isinstance(text_val, str):
                                if tokenizer:
                                    toks = np.array(tokenizer.encode(text_val[:seq_len * 4], disallowed_special=(), allowed_special="all")[:seq_len], dtype=np.int32)
                                else:
                                    toks = np.frombuffer(text_val[:seq_len * 4].encode("utf-8"), dtype=np.uint8).astype(np.int32)
                            elif isinstance(text_val, (bytes, bytearray)):
                                toks = np.frombuffer(text_val, dtype=np.int32)
                            elif isinstance(text_val, (list, np.ndarray)):
                                toks = np.array(text_val[:seq_len], dtype=np.int32)
                            else:
                                toks = np.random.randint(0, 32000, size=seq_len, dtype=np.int32)

                            if len(toks) < seq_len:
                                toks = np.pad(toks, (0, seq_len - len(toks)), constant_values=0)
                            elif len(toks) > seq_len:
                                toks = toks[:seq_len]

                            tok_d = (time.perf_counter() - tok_t0) * 1000.0 / max(1, len(raw_texts))

                            while not stop_event.is_set():
                                try:
                                    sample_queue.put((toks, io_d, tok_d), timeout=0.1)
                                    break
                                except queue.Full:
                                    continue
                except Exception as e:
                    logging.error(f"[Worker {worker_id}] Error reading {fpath}: {e}")

        # Start concurrent workers
        with ThreadPoolExecutor(max_workers=num_streams) as pool:
            futures = [pool.submit(parquet_worker, i) for i in range(num_streams)]
            
            shuffle_buffer = []

            # Prime buffer
            logging.info(f"[Parquet] Priming shuffle buffer (target: {min(buf_size, total_steps * batch_size)} items)...")
            target_prime = min(buf_size, total_steps * batch_size)
            while len(shuffle_buffer) < target_prime:
                try:
                    item = sample_queue.get(timeout=10.0)
                    shuffle_buffer.append(item)
                except queue.Empty:
                    if not any(f.running() for f in futures) and sample_queue.empty():
                        break

            for step in range(total_steps):
                s_start = time.perf_counter()
                
                target_fill = min(buf_size, (total_steps - step) * batch_size)
                while len(shuffle_buffer) < target_fill:
                    try:
                        item = sample_queue.get(timeout=0.05)
                        shuffle_buffer.append(item)
                    except queue.Empty:
                        if not any(f.running() for f in futures) and sample_queue.empty():
                            break
                        break

                batch_tokens = []
                step_io_durs = []
                step_tok_durs = []

                while len(batch_tokens) < batch_size:
                    if not shuffle_buffer:
                        try:
                            item = sample_queue.get(timeout=10.0)
                            shuffle_buffer.append(item)
                        except queue.Empty:
                            if not any(f.running() for f in futures) and sample_queue.empty():
                                break
                            continue

                    if shuffle_mode in ("two_stage", "global"):
                        pick_idx = random.randint(0, len(shuffle_buffer) - 1)
                        token_item, io_d, tok_d = shuffle_buffer.pop(pick_idx)
                    else:
                        token_item, io_d, tok_d = shuffle_buffer.pop(0)

                    batch_tokens.append(token_item)
                    step_io_durs.append(io_d)
                    step_tok_durs.append(tok_d)

                if not batch_tokens:
                    break

                batch_tensor = np.stack(batch_tokens, axis=0)
                s_dur = (time.perf_counter() - s_start) * 1000.0

                if step == 0:
                    first_batch_dur_ms = (time.perf_counter() - pipeline_init_start) * 1000.0

                step_latencies_ms.append(s_dur)
                total_samples += len(batch_tokens)
                total_bytes += batch_tensor.nbytes
                if step_io_durs:
                    io_latencies_ms.append(sum(step_io_durs))
                if step_tok_durs:
                    tokenize_latencies_ms.append(sum(step_tok_durs))

                if (step + 1) % 100 == 0:
                    logging.info(f"  [MaxText Step {step + 1}/{total_steps}] Batch latency: {s_dur:.2f} ms [Parquet CPU Tokenizer Two-Stage Shuffle]")

            stop_event.set()
            while not sample_queue.empty():
                try:
                    sample_queue.get_nowait()
                except queue.Empty:
                    break

    total_duration_sec = time.perf_counter() - bench_start
    avg_batch_ms = np.mean(step_latencies_ms)
    p50_batch_ms = np.percentile(step_latencies_ms, 50)
    p95_batch_ms = np.percentile(step_latencies_ms, 95)
    p99_batch_ms = np.percentile(step_latencies_ms, 99)
    samples_per_sec = total_samples / total_duration_sec if total_duration_sec > 0 else 0.0

    logging.info("==================================================================================")
    logging.info("                 MAXTEXT IN-TREE STANDALONE DATALOADER SUMMARY                    ")
    logging.info("==================================================================================")
    logging.info(f"Dataset Pipeline Format  : {fmt_name}")
    logging.info(f"Dataset Path             : {dpath}")
    logging.info(f"Shuffle Strategy         : {shuffle_mode} (Interleaved {num_streams} Streams + Window Buffer {buf_size})")
    logging.info(f"Total Batches Ingested   : {len(step_latencies_ms)} batches")
    logging.info(f"Batch Size (Per Step)    : {batch_size} samples")
    logging.info(f"Total Samples Ingested   : {total_samples} samples")
    logging.info(f"Total Tensor Payload     : {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024 * 1024 * 1024):.4f} GB)")
    logging.info(f"Time to First Batch TTFB : {first_batch_dur_ms:.2f} ms ({first_batch_dur_ms / 1000.0:.4f} s) [Wall-Clock from Pipeline Init]")
    logging.info(f"  ├── Shard Discovery Lat: {shard_discovery_dur_ms:.2f} ms ({'manifest.json' if used_manifest else 'directory globbing'})")
    if is_arrayrecord and shuffle_mode == "global":
        logging.info(f"  ├── Footer Scan Latency: {idx_dur_ms:.2f} ms")
        logging.info(f"  └── Range Read Chunk   : {chunk_records} records ({chunk_records * 8} KB/Range Read, {chunks_per_batch} chunks/batch)")
    logging.info(f"End-to-End Load Lat (Avg): {avg_batch_ms:.2f} ms")
    logging.info(f"End-to-End Load Lat (p50): {p50_batch_ms:.2f} ms")
    logging.info(f"End-to-End Load Lat (p95): {p95_batch_ms:.2f} ms")
    logging.info(f"End-to-End Load Lat (p99): {p99_batch_ms:.2f} ms")
    if not is_arrayrecord and tokenize_latencies_ms:
        logging.info(f"  └── CPU Tokenizer Avg  : {np.mean(tokenize_latencies_ms):.2f} ms ({min(100.0, np.mean(tokenize_latencies_ms) / max(0.01, avg_batch_ms) * 100):.1f}% of Step Latency)")
        logging.info(f"  └── Storage I/O Avg    : {np.mean(io_latencies_ms):.2f} ms")
    logging.info(f"Effective Ingestion Rate : {samples_per_sec:.2f} samples/sec")
    logging.info(f"Tensor Pipeline Speed    : {(total_bytes / (1024 * 1024)) / total_duration_sec:.2f} MB/s")
    logging.info("==================================================================================")


def main():
    args, extra = parse_args()
    run_standalone_dataloader(args, extra)


if __name__ == "__main__":
    main()
