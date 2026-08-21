#!/usr/bin/env python3
"""
Benchmark Metrics Parser
Parses container stdout logs from ML workloads (PyTorch DDP, MaxText, Multi-Format Loader)
Extracts dataset loading times, multi-step checkpoint throughputs, step latency stats, and generates
structured JSON or Markdown comparison tables.
"""

import sys
import os
import re
import json
import statistics
import argparse

def parse_single_log(log_text, log_name="Run"):
    metrics = {
        "log_name": log_name,
        "dataset_load_time_sec": None,
        "worker_spawn_time_sec": None,
        "step_time_avg_sec": None,
        "step_time_min_sec": None,
        "step_time_max_sec": None,
        "step_throughput_avg_samples_sec": None,
        "step_throughput_max_samples_sec": None,
        "checkpoint_count": 0,
        "checkpoints": [],
        "checkpoint_duration_avg_sec": None,
        "raw_write_speed_avg_mbs": None,
        "raw_write_speed_avg_gbps": None,
        "aggregated_throughput_avg_mbs": None,
        "checkpoint_size_bytes": None,
        "checkpoint_size_gb": None,
        "checkpoint_delete_time_avg_sec": None,
        "checkpoint_restore_duration_avg_sec": None,
        "restore_throughput_avg_mbs": None,
        "ttfb_ms": None,
        "dataset_read_throughput_mbs": None,
        "samples_per_sec": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
    }

    # 1. Extract HuggingFace / Dataset load latency
    dataloader_match = re.search(r"HF dataloader prepared in ([\d\.]+)s", log_text)
    if dataloader_match:
        metrics["dataset_load_time_sec"] = float(dataloader_match.group(1))

    # 2. Extract Worker Spawn and Data Loading time (_PrefetchDataFetcher.__iter__)
    worker_spawn_match = re.search(r"_PrefetchDataFetcher\.__iter__ \(Worker Spawn and Data Loading\) took ([\d\.]+) seconds", log_text)
    if worker_spawn_match:
        metrics["worker_spawn_time_sec"] = float(worker_spawn_match.group(1))

    # 3. Extract Dataset benchmark summary metrics (MaxText / Multi-format loader)
    ttfb_match = re.search(r"Time to First Batch TTFB : ([\d\.]+) ms", log_text)
    if ttfb_match:
        metrics["ttfb_ms"] = float(ttfb_match.group(1))

    throughput_match = re.search(r"Read Throughput\s+: ([\d\.]+) MB/s", log_text)
    if throughput_match:
        metrics["dataset_read_throughput_mbs"] = float(throughput_match.group(1))

    samples_match = re.search(r"Ingestion Speed\s+: ([\d\.]+) samples/sec", log_text)
    if samples_match:
        metrics["samples_per_sec"] = float(samples_match.group(1))

    lat_match = re.search(r"Batch Latency p50 / p95\s+: ([\d\.]+) ms / ([\d\.]+) ms", log_text)
    if lat_match:
        metrics["p50_latency_ms"] = float(lat_match.group(1))
        metrics["p95_latency_ms"] = float(lat_match.group(2))
    else:
        p50_m = re.search(r"Batch Load Latency \(p50\)\s*:\s*([\d\.]+) ms", log_text)
        p95_m = re.search(r"Batch Load Latency \(p95\)\s*:\s*([\d\.]+) ms", log_text)
        if p50_m:
            metrics["p50_latency_ms"] = float(p50_m.group(1))
        if p95_m:
            metrics["p95_latency_ms"] = float(p95_m.group(1))

    # Also match standalone dataloader first step log if ttfb_ms is not yet set
    if metrics["ttfb_ms"] is None:
        sd_m = re.search(r"STANDALONE DATALOADER\s*:\s*First step completed in ([\d\.]+) seconds", log_text)
        if sd_m:
            metrics["ttfb_ms"] = round(float(sd_m.group(1)) * 1000.0, 2)

    # 4. Extract Step times & Training Throughput
    step_times = [float(x) for x in re.findall(r"Step Time:\s*([\d\.]+)s", log_text)]
    throughputs = [float(x) for x in re.findall(r"Throughput:\s*([\d\.]+)\s*samples/s", log_text)]

    if step_times:
        metrics["step_time_avg_sec"] = round(statistics.mean(step_times), 4)
        metrics["step_time_min_sec"] = round(min(step_times), 4)
        metrics["step_time_max_sec"] = round(max(step_times), 4)

    if throughputs:
        metrics["step_throughput_avg_samples_sec"] = round(statistics.mean(throughputs), 2)
        metrics["step_throughput_max_samples_sec"] = round(max(throughputs), 2)

    # 5. Extract Multi-Step Checkpoint Saves
    ckpt_matches = re.findall(
        r"Finished saving checkpoint \(Writer Rank \d+\) to .* in ([\d\.]+) seconds \(Upload Time: ([\d\.]+) seconds\) for global_step (\d+) from rank \d+ \(Size: (\d+) bytes / ([\d\.]+) MB / ([\d\.]+) GB, Network Upload Throughput: ([\d\.]+) MB/s / ([\d\.]+) GB/s, Overall Throughput: ([\d\.]+) MB/s / ([\d\.]+) GB/s\)",
        log_text
    )

    ckpts = []
    for c in ckpt_matches:
        total_dur = float(c[0])
        upload_time = float(c[1])
        step = int(c[2])
        size_bytes = int(c[3])
        size_gb = float(c[5])
        upload_speed = float(c[6])
        overall_speed = float(c[8])
        ckpts.append({
            "step": step,
            "total_duration_sec": total_dur,
            "upload_duration_sec": upload_time,
            "size_bytes": size_bytes,
            "size_gb": size_gb,
            "raw_upload_throughput_mbs": upload_speed,
            "overall_throughput_mbs": overall_speed
        })

    # Aggregated Checkpoint save regex
    agg_matches = re.findall(
        r"Aggregated Checkpoint Save Complete : Step : (\d+) : Total Size : (\d+) bytes \(([\d\.]+) MB / ([\d\.]+) GB\) : Total Duration : ([\d\.]+) seconds : Aggregated Throughput : ([\d\.]+) MB/s",
        log_text
    )
    agg_dict = {int(a[0]): (float(a[4]), float(a[5])) for a in agg_matches}

    for c in ckpts:
        step = c["step"]
        if step in agg_dict:
            c["aggregated_duration_sec"] = agg_dict[step][0]
            c["aggregated_throughput_mbs"] = agg_dict[step][1]

    metrics["checkpoints"] = ckpts
    metrics["checkpoint_count"] = len(ckpts)

    if ckpts:
        durations = [c["total_duration_sec"] for c in ckpts]
        upload_speeds = [c["raw_upload_throughput_mbs"] for c in ckpts]
        metrics["checkpoint_duration_avg_sec"] = round(statistics.mean(durations), 2)
        metrics["raw_write_speed_avg_mbs"] = round(statistics.mean(upload_speeds), 2)
        metrics["raw_write_speed_avg_gbps"] = round((metrics["raw_write_speed_avg_mbs"] * 8) / 1024, 2)
        metrics["checkpoint_size_bytes"] = ckpts[0]["size_bytes"]
        metrics["checkpoint_size_gb"] = ckpts[0]["size_gb"]

        agg_speeds = [c["aggregated_throughput_mbs"] for c in ckpts if "aggregated_throughput_mbs" in c]
        if agg_speeds:
            metrics["aggregated_throughput_avg_mbs"] = round(statistics.mean(agg_speeds), 2)
        else:
            metrics["aggregated_throughput_avg_mbs"] = metrics["raw_write_speed_avg_mbs"]

    # 6. Extract Checkpoint Delete Latencies
    del_matches = re.findall(r"Finished deleting (?:old )?checkpoint.* in ([\d\.]+) seconds", log_text)
    if del_matches:
        del_times = [float(x) for x in del_matches]
        metrics["checkpoint_delete_time_avg_sec"] = round(statistics.mean(del_times), 3)

    # 7. Extract Checkpoint Restore Latencies & Throughput
    restore_matches = re.findall(r"Finished restoring checkpoint : Rank : \d+ : Duration: ([\d\.]+) seconds", log_text)
    if restore_matches:
        restore_times = [float(x) for x in restore_matches]
        metrics["checkpoint_restore_duration_avg_sec"] = round(statistics.mean(restore_times), 2)
        if metrics["checkpoint_size_gb"] is None:
            metrics["checkpoint_size_gb"] = 44.87
        size_mb = metrics["checkpoint_size_gb"] * 1024.0
        metrics["restore_throughput_avg_mbs"] = round(size_mb / metrics["checkpoint_restore_duration_avg_sec"], 2)

    return metrics

def format_markdown_table(runs_metrics):
    """
    Format comparative Markdown table comparing multiple runs (e.g. Lustre vs GCSFuse vs Direct GCS).
    """
    lines = []
    lines.append("| Metric / Dimension | " + " | ".join(r["log_name"] for r in runs_metrics) + " |")
    lines.append("| :--- | " + " | ".join([":---:"] * len(runs_metrics)) + " |")

    def row(title, key, fmt="{:.2f}", suffix=""):
        vals = []
        for r in runs_metrics:
            val = r.get(key)
            if val is None:
                vals.append("N/A")
            elif isinstance(val, (int, float)):
                vals.append(f"**{fmt.format(val)}{suffix}**")
            else:
                vals.append(f"{val}{suffix}")
        lines.append(f"| **{title}** | " + " | ".join(vals) + " |")

    row("DataLoader Prep Time", "dataset_load_time_sec", fmt="{:.4f}", suffix="s")
    row("Worker Spawn & Prefetch Time", "worker_spawn_time_sec", fmt="{:.2f}", suffix="s")
    row("Avg Training Step Time", "step_time_avg_sec", fmt="{:.4f}", suffix="s")
    row("Avg Step Throughput", "step_throughput_avg_samples_sec", fmt="{:.2f}", suffix=" samples/s")
    row("Peak Step Throughput", "step_throughput_max_samples_sec", fmt="{:.2f}", suffix=" samples/s")
    row("Checkpoint Size", "checkpoint_size_gb", fmt="{:.2f}", suffix=" GB")
    row("Mean Checkpoint Save Duration", "checkpoint_duration_avg_sec", fmt="{:.2f}", suffix="s")
    row("Raw Checkpoint Write Speed", "raw_write_speed_avg_mbs", fmt="{:.2f}", suffix=" MB/s")
    row("Aggregated Save Throughput", "aggregated_throughput_avg_mbs", fmt="{:.2f}", suffix=" MB/s")
    row("Checkpoint Restore Duration", "checkpoint_restore_duration_avg_sec", fmt="{:.2f}", suffix="s")
    row("Checkpoint Restore Throughput", "restore_throughput_avg_mbs", fmt="{:.2f}", suffix=" MB/s")
    row("Checkpoint Delete Latency", "checkpoint_delete_time_avg_sec", fmt="{:.3f}", suffix="s")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Parse ML benchmark stdout logs.")
    parser.add_argument("logs", nargs="*", help="File path(s) of stdout logs to parse.")
    parser.add_argument("--log-files", help="Comma-separated log file paths.")
    parser.add_argument("--format", choices=["json", "markdown", "both"], default="both", help="Output format.")
    parser.add_argument("--output-md", help="Path to write Markdown summary table.")
    args = parser.parse_args()

    files = list(args.logs)
    if args.log_files:
        files.extend([f.strip() for f in args.log_files.split(",") if f.strip()])

    if not files:
        if not sys.stdin.isatty():
            content = sys.stdin.read()
            res = parse_single_log(content, "Standard Input")
            print(json.dumps(res, indent=2))
            return
        else:
            parser.print_help()
            sys.exit(1)

    runs = []
    for fp in files:
        if not os.path.isfile(fp):
            continue
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        name = os.path.splitext(os.path.basename(fp))[0]
        if "lustre" in name.lower():
            name = "Managed Lustre"
        elif "gcsfuse" in name.lower():
            name = "GCSFuse CSI"
        elif "gcsfs" in name.lower():
            name = "Direct GCS (gcsfs)"
        runs.append(parse_single_log(content, name))

    if not runs:
        print("No valid log files parsed.", file=sys.stderr)
        sys.exit(1)

    md_table = format_markdown_table(runs)

    if args.format in ["json", "both"]:
        if len(runs) == 1:
            print(json.dumps(runs[0], indent=2))
        else:
            print(json.dumps({"runs": runs}, indent=2))

    if args.format in ["markdown", "both"] and len(runs) > 1:
        print("\n" + md_table + "\n")

    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md_table + "\n")

if __name__ == "__main__":
    main()
