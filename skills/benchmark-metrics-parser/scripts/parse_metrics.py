#!/usr/bin/env python3
import sys
import re
import json
import statistics

def parse_single_log(log_text):
    metrics = {
        "dataset_load_time_sec": None,
        "raw_write_speed_mbs": None,
        "raw_write_speed_gbps": None,
        "checkpoint_duration_sec": None,
        "aggregated_throughput_mbs": None,
        "checkpoint_size_bytes": None,
        "checkpoint_size_gb": None,
        "ttfb_ms": None,
        "dataset_read_throughput_mbs": None,
        "samples_per_sec": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
    }

    # Extract dataset load latency
    dataloader_match = re.search(r"HF dataloader prepared in ([\d\.]+)s", log_text)
    if dataloader_match:
        metrics["dataset_load_time_sec"] = float(dataloader_match.group(1))

    # Extract dataset benchmark summary metrics
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

    # Extract aggregated checkpoint save metrics
    agg_match = re.search(
        r"Aggregated Checkpoint Save Complete : Step : \d+ : Total Size : (\d+) bytes \(([\d\.]+) MB / ([\d\.]+) GB\) : Total Duration : ([\d\.]+) seconds : Aggregated Throughput : ([\d\.]+) MB/s / ([\d\.]+) GB/s",
        log_text
    )
    if agg_match:
        metrics["checkpoint_size_bytes"] = int(agg_match.group(1))
        metrics["checkpoint_size_gb"] = float(agg_match.group(3))
        metrics["checkpoint_duration_sec"] = float(agg_match.group(4))
        metrics["aggregated_throughput_mbs"] = float(agg_match.group(5))

    # Extract raw write speed
    raw_speed_match = re.search(
        r"Finished saving checkpoint .* in ([\d\.]+) seconds \(Upload Time: ([\d\.]+) seconds\)",
        log_text
    )
    if raw_speed_match and metrics.get("checkpoint_size_bytes"):
        upload_time = float(raw_speed_match.group(2))
        if upload_time > 0:
            mbs = (metrics["checkpoint_size_bytes"] / (1024 * 1024)) / upload_time
            metrics["raw_write_speed_mbs"] = round(mbs, 2)
            metrics["raw_write_speed_gbps"] = round((mbs * 8) / 1024, 2)

    return metrics

def aggregate_multiple_runs(runs_data):
    """
    Given a list of metric dicts for multiple runs, calculate mean, min, max.
    """
    summary = {}
    keys = [
        "dataset_load_time_sec",
        "ttfb_ms",
        "dataset_read_throughput_mbs",
        "samples_per_sec",
        "p50_latency_ms",
        "p95_latency_ms",
        "raw_write_speed_mbs",
        "raw_write_speed_gbps",
        "checkpoint_duration_sec",
        "aggregated_throughput_mbs",
    ]
    
    for key in keys:
        values = [r[key] for r in runs_data if r.get(key) is not None]
        if values:
            summary[key] = {
                "mean": round(statistics.mean(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
                "samples": len(values)
            }
        else:
            summary[key] = None
            
    return summary

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = sys.stdin.read()

    # Support JSON array of multiple logs input or single log string
    try:
        data = json.loads(content)
        if isinstance(data, list):
            parsed_runs = [parse_single_log(item) for item in data]
            aggregated = aggregate_multiple_runs(parsed_runs)
            print(json.dumps({"individual_runs": parsed_runs, "aggregated_summary": aggregated}, indent=2))
            return
    except Exception:
        pass

    result = parse_single_log(content)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
