#!/usr/bin/env python3
"""
Unit tests for skills/benchmark-metrics-parser/scripts/parse_metrics.py
Validates log parsing across PyTorch DDP, MaxText, and Multi-Format DataLoader logs,
multi-run statistical calculations, and Markdown/JSON table formatters.
"""

import importlib.util
import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PARSE_METRICS_PATH = os.path.join(
    PROJECT_ROOT, "skills/benchmark-metrics-parser/scripts/parse_metrics.py"
)

# Dynamically import parse_metrics from its path
spec = importlib.util.spec_from_file_location("parse_metrics", PARSE_METRICS_PATH)
parse_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse_metrics)

parse_single_log = parse_metrics.parse_single_log
format_markdown_table = parse_metrics.format_markdown_table

FIXTURES_LOG_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/logs")


class TestParseMetrics(unittest.TestCase):
    """Tests for parse_metrics.py extracting metrics from log formats."""

    def test_parse_pytorch_ddp_log(self):
        log_path = os.path.join(FIXTURES_LOG_DIR, "pytorch_llama_ddp.log")
        self.assertTrue(os.path.exists(log_path), f"Fixture not found: {log_path}")

        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()

        metrics = parse_single_log(log_content, log_name="GCSFuse-PyTorch")

        # 1. Hugging Face Dataloader Prep Time
        self.assertIsNotNone(metrics["dataset_load_time_sec"])
        self.assertAlmostEqual(metrics["dataset_load_time_sec"], 0.8983, places=4)

        # 2. Worker Spawn & Data Loading Time
        self.assertIsNotNone(metrics["worker_spawn_time_sec"])
        self.assertAlmostEqual(metrics["worker_spawn_time_sec"], 20.71, places=2)

        # 3. Training Step Metrics
        self.assertIsNotNone(metrics["step_time_avg_sec"])
        self.assertGreater(metrics["step_throughput_avg_samples_sec"], 300.0)

        # 4. Checkpoints
        self.assertEqual(metrics["checkpoint_count"], 2)
        self.assertIsNotNone(metrics["checkpoint_duration_avg_sec"])
        self.assertAlmostEqual(metrics["checkpoint_duration_avg_sec"], (108.34 + 105.10) / 2, places=2)
        self.assertAlmostEqual(metrics["raw_write_speed_avg_mbs"], (447.31 + 452.10) / 2, places=2)
        self.assertAlmostEqual(metrics["aggregated_throughput_avg_mbs"], (426.43 + 430.20) / 2, places=2)
        self.assertAlmostEqual(metrics["checkpoint_size_gb"], 44.87, places=2)

        # 5. Checkpoint Delete Time
        self.assertAlmostEqual(metrics["checkpoint_delete_time_avg_sec"], 2.37, places=2)

    def test_parse_maxtext_intree_log(self):
        log_path = os.path.join(FIXTURES_LOG_DIR, "maxtext_intree.log")
        self.assertTrue(os.path.exists(log_path), f"Fixture not found: {log_path}")

        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()

        metrics = parse_single_log(log_content, log_name="MaxText-ArrayRecord")

        # Ingestion metrics
        self.assertIsNotNone(metrics["ttfb_ms"])
        self.assertAlmostEqual(metrics["ttfb_ms"], 2810.0, places=1)
        self.assertAlmostEqual(metrics["dataset_read_throughput_mbs"], 38.19, places=2)
        self.assertAlmostEqual(metrics["samples_per_sec"], 4888.0, places=1)
        self.assertAlmostEqual(metrics["p50_latency_ms"], 19.57, places=2)
        self.assertAlmostEqual(metrics["p95_latency_ms"], 33.83, places=2)

    def test_parse_multiformat_parquet_log(self):
        log_path = os.path.join(FIXTURES_LOG_DIR, "multiformat_parquet.log")
        self.assertTrue(os.path.exists(log_path), f"Fixture not found: {log_path}")

        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()

        metrics = parse_single_log(log_content, log_name="MultiFormat-Parquet")

        self.assertAlmostEqual(metrics["ttfb_ms"], 7280.0, places=1)
        self.assertAlmostEqual(metrics["dataset_read_throughput_mbs"], 25.68, places=2)
        self.assertAlmostEqual(metrics["samples_per_sec"], 3287.0, places=1)
        self.assertAlmostEqual(metrics["p50_latency_ms"], 23.76, places=2)
        self.assertAlmostEqual(metrics["p95_latency_ms"], 39.97, places=2)

    def test_format_markdown_and_json(self):
        log_path_1 = os.path.join(FIXTURES_LOG_DIR, "pytorch_llama_ddp.log")
        log_path_2 = os.path.join(FIXTURES_LOG_DIR, "maxtext_intree.log")

        with open(log_path_1, "r", encoding="utf-8") as f:
            m1 = parse_single_log(f.read(), log_name="Run1")
        with open(log_path_2, "r", encoding="utf-8") as f:
            m2 = parse_single_log(f.read(), log_name="Run2")

        md_output = format_markdown_table([m1, m2])
        self.assertIn("Run1", md_output)
        self.assertIn("Run2", md_output)
        self.assertIn("| **DataLoader Prep Time** |", md_output)

        json_raw = json.dumps({"runs": [m1, m2]})
        parsed_json = json.loads(json_raw)
        self.assertEqual(len(parsed_json["runs"]), 2)
        self.assertEqual(parsed_json["runs"][0]["log_name"], "Run1")


if __name__ == "__main__":
    unittest.main()
