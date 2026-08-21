#!/usr/bin/env python3
"""
Unit tests for tools/datasets/converters/parquet_to_arrayrecord.py
Validates CLI parsing and local Parquet -> ArrayRecord conversion and manifest generation.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.datasets.converters.parquet_to_arrayrecord import (
    array_record_module,
    convert_parquet_to_arrayrecord,
    parse_args,
)
from tools.datasets.generator import generate_parquet_dataset


class TestParquetToArrayRecord(unittest.TestCase):
    """Tests for Parquet to ArrayRecord converter."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_parquet_conv_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_cli_args_parsing(self):
        test_args = [
            "--input-path", "/tmp/parquet",
            "--output-path", "/tmp/arrayrecord",
            "--sequence-length", "1024",
            "--num-workers", "4",
        ]
        with patch("sys.argv", ["parquet_to_arrayrecord.py"] + test_args):
            args = parse_args()
            self.assertEqual(args.input_path, "/tmp/parquet")
            self.assertEqual(args.output_path, "/tmp/arrayrecord")
            self.assertEqual(args.sequence_length, 1024)
            self.assertEqual(args.num_workers, 4)

    @unittest.skipIf(array_record_module is None, "array_record python package is not installed")
    def test_convert_local_parquet_to_arrayrecord(self):
        parquet_in = os.path.join(self.test_dir, "parquet_in")
        arrayrecord_out = os.path.join(self.test_dir, "arrayrecord_out")

        # 1. Generate dummy parquet shards
        generate_parquet_dataset(
            output_path=parquet_in,
            total_size_mb=0.5,
            num_files=2,
            sequence_length=64,
            metadata_bytes_per_row=16,
        )

        # 2. Convert to ArrayRecord
        convert_parquet_to_arrayrecord(
            input_path=parquet_in,
            output_path=arrayrecord_out,
            text_column="text",
            sequence_length=64,
            max_files=2,
            num_workers=2,
        )

        # 3. Check output files & manifest
        self.assertTrue(os.path.isdir(arrayrecord_out))
        manifest_file = os.path.join(arrayrecord_out, "manifest.json")
        self.assertTrue(os.path.exists(manifest_file), "manifest.json should be generated")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["dataset_format"], "arrayrecord")
        self.assertEqual(manifest["num_shards"], 2)
        self.assertGreater(manifest["total_records"], 0)
        self.assertEqual(len(manifest["shards"]), 2)


if __name__ == "__main__":
    unittest.main()
