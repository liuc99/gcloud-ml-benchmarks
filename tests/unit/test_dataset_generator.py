#!/usr/bin/env python3
"""
Unit tests for tools/datasets/generator.py
Validates local synthetic Parquet dataset generation, column schema,
file count, and size calculation.
"""

import os
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

from tools.datasets.generator import generate_parquet_dataset, parse_args


class TestDatasetGenerator(unittest.TestCase):
    """Tests for the synthetic dataset generator."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_dataset_gen_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @unittest.skipIf(pa is None or pq is None, "pyarrow is required for dataset generation")
    def test_generate_parquet_dataset_locally(self):
        out_dir = os.path.join(self.test_dir, "parquet_out")
        
        generate_parquet_dataset(
            output_path=out_dir,
            total_size_mb=0.5,  # Small size for fast test execution
            num_files=3,
            sequence_length=64,
            metadata_bytes_per_row=16,
        )

        self.assertTrue(os.path.isdir(out_dir))
        files = sorted([f for f in os.listdir(out_dir) if f.endswith(".parquet")])
        self.assertEqual(len(files), 3)

        # Inspect first shard schema
        first_file = os.path.join(out_dir, files[0])
        table = pq.read_table(first_file)

        self.assertIn("input_ids", table.column_names)
        self.assertIn("label", table.column_names)
        self.assertIn("attention_mask", table.column_names)
        self.assertIn("sample_id", table.column_names)
        self.assertIn("extra_metadata_bytes", table.column_names)

        # Verify row count is positive
        self.assertGreater(table.num_rows, 0)
        self.assertEqual(len(table["input_ids"][0].as_py()), 64)

    def test_cli_argument_parsing(self):
        test_args = [
            "--output-path", "/tmp/dummy",
            "--total-size-mb", "256",
            "--num-files", "4",
            "--sequence-length", "1024",
            "--metadata-bytes-per-row", "128",
        ]
        import unittest.mock
        with unittest.mock.patch("sys.argv", ["generator.py"] + test_args):
            args = parse_args()
            self.assertEqual(args.output_path, "/tmp/dummy")
            self.assertEqual(args.total_size_mb, 256.0)
            self.assertEqual(args.num_files, 4)
            self.assertEqual(args.sequence_length, 1024)
            self.assertEqual(args.metadata_bytes_per_row, 128)


if __name__ == "__main__":
    unittest.main()
