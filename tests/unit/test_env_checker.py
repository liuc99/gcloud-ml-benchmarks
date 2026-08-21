#!/usr/bin/env python3
"""
Unit tests for tools/infrastructure/env_checker.py
Validates environment pre-flight diagnostics, CLI checks,
Python package checks, table/JSON formatting, and error reporting.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.infrastructure.env_checker import (
    CLI_TOOLS,
    PYTHON_PACKAGES,
    check_auth_and_connectivity,
    check_cli_tools,
    check_python_packages,
    run_env_check,
)


class TestEnvChecker(unittest.TestCase):
    """Tests for environment diagnostic checker."""

    def test_check_cli_tools(self):
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/" + x if x == "python3" else None), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Python 3.11.0\n", stderr="")
            results, ok = check_cli_tools()
            self.assertIn("python3", results)
            self.assertTrue(results["python3"]["present"])
            # gcloud was mocked as missing
            self.assertIn("gcloud", results)
            self.assertFalse(results["gcloud"]["present"])
            self.assertFalse(ok)  # Because required gcloud was missing

    def test_check_python_packages(self):
        # pyyaml is known to be installed in this env
        results, missing, ok = check_python_packages()
        self.assertIn("pyyaml", results)
        self.assertTrue(results["pyyaml"]["present"])
        self.assertIsInstance(missing, list)

    def test_check_auth_and_connectivity(self):
        with patch("tools.infrastructure.env_checker.run_cmd") as mock_cmd:
            mock_cmd.side_effect = lambda cmd, **kwargs: (
                "user@example.com" if "account" in cmd
                else "my-test-project" if "project" in cmd
                else "gke_test_context" if "current-context" in cmd
                else "Kubernetes control plane is running" if "cluster-info" in cmd
                else None
            )
            auth = check_auth_and_connectivity()
            self.assertEqual(auth["status"], "PASS")
            self.assertEqual(auth["gcloud_account"], "user@example.com")
            self.assertEqual(auth["gcloud_project"], "my-test-project")
            self.assertTrue(auth["k8s_cluster_reachable"])

    def test_run_env_check_structure(self):
        summary, passed = run_env_check()
        self.assertIn("overall_status", summary)
        self.assertIn("cli_tools", summary)
        self.assertIn("python_packages", summary)
        self.assertIn("auth_and_connectivity", summary)
        self.assertIsInstance(summary["cli_tools"], dict)


if __name__ == "__main__":
    unittest.main()
