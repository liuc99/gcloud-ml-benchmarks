#!/usr/bin/env python3
"""
Agent Skills Contract, Guardrails & Integrity Test Suite.

Ensures that:
1. All SKILL.md files have valid YAML frontmatter with correct names & descriptions.
2. All referenced relative markdown files exist on disk (zero broken links).
3. All critical Guardrails & Rules from AGENTS.md remain intact in ml-benchmark-orchestrator.
4. Primary Plan Review table template schema contains all required dimensions.
5. All CLI tool commands in skills match actual tool ArgumentParser schemas.
6. All workload Helm charts in workloads/ compile and render valid manifests with `helm template`.
"""

import glob
import os
import re
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.infrastructure.bucket_manager import parse_args as parse_bucket_args
from tools.infrastructure.cluster_manager import parse_args as parse_cluster_args
from tools.infrastructure.env_checker import parse_args as parse_env_args
from tools.datasets.generator import parse_args as parse_gen_args
from tools.datasets.converters.parquet_to_arrayrecord import parse_args as parse_conv_args


class TestSkillsContract(unittest.TestCase):
    """Automated integrity and regression tests for Agent Skills."""

    def test_skills_yaml_frontmatter(self):
        """Verifies every SKILL.md has valid YAML frontmatter with matching name."""
        skill_files = glob.glob(os.path.join(PROJECT_ROOT, "skills/*/SKILL.md"))
        self.assertTrue(len(skill_files) >= 4, f"Found only {len(skill_files)} skill files")

        for skill_file in skill_files:
            dir_name = os.path.basename(os.path.dirname(skill_file))
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertTrue(content.startswith("---"), f"{skill_file} must start with YAML frontmatter delimiter '---'")
            parts = content.split("---", 2)
            self.assertGreaterEqual(len(parts), 3, f"{skill_file} must have a closing '---' for frontmatter")

            frontmatter = yaml.safe_load(parts[1])
            self.assertIsInstance(frontmatter, dict, f"Invalid YAML frontmatter in {skill_file}")
            self.assertIn("name", frontmatter, f"Missing 'name' in frontmatter: {skill_file}")
            self.assertIn("description", frontmatter, f"Missing 'description' in frontmatter: {skill_file}")
            self.assertEqual(frontmatter["name"], dir_name, f"Skill name '{frontmatter['name']}' does not match directory '{dir_name}'")
            self.assertGreater(len(frontmatter["description"]), 20, f"Skill description too short in {skill_file}")

    def test_skills_file_reference_integrity(self):
        """Verifies that all local markdown file links across skills and docs point to existing files."""
        md_files = glob.glob(os.path.join(PROJECT_ROOT, "**/*.md"), recursive=True)

        for md_file in md_files:
            file_dir = os.path.dirname(md_file)
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Find markdown links: [text](path)
            links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
            for text, link in links:
                if link.startswith("http://") or link.startswith("https://") or link.startswith("#") or link.startswith("mailto:"):
                    continue
                # Disallow absolute or file:// URI schemes in documentation
                self.assertFalse(
                    link.startswith("file://"),
                    f"Absolute file:// URI not allowed in {md_file}: [{text}]({link})"
                )
                self.assertFalse(
                    link.startswith("/usr/local") or link.startswith("/home/"),
                    f"Machine-specific absolute path not allowed in {md_file}: [{text}]({link})"
                )

                clean_link = link.split("#")[0]
                if not clean_link:
                    continue
                target_path = os.path.normpath(os.path.join(file_dir, clean_link))

                self.assertTrue(
                    os.path.exists(target_path),
                    f"Broken link in {md_file}: link '{link}' points to non-existent path '{target_path}'"
                )


    def test_guardrail_rules_invariants(self):
        """Verifies critical protocol rules are not accidentally removed during skill edits."""
        orchestrator_file = os.path.join(PROJECT_ROOT, "skills/ml-benchmark-orchestrator/SKILL.md")
        self.assertTrue(os.path.exists(orchestrator_file))

        with open(orchestrator_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Invariant Guardrail Checkpoints
        invariants = [
            ("Interactive Alignment", "INTERACTIVE ALIGNMENT FIRST"),
            ("Persistent Asset Protection", "PERSISTENT RESOURCE & DATASET PROTECTION"),
            ("Immutability Rule", "ABSOLUTE IMMUTABILITY RULE"),
            ("Complete Pod Lifecycle Monitoring", "COMPLETE WORKLOAD MONITORING"),
            ("No Freestyle Protocol", "ABSOLUTE NO-FREESTYLE & FAIL-FAST PROTOCOL"),
            ("No Autonomous Bucket Scan", "STRICT PROHIBITION ON AUTONOMOUS BUCKET SCANNING"),
            ("Dataloader Worker Count", "MANDATORY DATALOADER WORKER COUNT VISIBILITY"),
            ("Access Mode Alignment", "MANDATORY STORAGE ACCESS MODE ALIGNMENT"),
            ("Metadata Storm Governance", "METADATA STORM GOVERNANCE"),
        ]

        for rule_name, keyword in invariants:
            self.assertIn(keyword, content, f"Critical Guardrail '{rule_name}' ({keyword}) missing from orchestrator skill!")

    def test_plan_review_table_schema(self):
        """Verifies that Execution Plan Review Table contains all mandatory rows."""
        orchestrator_file = os.path.join(PROJECT_ROOT, "skills/ml-benchmark-orchestrator/SKILL.md")
        with open(orchestrator_file, "r", encoding="utf-8") as f:
            content = f.read()

        mandatory_rows = [
            "**Workload & Model**",
            "**Target GKE Cluster",
            "**Compute Node Specs**",
            "**GCSFuse CSI Driver Version**",
            "**VPC Network MTU Setting**",
            "**Input Dataset Path",
            "**Dataset Format",
            "**Shuffle Strategies",
            "**DataLoader Workers",
            "**Storage Backends Under Test**",
        ]

        for row in mandatory_rows:
            self.assertIn(row, content, f"Mandatory row '{row}' missing from Execution Plan Review Table template!")

    def test_helm_chart_templates_dry_run(self):
        """Renders all workload Helm charts with `helm template` to ensure zero template syntax errors."""
        charts = glob.glob(os.path.join(PROJECT_ROOT, "workloads/*/helm_chart"))
        self.assertGreaterEqual(len(charts), 4, "Should have at least 4 workload charts")

        for chart_dir in charts:
            res = subprocess.run(
                ["helm", "template", "test-release", chart_dir, "-f", os.path.join(chart_dir, "values_base.yaml")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                res.returncode,
                0,
                f"Helm template compilation failed for chart {chart_dir}:\n{res.stderr}"
            )

    def test_bucket_manager_commands_in_skills(self):
        skill_files = glob.glob(os.path.join(PROJECT_ROOT, "skills/**/*.md"), recursive=True)

        for skill_file in skill_files:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()

            matches = re.findall(r"python3 tools/infrastructure/bucket_manager\.py([^\n`]+)", content)
            for m in matches:
                clean_cmd = m.replace("\\", " ").replace("${PROJECT_ID}", "test-proj").replace("${BUCKET_NAME}", "test-b").replace("${PREFIX_PATH}", "prefix").replace("${REGION}", "us-central1").replace("${ZONE}", "us-central1-b").strip()
                tokens = shlex.split(clean_cmd)
                with patch("sys.argv", ["bucket_manager.py"] + tokens):
                    try:
                        args = parse_bucket_args()
                        self.assertIsNotNone(args.action)
                    except SystemExit:
                        self.fail(f"Invalid bucket_manager CLI snippet in {skill_file}: python3 tools/infrastructure/bucket_manager.py {clean_cmd}")

    def test_dataset_generator_commands_in_skills(self):
        skill_files = glob.glob(os.path.join(PROJECT_ROOT, "skills/**/*.md"), recursive=True)

        for skill_file in skill_files:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()

            matches = re.findall(r"python3 tools/datasets/generator\.py([^\n`]+)", content)
            for m in matches:
                clean_cmd = m.replace("\\", " ").replace("${BUCKET_NAME}", "test-b").strip()
                tokens = shlex.split(clean_cmd)
                if not tokens:
                    continue
                with patch("sys.argv", ["generator.py"] + tokens):
                    try:
                        args = parse_gen_args()
                        self.assertIsNotNone(args.output_path)
                    except SystemExit:
                        self.fail(f"Invalid generator CLI snippet in {skill_file}: python3 tools/datasets/generator.py {clean_cmd}")

    def test_cluster_manager_commands_in_skills(self):
        skill_files = glob.glob(os.path.join(PROJECT_ROOT, "skills/**/*.md"), recursive=True)

        for skill_file in skill_files:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()

            matches = re.findall(r"python3 tools/infrastructure/cluster_manager\.py([^\n`]+)", content)
            for m in matches:
                clean_cmd = m.replace("\\", " ").replace("${CLUSTER_NAME}", "c").replace("${ZONE}", "z").replace("${PROJECT_ID}", "p").strip()
                tokens = shlex.split(clean_cmd)
                with patch("sys.argv", ["cluster_manager.py"] + tokens):
                    try:
                        args = parse_cluster_args()
                        self.assertIsNotNone(args.format)
                    except SystemExit:
                        self.fail(f"Invalid cluster_manager CLI snippet in {skill_file}: python3 tools/infrastructure/cluster_manager.py {clean_cmd}")


if __name__ == "__main__":
    unittest.main()
