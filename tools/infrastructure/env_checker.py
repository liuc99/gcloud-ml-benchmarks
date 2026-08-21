#!/usr/bin/env python3
"""
Environment and Dependency Pre-flight Diagnostic Tool.

Checks local CLI tools (kubectl, helm, gcloud, python3, git),
Python library dependencies (google-cloud-storage, pyyaml, pyarrow, etc.),
and GCP/Kubernetes authentication and cluster connectivity.
"""

import argparse
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ENV_CHECKER] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)


# Required CLI tools and their purpose / install guidance
CLI_TOOLS = {
    "gcloud": {
        "required": True,
        "purpose": "GCP resource management and authentication",
        "install_cmd": "Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install",
        "version_cmd": "gcloud --version",
    },
    "kubectl": {
        "required": True,
        "purpose": "Kubernetes cluster diagnostics and pod status monitoring",
        "install_cmd": "gcloud components install kubectl",
        "version_cmd": "kubectl version --client -o json",
    },
    "helm": {
        "required": True,
        "purpose": "Deploying and managing workload benchmark Helm charts",
        "install_cmd": "https://helm.sh/docs/intro/install/",
        "version_cmd": "helm version --short",
    },
    "python3": {
        "required": True,
        "purpose": "Executing repository tools and data processing scripts",
        "install_cmd": "Install Python 3.8+",
        "version_cmd": "python3 --version",
    },
    "git": {
        "required": False,
        "purpose": "Source version control inspection",
        "install_cmd": "sudo apt-get install git",
        "version_cmd": "git --version",
    },
}

# Required Python packages
PYTHON_PACKAGES = {
    "google.cloud.storage": {
        "package_name": "google-cloud-storage",
        "required": True,
        "purpose": "GCS bucket lifecycle management and dataset inspection",
    },
    "yaml": {
        "package_name": "pyyaml",
        "required": True,
        "purpose": "Parsing Helm values files and Kubernetes manifests",
    },
    "pyarrow": {
        "package_name": "pyarrow",
        "required": False,
        "purpose": "Parquet dataset inspection and preprocessing",
    },
    "pandas": {
        "package_name": "pandas",
        "required": False,
        "purpose": "Data manipulation and analysis",
    },
    "requests": {
        "package_name": "requests",
        "required": False,
        "purpose": "HTTP API requests",
    },
}


def run_cmd(cmd, timeout=10):
    try:
        res = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True, timeout=timeout
        )
        return res.stdout.strip()
    except Exception as e:
        logging.debug(f"Command failed: {cmd} | Exception: {e}")
        return None


def check_cli_tools():
    results = {}
    all_critical_passed = True

    for tool, meta in CLI_TOOLS.items():
        executable_path = shutil.which(tool)
        present = executable_path is not None
        version_str = "Not installed"

        if present:
            raw_ver = run_cmd(meta["version_cmd"])
            if raw_ver:
                if tool == "kubectl" and raw_ver.startswith("{"):
                    try:
                        vdata = json.loads(raw_ver)
                        version_str = vdata.get("clientVersion", {}).get("gitVersion", raw_ver)
                    except Exception:
                        version_str = raw_ver.split("\n")[0]
                else:
                    version_str = raw_ver.split("\n")[0]
            else:
                version_str = "Installed (version string unavailable)"

        status = "PASS" if present else ("FAIL" if meta["required"] else "WARN")
        if meta["required"] and not present:
            all_critical_passed = False

        results[tool] = {
            "present": present,
            "status": status,
            "required": meta["required"],
            "path": executable_path or "N/A",
            "version": version_str,
            "purpose": meta["purpose"],
            "install_cmd": meta["install_cmd"],
        }

    return results, all_critical_passed


def check_python_packages():
    results = {}
    missing_packages = []
    all_critical_passed = True

    for mod_name, meta in PYTHON_PACKAGES.items():
        spec = importlib.util.find_spec(mod_name)
        present = spec is not None
        version = "N/A"

        if present:
            try:
                mod = importlib.import_module(mod_name)
                version = getattr(mod, "__version__", "Installed")
            except Exception:
                version = "Installed"

        status = "PASS" if present else ("FAIL" if meta["required"] else "WARN")
        if not present:
            missing_packages.append(meta["package_name"])
            if meta["required"]:
                all_critical_passed = False

        results[meta["package_name"]] = {
            "module": mod_name,
            "present": present,
            "status": status,
            "required": meta["required"],
            "version": version,
            "purpose": meta["purpose"],
        }

    return results, missing_packages, all_critical_passed


def check_auth_and_connectivity():
    gcloud_account = run_cmd("gcloud config get-value account 2>/dev/null")
    gcloud_project = run_cmd("gcloud config get-value project 2>/dev/null")
    k8s_context = run_cmd("kubectl config current-context 2>/dev/null")

    k8s_connected = False
    if k8s_context:
        # Quick check if cluster endpoint is reachable
        cluster_info = run_cmd("kubectl cluster-info 2>/dev/null", timeout=5)
        k8s_connected = cluster_info is not None

    status = "PASS" if (gcloud_account and gcloud_project and k8s_connected) else "WARN"

    return {
        "status": status,
        "gcloud_account": gcloud_account or "Not authenticated (Run: gcloud auth login)",
        "gcloud_project": gcloud_project or "Not set (Run: gcloud config set project <PROJECT_ID>)",
        "k8s_context": k8s_context or "No active context (Run: gcloud container clusters get-credentials ...)",
        "k8s_cluster_reachable": k8s_connected,
    }


def run_env_check():
    cli_results, cli_ok = check_cli_tools()
    py_results, missing_pkgs, py_ok = check_python_packages()
    auth_results = check_auth_and_connectivity()

    overall_passed = cli_ok and py_ok

    summary = {
        "overall_status": "PASS" if overall_passed else "FAIL",
        "cli_tools": cli_results,
        "python_packages": py_results,
        "missing_python_packages": missing_pkgs,
        "suggested_pip_command": f"pip install {' '.join(missing_pkgs)}" if missing_pkgs else None,
        "auth_and_connectivity": auth_results,
    }

    return summary, overall_passed


def print_table_report(summary):
    print("\n==========================================================================")
    print("           ENVIRONMENT & DEPENDENCY PRE-FLIGHT DIAGNOSTIC REPORT          ")
    print("==========================================================================")

    overall_str = "✅ PASS" if summary["overall_status"] == "PASS" else "❌ FAIL (Action Required)"
    print(f"Overall Pre-flight Status: {overall_str}\n")

    print("--- 1. System CLI Tools ---")
    print(f"{'TOOL':<12} {'STATUS':<8} {'REQUIRED':<10} {'VERSION / PATH'}")
    print("-" * 74)
    for tool, meta in summary["cli_tools"].items():
        status_icon = "✅ PASS" if meta["status"] == "PASS" else ("❌ FAIL" if meta["status"] == "FAIL" else "⚠️ WARN")
        req_str = "Yes" if meta["required"] else "No"
        print(f"{tool:<12} {status_icon:<8} {req_str:<10} {meta['version']}")
        if meta["status"] == "FAIL":
            print(f"   ↳ Remediation: {meta['install_cmd']}")

    print("\n--- 2. Python Packages ---")
    print(f"{'PACKAGE':<22} {'STATUS':<8} {'REQUIRED':<10} {'VERSION'}")
    print("-" * 74)
    for pkg, meta in summary["python_packages"].items():
        status_icon = "✅ PASS" if meta["status"] == "PASS" else ("❌ FAIL" if meta["status"] == "FAIL" else "⚠️ WARN")
        req_str = "Yes" if meta["required"] else "No"
        print(f"{pkg:<22} {status_icon:<8} {req_str:<10} {meta['version']}")

    if summary["missing_python_packages"]:
        print(f"\n💡 Missing Python Packages detected. Run the following command to install:")
        print(f"   {summary['suggested_pip_command']}")

    print("\n--- 3. GCP & Kubernetes Authentication / Context ---")
    auth = summary["auth_and_connectivity"]
    print(f"GCP Active Account:   {auth['gcloud_account']}")
    print(f"GCP Active Project:   {auth['gcloud_project']}")
    print(f"K8s Active Context:   {auth['k8s_context']}")
    k8s_reach_str = "✅ Connected" if auth['k8s_cluster_reachable'] else "⚠️ Unreachable / Timeout"
    print(f"K8s Cluster Access:   {k8s_reach_str}")
    print("==========================================================================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Environment & Dependency Pre-flight Checker")
    parser.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "json"],
        help="Output format: 'table' or 'json'",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    summary, passed = run_env_check()

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print_table_report(summary)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
