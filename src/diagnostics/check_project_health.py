from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_DIR / "outputs" / "analysis" / "project_health_report.md"


def status_line(ok: bool, label: str, detail: str = "") -> str:
    marker = "OK" if ok else "CHECK"
    suffix = f" - {detail}" if detail else ""
    return f"- {marker}: {label}{suffix}"


def import_check(module_name: str, import_target: str | None = None) -> tuple[bool, str]:
    try:
        importlib.import_module(import_target or module_name)
        return True, "imported"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def file_check(relative_path: str) -> tuple[bool, str]:
    path = PROJECT_DIR / relative_path
    if not path.exists():
        return False, "missing"
    if path.is_file():
        return True, f"{path.stat().st_size} bytes"
    return True, "directory exists"


def csv_count(relative_path: str) -> tuple[bool, str]:
    path = PROJECT_DIR / relative_path
    if not path.exists():
        return False, "missing"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return False, f"cannot read CSV: {exc}"
    return True, f"{len(frame)} rows"


def env_check() -> list[str]:
    azure_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT"]
    openai_vars = ["OPENAI_API_KEY"]
    azure_ready = all(os.getenv(name) for name in azure_vars)
    openai_ready = all(os.getenv(name) for name in openai_vars)
    missing_azure = [name for name in azure_vars if not os.getenv(name)]
    lines = [
        status_line(azure_ready, "Azure OpenAI credentials", "configured" if azure_ready else f"missing: {', '.join(missing_azure)}"),
        status_line(openai_ready, "OpenAI API key", "configured" if openai_ready else "missing: OPENAI_API_KEY"),
    ]
    return lines


def main(args: argparse.Namespace) -> None:
    sections: list[str] = []
    failures = 0

    sections.append("# Project Health Report\n")
    sections.append("## Runtime")
    sections.append(status_line(True, "Python", sys.version.replace("\n", " ")))
    sections.append(status_line(True, "Executable", sys.executable))
    sections.append(status_line(True, "Platform", platform.platform()))

    sections.append("\n## Package Imports")
    packages = [
        ("pandas", None),
        ("numpy", None),
        ("scipy.stats", "scipy.stats"),
        ("sentence_transformers", None),
        ("faiss", None),
        ("openai", None),
    ]
    for label, target in packages:
        ok, detail = import_check(label, target)
        failures += 0 if ok or label in {"faiss", "openai"} else 1
        sections.append(status_line(ok, label, detail))

    sections.append("\n## Required Files")
    required_files = [
        "data/processed/controlled_candidate_profiles.csv",
        "data/processed/job_descriptions.csv",
        "prompts/baseline_prompt.md",
        "prompts/grounded_prompt.md",
        "config/demographic_conditions.json",
        "config/job_descriptions.json",
        "knowledge_base/fairness_guidelines.md",
        "knowledge_base/debiasing_principles.md",
        "knowledge_base/evaluation_rubric.md",
        "knowledge_base/hiring_criteria.md",
    ]
    for relative in required_files:
        ok, detail = file_check(relative)
        failures += 0 if ok else 1
        sections.append(status_line(ok, relative, detail))

    sections.append("\n## Data Tables")
    for relative in [
        "data/processed/controlled_candidate_profiles.csv",
        "data/processed/job_descriptions.csv",
        "outputs/ai/baseline_outputs.csv",
        "outputs/ai/rag_outputs.csv",
    ]:
        ok, detail = csv_count(relative)
        sections.append(status_line(ok, relative, detail))

    sections.append("\n## API Environment")
    sections.extend(env_check())

    report = "\n".join(sections) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {REPORT_PATH}")

    if args.strict and failures:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check project health before final thesis runs.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if required files/imports are missing.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
