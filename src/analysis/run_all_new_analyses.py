"""
Runs all the new analyses added during the methodology review in one pass:
  1. Robustness re-analysis (pseudoreplication, proportional range, subgroups,
     intersectional) -- uses existing data, no dependency.
  2. Ablation comparison (baseline vs. ablation vs. grounded) -- requires
     outputs/ai/ablation_fields_no_retrieval_outputs.csv to exist; skipped
     with a clear message if it doesn't (e.g. still generating).
  3. Ragas manual-coding sample export -- uses existing data, no dependency.

This does NOT run any API generation itself (the ablation condition or a
retrieval-query-fixed regeneration must be run separately, since those cost
API calls and take real time) -- it only runs the analysis steps that are
free and instant once the relevant data files exist.

Usage:
    python src\\analysis\\run_all_new_analyses.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = PROJECT_DIR / "outputs" / "ai"
ANALYSIS_DIR = PROJECT_DIR / "src" / "analysis"


def run_step(label: str, script_name: str, required_file: Path | None = None, required_file_hint: str = "") -> None:
    print(f"\n=== {label} ===", flush=True)
    if required_file is not None and not required_file.exists():
        print(f"SKIPPED: {required_file.name} not found yet.")
        if required_file_hint:
            print(required_file_hint)
        return
    script_path = ANALYSIS_DIR / script_name
    completed = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_DIR)
    if completed.returncode:
        print(f"Step failed (exit code {completed.returncode}): {label}")


def main() -> None:
    run_step(
        "1. Robustness re-analysis (pseudoreplication, proportional range, subgroups, intersectional)",
        "robustness_reanalysis.py",
    )

    run_step(
        "2. Ablation comparison (baseline vs. ablation vs. grounded)",
        "ablation_comparison.py",
        required_file=AI_DIR / "ablation_fields_no_retrieval_outputs.csv",
        required_file_hint=(
            "Generate it first with:\n"
            "  python src/generation/run_baseline_llm.py "
            "--prompt-file prompts/ablation_fields_no_retrieval_prompt.md "
            "--output-prefix ablation_fields_no_retrieval_outputs --runs 3 --api-provider azure"
        ),
    )

    run_step(
        "3. Ragas manual-coding sample export",
        "ragas_manual_coding_sample.py",
        required_file=AI_DIR / "rag_outputs.csv",
    )

    print("\nAll available steps complete. Check outputs/analysis/ for:")
    print("  - robustness_reanalysis.md")
    print("  - ablation_comparison.md (if the ablation data was ready)")
    print("  - ragas_manual_coding_sample.csv (fill in coder_verdict by hand, then re-read it for the distribution)")


if __name__ == "__main__":
    main()