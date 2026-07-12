from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

SCRIPT_PATHS = {
    "generate_ai_outputs.py": PROJECT_DIR / "src" / "pipeline" / "generate_ai_outputs.py",
    "inspect_retrieval_similarities.py": PROJECT_DIR / "src" / "diagnostics" / "inspect_retrieval_similarities.py",
    "analyze_bias_disparities_enhanced.py": PROJECT_DIR / "src" / "analysis" / "analyze_bias_disparities_enhanced.py",
    "statistical_tests.py": PROJECT_DIR / "src" / "analysis" / "statistical_tests.py",
    "effect_sizes.py": PROJECT_DIR / "src" / "analysis" / "effect_sizes.py",
    "advanced_results_summary.py": PROJECT_DIR / "src" / "analysis" / "advanced_results_summary.py",
    "bias_category_coverage.py": PROJECT_DIR / "src" / "analysis" / "bias_category_coverage.py",
    "validate_ai_outputs.py": PROJECT_DIR / "src" / "diagnostics" / "validate_ai_outputs.py",
    "results_readiness_report.py": PROJECT_DIR / "src" / "diagnostics" / "results_readiness_report.py",
}


def run_step(label: str, command: list[str], continue_on_error: bool = False) -> None:
    print(f"\n=== {label} ===", flush=True)
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_DIR)
    if completed.returncode and not continue_on_error:
        raise SystemExit(completed.returncode)
    if completed.returncode:
        print(f"Skipped after error in optional step: {label}", flush=True)


def python_script(script_name: str, *args: str) -> list[str]:
    return [sys.executable, str(SCRIPT_PATHS[script_name]), *args]


def main(args: argparse.Namespace) -> None:
    generation_args = [
        "--embedding-provider",
        "hash" if args.dry_run else args.embedding_provider,
        "--runs",
        str(args.runs),
    ]
    if args.limit is not None:
        generation_args.extend(["--limit", str(args.limit)])
    if args.use_sample:
        generation_args.append("--use-sample")
    if args.dry_run:
        generation_args.append("--dry-run")
    if args.resume:
        generation_args.append("--resume")
    if args.rebuild_rag_index:
        generation_args.append("--rebuild-rag-index")
    if args.api_provider:
        generation_args.extend(["--api-provider", args.api_provider])
    if args.model:
        generation_args.extend(["--model", args.model])

    if not args.skip_generation:
        run_step("Generate baseline and grounded RAG outputs", python_script("generate_ai_outputs.py", *generation_args))
        if not args.dry_run:
            run_step("Validate AI output schema", python_script("validate_ai_outputs.py"), continue_on_error=True)

    if args.dry_run and not args.analyze_dry_run:
        print("\nDry-run generation completed. Skipping analysis tables by default.", flush=True)
        return

    if args.skip_analysis:
        return

    analysis_steps = [
        ("Inspect retrieval similarities", python_script("inspect_retrieval_similarities.py"), True),
        ("Analyze bias disparities", python_script("analyze_bias_disparities_enhanced.py"), False),
        ("Run statistical tests", python_script("statistical_tests.py"), True),
        ("Calculate effect sizes", python_script("effect_sizes.py"), True),
        ("Generate advanced results summary", python_script("advanced_results_summary.py"), True),
        ("Analyze bias-category coverage", python_script("bias_category_coverage.py"), True),
    ]
    for label, command, optional in analysis_steps:
        run_step(label, command, continue_on_error=optional)

    run_step("Generate results-readiness report", python_script("results_readiness_report.py"), continue_on_error=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the thesis computational workflow.")
    parser.add_argument("--embedding-provider", choices=["sentence-transformers", "openai"], default="sentence-transformers")
    parser.add_argument("--api-provider", choices=["auto", "openai", "azure"], default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-sample", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip generation records already present in the output files. Use this to safely re-run after an interrupted or partial generation.")
    parser.add_argument("--rebuild-rag-index", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
