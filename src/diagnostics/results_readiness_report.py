from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_DIR / "outputs" / "analysis" / "results_readiness_report.md"


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def line(ok: bool, text: str) -> str:
    return f"- {'OK' if ok else 'CHECK'}: {text}"


def output_summary(name: str, frame: pd.DataFrame | None) -> list[str]:
    if frame is None:
        return [line(False, f"{name} output is missing.")]
    lines = [line(len(frame) > 0, f"{name} rows: {len(frame)}")]
    for column in ["gender_condition", "ethnicity_condition", "age_condition"]:
        if column in frame.columns:
            lines.append(line(True, f"{name} {column} groups: {', '.join(sorted(map(str, frame[column].dropna().unique())))}"))
        else:
            lines.append(line(False, f"{name} missing {column}"))
    if "role_alignment_index" in frame.columns:
        values = pd.to_numeric(frame["role_alignment_index"], errors="coerce")
        lines.append(line(values.notna().all(), f"{name} role_alignment_index valid values: {int(values.notna().sum())}/{len(values)}"))
    else:
        lines.append(line(False, f"{name} missing role_alignment_index"))
    return lines


def main(args: argparse.Namespace) -> None:
    data_dir = PROJECT_DIR / "data"
    outputs_dir = PROJECT_DIR / "outputs"

    candidates = read_csv(data_dir / "processed" / "controlled_candidate_profiles.csv")
    jobs = read_csv(data_dir / "processed" / "job_descriptions.csv")
    baseline = read_csv(outputs_dir / "ai" / "baseline_outputs.csv")
    rag = read_csv(outputs_dir / "ai" / "rag_outputs.csv")

    lines: list[str] = ["# Results Readiness Report", ""]

    lines.append("## Data Preparation")
    if candidates is None:
        lines.append(line(False, "Controlled candidate profiles are missing."))
    else:
        base_count = candidates["base_candidate_id"].nunique() if "base_candidate_id" in candidates.columns else "unknown"
        lines.append(line(True, f"Controlled candidate profiles: {len(candidates)} rows"))
        lines.append(line(True, f"Base resumes: {base_count}"))
        if "source_dataset" in candidates.columns:
            lines.append(line(True, f"Source datasets: {', '.join(sorted(map(str, candidates['source_dataset'].dropna().unique())))}"))
    lines.append(line(jobs is not None and len(jobs) > 0, f"Job descriptions: {0 if jobs is None else len(jobs)}"))

    lines.append("\n## AI Outputs")
    lines.extend(output_summary("baseline", baseline))
    lines.extend(output_summary("grounded RAG", rag))

    expected_rows = None
    if candidates is not None and jobs is not None:
        expected_rows = len(candidates) * len(jobs)
        lines.append(line(True, f"Expected rows per run: {expected_rows}"))
    for name, frame in [("baseline", baseline), ("grounded RAG", rag)]:
        if frame is not None and expected_rows is not None:
            runs = frame["run_id"].nunique() if "run_id" in frame.columns else 1
            lines.append(line(len(frame) == expected_rows * runs, f"{name} row coverage: {len(frame)} rows across {runs} run(s)"))

    lines.append("\n## Analysis Tables")
    analysis_files = [
        "combined_ai_outputs.csv",
        "system_comparison.csv",
        "disparity_summary.csv",
        "intersectional_disparity_summary.csv",
        "enhanced_metric_group_summary.csv",
        "enhanced_disparity_ranges.csv",
        "retrieval_similarity_summary.csv",
        "statistical_tests_ai_outputs.csv",
        "effect_sizes_ai_outputs.csv",
        "counterfactual_disparity_summary.csv",
        "bias_category_coverage_summary.csv",
        "ai_output_schema_report.csv",
    ]
    for filename in analysis_files:
        path = outputs_dir / "analysis" / filename
        lines.append(line(path.exists(), f"{filename}: {'present' if path.exists() else 'missing'}"))

    lines.append("\n## Research Question Readiness")
    rq_ready = baseline is not None and rag is not None and candidates is not None
    lines.append(line(rq_ready, "RQ1-RQ3 (Section 2.8) can be tested when baseline and grounded RAG outputs are present."))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {REPORT_PATH}")

    if args.strict and any(item.startswith("- CHECK") for item in lines):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a thesis results-readiness report.")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
