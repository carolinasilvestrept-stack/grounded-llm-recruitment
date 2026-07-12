from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = PROJECT_DIR / "outputs" / "ai"
OUT_DIR = PROJECT_DIR / "outputs" / "analysis"

REQUIRED_COLUMNS = [
    "system_condition",
    "model",
    "run_id",
    "variant_id",
    "base_candidate_id",
    "job_id",
    "gender_condition",
    "ethnicity_condition",
    "age_condition",
    "candidate_summary",
    "job_relevant_strengths",
    "job_relevant_concerns",
    "structured_observations",
    "role_alignment_indicators",
    "supporting_evidence",
    "limits_of_assessment",
    "supportiveness_index",
    "role_alignment_index",
]

JSON_COLUMNS = {
    "job_relevant_strengths": list,
    "job_relevant_concerns": list,
    "structured_observations": dict,
    "role_alignment_indicators": dict,
    "supporting_evidence": list,
    "limits_of_assessment": list,
}


def parse_json_cell(value: object):
    if pd.isna(value):
        raise ValueError("empty")
    if isinstance(value, (list, dict)):
        return value
    return json.loads(str(value))


def validate_file(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return [{"file": path.name, "check": "file_exists", "status": "missing", "detail": str(path)}]

    frame = pd.read_csv(path)
    rows.append({"file": path.name, "check": "row_count", "status": "ok" if len(frame) else "fail", "detail": len(frame)})

    for column in REQUIRED_COLUMNS:
        rows.append({
            "file": path.name,
            "check": f"column:{column}",
            "status": "ok" if column in frame.columns else "fail",
            "detail": "",
        })

    for column in ["role_alignment_index", "supportiveness_index"]:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        missing = int(values.isna().sum())
        out_of_range = int(((values < 0) | (values > 100)).sum())
        rows.append({
            "file": path.name,
            "check": f"numeric:{column}",
            "status": "ok" if missing == 0 and out_of_range == 0 else "fail",
            "detail": f"missing={missing}; out_of_range={out_of_range}",
        })

    for column, expected_type in JSON_COLUMNS.items():
        if column not in frame.columns:
            continue
        failures = 0
        for value in frame[column].head(100):
            try:
                parsed = parse_json_cell(value)
                if not isinstance(parsed, expected_type):
                    failures += 1
            except Exception:
                failures += 1
        rows.append({
            "file": path.name,
            "check": f"json:{column}",
            "status": "ok" if failures == 0 else "fail",
            "detail": f"failures_in_first_100={failures}",
        })

    for column in ["gender_condition", "ethnicity_condition", "age_condition"]:
        if column in frame.columns:
            groups = sorted(str(value) for value in frame[column].dropna().unique())
            rows.append({"file": path.name, "check": f"coverage:{column}", "status": "ok", "detail": "; ".join(groups)})

    return rows


def write_markdown(results: pd.DataFrame, path: Path) -> None:
    lines = ["# AI Output Schema Report", ""]
    if results.empty:
        lines.append("No validation results.")
    else:
        for file_name, file_results in results.groupby("file", dropna=False):
            lines.append(f"## {file_name}")
            for _, row in file_results.iterrows():
                lines.append(f"- {row['status']}: {row['check']} ({row['detail']})")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [AI_DIR / name for name in args.files]
    results = pd.DataFrame([row for path in files for row in validate_file(path)])
    csv_path = OUT_DIR / "ai_output_schema_report.csv"
    md_path = OUT_DIR / "ai_output_schema_report.md"
    results.to_csv(csv_path, index=False)
    write_markdown(results, md_path)
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")

    failing = results[results["status"].isin(["fail", "missing"])]
    if not failing.empty:
        print(f"Schema checks needing attention: {len(failing)}")
        if args.strict:
            raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AI output CSV schema and basic content.")
    parser.add_argument("--files", nargs="+", default=["baseline_outputs.csv", "rag_outputs.csv"])
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

