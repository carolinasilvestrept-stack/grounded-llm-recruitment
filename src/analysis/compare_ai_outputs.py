import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def find_project_dir() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "outputs").exists() or (candidate / "data").exists():
            return candidate
    return here.parent

PROJECT_DIR = find_project_dir()
OUTPUT_DIR = PROJECT_DIR / "outputs" / "analysis"
BASELINE_PATH = PROJECT_DIR / "outputs" / "ai" / "baseline_outputs.csv"
RAG_PATH = PROJECT_DIR / "outputs" / "ai" / "rag_outputs.csv"


def load_outputs(path: Path, condition: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "system_condition" not in frame.columns:
        frame["system_condition"] = condition
    if "alignment_index" not in frame.columns and "role_alignment_index" in frame.columns:
        frame["alignment_index"] = pd.to_numeric(frame["role_alignment_index"], errors="coerce")
    if "alignment_index" not in frame.columns and "score" in frame.columns:
        frame["alignment_index"] = pd.to_numeric(frame["score"], errors="coerce")
    if "alignment_index" not in frame.columns:
        frame["alignment_index"] = pd.NA
    frame["alignment_index"] = pd.to_numeric(frame["alignment_index"], errors="coerce")
    return frame


def consolidate(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "system_condition", "variant_id", "base_candidate_id", "job_id", "job_title",
        "gender_condition", "ethnicity_condition", "age_condition",
    ]
    keys = [k for k in keys if k in frame.columns]
    agg = {"alignment_index": "mean"}
    for col in ["candidate_summary", "qualification_alignment", "bias_awareness_notes", "fairness_guidance", "retrieved_chunk_ids"]:
        if col in frame.columns:
            agg[col] = "first"
    return frame.groupby(keys, dropna=False, as_index=False).agg(agg)


def build_system_comparison(baseline: pd.DataFrame, rag: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant_id", "base_candidate_id", "job_id", "job_title", "gender_condition", "ethnicity_condition", "age_condition"]
    keys = [k for k in keys if k in baseline.columns and k in rag.columns]
    comparison = baseline[keys + ["alignment_index"]].merge(
        rag[keys + ["alignment_index"]], on=keys, suffixes=("_baseline", "_rag")
    )
    comparison["alignment_difference_rag_minus_baseline"] = comparison["alignment_index_rag"] - comparison["alignment_index_baseline"]
    comparison["absolute_alignment_difference"] = comparison["alignment_difference_rag_minus_baseline"].abs()
    return comparison


def disparity_summary(combined: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for system_condition, system_frame in combined.groupby("system_condition"):
        for dimension in ["gender_condition", "ethnicity_condition", "age_condition"]:
            if dimension not in system_frame.columns:
                continue
            grouped = system_frame.groupby(dimension, dropna=False).agg(
                n=("alignment_index", "count"),
                mean_alignment=("alignment_index", "mean"),
                std_alignment=("alignment_index", "std"),
            ).reset_index().rename(columns={dimension: "group"})
            max_mean = grouped["mean_alignment"].max()
            for _, row in grouped.iterrows():
                rows.append({
                    "system_condition": system_condition,
                    "dimension": dimension,
                    "group": row["group"],
                    "n": row["n"],
                    "mean_alignment": row["mean_alignment"],
                    "std_alignment": row["std_alignment"],
                    "alignment_gap_from_highest_group": row["mean_alignment"] - max_mean,
                })
    return pd.DataFrame(rows)


def intersectional_summary(combined: pd.DataFrame) -> pd.DataFrame:
    dims = ["gender_condition", "ethnicity_condition", "age_condition"]
    if not all(d in combined.columns for d in dims):
        return pd.DataFrame()
    return combined.groupby(["system_condition"] + dims, dropna=False).agg(
        n=("alignment_index", "count"),
        mean_alignment=("alignment_index", "mean"),
        std_alignment=("alignment_index", "std"),
    ).reset_index()


def write_csv(frame: pd.DataFrame, name: str, output_prefix: str) -> Path:
    path = OUTPUT_DIR / f"{output_prefix}{name}"
    frame.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and grounded RAG AI-output indicators.")
    parser.add_argument("--baseline-output", type=Path, default=BASELINE_PATH)
    parser.add_argument("--rag-output", type=Path, default=RAG_PATH)
    parser.add_argument("--output-prefix", default="")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = consolidate(load_outputs(args.baseline_output, "baseline_llm"))
    rag = consolidate(load_outputs(args.rag_output, "grounded_rag"))
    combined = pd.concat([baseline, rag], ignore_index=True)
    comparison = build_system_comparison(baseline, rag)
    disparity = disparity_summary(combined)
    intersectional = intersectional_summary(combined)

    combined_path = write_csv(combined, "combined_ai_outputs.csv", args.output_prefix)
    comparison_path = write_csv(comparison, "system_comparison.csv", args.output_prefix)
    disparity_path = write_csv(disparity, "disparity_summary.csv", args.output_prefix)
    write_csv(intersectional, "intersectional_disparity_summary.csv", args.output_prefix)

    summary = {
        "baseline_mean_alignment": float(baseline["alignment_index"].mean()),
        "rag_mean_alignment": float(rag["alignment_index"].mean()),
        "mean_alignment_difference_rag_minus_baseline": float(comparison["alignment_difference_rag_minus_baseline"].mean()) if not comparison.empty else None,
        "mean_absolute_alignment_difference": float(comparison["absolute_alignment_difference"].mean()) if not comparison.empty else None,
    }
    (OUTPUT_DIR / f"{args.output_prefix}ai_output_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Combined outputs: {combined_path}")
    print(f"System comparison: {comparison_path}")
    print(f"Disparity summary: {disparity_path}")


if __name__ == "__main__":
    main(parse_args())
