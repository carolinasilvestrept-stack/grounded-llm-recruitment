from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = PROJECT_DIR / "outputs" / "ai"
OUT_DIR = PROJECT_DIR / "outputs" / "analysis"
DOCS_DIR = PROJECT_DIR / "docs"

METRICS = ["role_alignment_index", "supportiveness_index"]
DEMOGRAPHIC_DIMS = ["gender_condition", "ethnicity_condition", "age_condition"]
INTERSECTIONAL_DIMS = ["gender_condition", "ethnicity_condition", "age_condition"]


def to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def paired_system_tests(baseline: pd.DataFrame, rag: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "variant_id",
        "base_candidate_id",
        "job_id",
        "job_title",
        "gender_condition",
        "ethnicity_condition",
        "age_condition",
        "run_id",
    ]
    keys = [key for key in keys if key in baseline.columns and key in rag.columns]
    rows: list[dict[str, Any]] = []

    for metric in [metric for metric in METRICS if metric in baseline.columns and metric in rag.columns]:
        merged = baseline[keys + [metric]].merge(
            rag[keys + [metric]],
            on=keys,
            suffixes=("_baseline", "_rag"),
        ).dropna(subset=[f"{metric}_baseline", f"{metric}_rag"])
        if len(merged) < 2:
            continue

        before = merged[f"{metric}_baseline"]
        after = merged[f"{metric}_rag"]
        diff = after - before
        t_stat, p_value = stats.ttest_rel(before, after)
        dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan

        rows.append(
            {
                "metric": metric,
                "n_pairs": len(merged),
                "baseline_mean": before.mean(),
                "grounded_rag_mean": after.mean(),
                "mean_difference_rag_minus_baseline": diff.mean(),
                "median_difference_rag_minus_baseline": diff.median(),
                "mean_absolute_difference": diff.abs().mean(),
                "paired_t_statistic": t_stat,
                "paired_p_value": p_value,
                "cohen_dz": dz,
            }
        )

    return pd.DataFrame(rows)


def disparity_range(frame: pd.DataFrame, metric: str, dims: list[str]) -> float:
    means = frame.groupby(dims)[metric].mean().dropna()
    if len(means) < 2:
        return np.nan
    return float(means.max() - means.min())


def disparity_change_by_job(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [metric for metric in METRICS if metric in df.columns]

    for metric in metrics:
        for job_id, job_frame in df.groupby("job_id"):
            job_title = job_frame["job_title"].dropna().iloc[0] if "job_title" in job_frame.columns and not job_frame["job_title"].dropna().empty else job_id
            for dimension in DEMOGRAPHIC_DIMS:
                values: dict[str, float] = {}
                for condition, condition_frame in job_frame.groupby("system_condition"):
                    values[condition] = disparity_range(condition_frame, metric, [dimension])
                rows.append(
                    {
                        "metric": metric,
                        "job_id": job_id,
                        "job_title": job_title,
                        "dimension": dimension,
                        "baseline_disparity": values.get("baseline_llm", np.nan),
                        "grounded_rag_disparity": values.get("grounded_rag", np.nan),
                        "disparity_change_rag_minus_baseline": values.get("grounded_rag", np.nan) - values.get("baseline_llm", np.nan),
                    }
                )

            values = {}
            for condition, condition_frame in job_frame.groupby("system_condition"):
                values[condition] = disparity_range(condition_frame, metric, INTERSECTIONAL_DIMS)
            rows.append(
                {
                    "metric": metric,
                    "job_id": job_id,
                    "job_title": job_title,
                    "dimension": "intersectional_gender_ethnicity_age",
                    "baseline_disparity": values.get("baseline_llm", np.nan),
                    "grounded_rag_disparity": values.get("grounded_rag", np.nan),
                    "disparity_change_rag_minus_baseline": values.get("grounded_rag", np.nan) - values.get("baseline_llm", np.nan),
                }
            )

    return pd.DataFrame(rows)


def text_output_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "system_condition": row.get("system_condition"),
                "variant_id": row.get("variant_id"),
                "job_id": row.get("job_id"),
                "gender_condition": row.get("gender_condition"),
                "ethnicity_condition": row.get("ethnicity_condition"),
                "age_condition": row.get("age_condition"),
                "strength_count": len(parse_list(row.get("job_relevant_strengths"))),
                "concern_count": len(parse_list(row.get("job_relevant_concerns"))),
                "evidence_count": len(parse_list(row.get("supporting_evidence"))),
                "limit_count": len(parse_list(row.get("limits_of_assessment"))),
            }
        )
    return pd.DataFrame(rows)


def counterfactual_disparity_by_case(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["system_condition", "base_candidate_id", "job_id", "job_title", "run_id"]
    group_cols = [column for column in group_cols if column in df.columns]

    for group_values, group in df.groupby(group_cols):
        group_info = dict(zip(group_cols, group_values if isinstance(group_values, tuple) else (group_values,)))
        variant_count = group["variant_id"].nunique() if "variant_id" in group.columns else len(group)
        for metric in [metric for metric in METRICS if metric in group.columns]:
            values = group[metric].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    **group_info,
                    "metric": metric,
                    "variant_count": variant_count,
                    "counterfactual_range": float(values.max() - values.min()),
                    "counterfactual_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min_value": float(values.min()),
                    "max_value": float(values.max()),
                    "mean_value": float(values.mean()),
                }
            )

    return pd.DataFrame(rows)


def summarize_counterfactual_disparity(counterfactual: pd.DataFrame) -> pd.DataFrame:
    if counterfactual.empty:
        return pd.DataFrame()
    return (
        counterfactual.groupby(["system_condition", "metric"])
        .agg(
            n_cases=("counterfactual_range", "count"),
            mean_counterfactual_range=("counterfactual_range", "mean"),
            median_counterfactual_range=("counterfactual_range", "median"),
            max_counterfactual_range=("counterfactual_range", "max"),
            mean_counterfactual_std=("counterfactual_std", "mean"),
        )
        .reset_index()
    )


def paired_counterfactual_tests(counterfactual: pd.DataFrame) -> pd.DataFrame:
    if counterfactual.empty:
        return pd.DataFrame()
    key_cols = ["base_candidate_id", "job_id", "job_title", "run_id", "metric"]
    key_cols = [column for column in key_cols if column in counterfactual.columns]
    baseline = counterfactual[counterfactual["system_condition"] == "baseline_llm"]
    rag = counterfactual[counterfactual["system_condition"] == "grounded_rag"]
    merged = baseline[key_cols + ["counterfactual_range"]].merge(
        rag[key_cols + ["counterfactual_range"]],
        on=key_cols,
        suffixes=("_baseline", "_rag"),
    )

    rows: list[dict[str, Any]] = []
    for metric, metric_frame in merged.groupby("metric"):
        before = metric_frame["counterfactual_range_baseline"]
        after = metric_frame["counterfactual_range_rag"]
        diff = after - before
        t_stat, p_value = stats.ttest_rel(before, after) if len(metric_frame) > 1 else (np.nan, np.nan)
        rows.append(
            {
                "metric": metric,
                "n_pairs": len(metric_frame),
                "baseline_mean_counterfactual_range": before.mean(),
                "grounded_rag_mean_counterfactual_range": after.mean(),
                "mean_change_rag_minus_baseline": diff.mean(),
                "median_change_rag_minus_baseline": diff.median(),
                "paired_t_statistic": t_stat,
                "paired_p_value": p_value,
                "cohen_dz": diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    paired: pd.DataFrame,
    disparity_jobs: pd.DataFrame,
    text_counts: pd.DataFrame,
    counterfactual_summary: pd.DataFrame,
    counterfactual_tests: pd.DataFrame,
    path: Path,
) -> None:
    role_row = paired[paired["metric"] == "role_alignment_index"].iloc[0]
    support_row = paired[paired["metric"] == "supportiveness_index"].iloc[0] if "supportiveness_index" in set(paired["metric"]) else None

    role_jobs = disparity_jobs[disparity_jobs["metric"] == "role_alignment_index"].copy()
    improved = int((role_jobs["disparity_change_rag_minus_baseline"] < 0).sum())
    worsened = int((role_jobs["disparity_change_rag_minus_baseline"] > 0).sum())
    unchanged = int((role_jobs["disparity_change_rag_minus_baseline"] == 0).sum())

    text_summary = text_counts.groupby("system_condition")[["strength_count", "concern_count", "evidence_count", "limit_count"]].mean().round(2)
    text_summary_rows = [
        "| System condition | Strengths | Concerns | Evidence items | Limits |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, row in text_summary.iterrows():
        text_summary_rows.append(
            f"| {condition} | {row['strength_count']:.2f} | {row['concern_count']:.2f} | "
            f"{row['evidence_count']:.2f} | {row['limit_count']:.2f} |"
        )

    counterfactual_rows = [
        "| System condition | Metric | Mean range | Median range | Max range |",
        "|---|---|---:|---:|---:|",
    ]
    if not counterfactual_summary.empty:
        for _, row in counterfactual_summary.iterrows():
            counterfactual_rows.append(
                f"| {row['system_condition']} | {row['metric']} | "
                f"{row['mean_counterfactual_range']:.2f} | {row['median_counterfactual_range']:.2f} | "
                f"{row['max_counterfactual_range']:.2f} |"
            )

    counterfactual_test_rows = [
        "| Metric | Baseline mean range | Grounded RAG mean range | Mean change | Paired p-value | Cohen dz |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if not counterfactual_tests.empty:
        for _, row in counterfactual_tests.iterrows():
            counterfactual_test_rows.append(
                f"| {row['metric']} | {row['baseline_mean_counterfactual_range']:.2f} | "
                f"{row['grounded_rag_mean_counterfactual_range']:.2f} | "
                f"{row['mean_change_rag_minus_baseline']:.2f} | {row['paired_p_value']:.4g} | {row['cohen_dz']:.3f} |"
            )

    lines = [
        "# Advanced Results Summary",
        "",
        "This file summarizes additional analyses that make the current computational results easier to interpret.",
        "",
        "## Paired System Differences",
        "",
        "Because the same candidate-job cases are evaluated by both systems, paired comparisons are methodologically useful.",
        "",
        "| Metric | Baseline mean | Grounded RAG mean | Mean difference | Paired p-value | Cohen dz |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in paired.iterrows():
        lines.append(
            f"| {row['metric']} | {row['baseline_mean']:.2f} | {row['grounded_rag_mean']:.2f} | "
            f"{row['mean_difference_rag_minus_baseline']:.2f} | {row['paired_p_value']:.4g} | {row['cohen_dz']:.3f} |"
        )

    role_rows = paired[paired["metric"] == "role_alignment_index"]
    role_direction = "different"
    if not role_rows.empty:
        role_difference = role_rows.iloc[0]["mean_difference_rag_minus_baseline"]
        role_direction = "higher" if role_difference > 0 else "lower"

    lines.extend(
        [
            "",
            f"The role-alignment paired comparison indicates that grounded RAG produced {role_direction} role-alignment values for the same candidate-job cases. This should not be interpreted as bias reduction by itself.",
            "",
            "## Job-Level Disparity Changes",
            "",
            f"Across role-alignment job-by-dimension comparisons, disparity decreased in {improved} cases, increased in {worsened} cases, and was unchanged in {unchanged} cases.",
            "",
            "The detailed job-level table is saved as `outputs/analysis/job_level_disparity_changes.csv`.",
            "",
            "## Counterfactual Disparity",
            "",
            "For each base resume, job, and run, this metric calculates the range across the 8 demographic variants. Lower values indicate more consistent treatment of otherwise equivalent profiles.",
            "",
            *counterfactual_rows,
            "",
            "Paired counterfactual comparisons:",
            "",
            *counterfactual_test_rows,
            "",
            "Detailed case-level counterfactual ranges are saved as `outputs/analysis/counterfactual_disparity_by_case.csv`.",
            "",
            "## Text Output Counts",
            "",
            "Average text-structure counts by system condition:",
            "",
            *text_summary_rows,
            "",
            "These counts help identify whether one condition produces more strengths, concerns, evidence items, or assessment limitations. They are diagnostic indicators rather than direct fairness metrics.",
            "",
            "## Interpretation",
            "",
            "The additional analyses strengthen the current interpretation: grounded RAG changes output behavior and improves traceability, but the fairness effect is mixed. In the current run, ethnicity and intersectional role-alignment disparities are lower, while gender and age disparities are higher.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    baseline_path = AI_DIR / "baseline_outputs.csv"
    rag_path = AI_DIR / "rag_outputs.csv"
    if not baseline_path.exists() or not rag_path.exists():
        raise FileNotFoundError("Missing baseline or grounded RAG outputs. Run the thesis pipeline first.")

    baseline = to_numeric(pd.read_csv(baseline_path), METRICS)
    rag = to_numeric(pd.read_csv(rag_path), METRICS)
    df = pd.concat([baseline, rag], ignore_index=True)

    paired = paired_system_tests(baseline, rag)
    job_disparities = disparity_change_by_job(df)
    counterfactual = counterfactual_disparity_by_case(df)
    counterfactual_summary = summarize_counterfactual_disparity(counterfactual)
    counterfactual_tests = paired_counterfactual_tests(counterfactual)
    counts = text_output_counts(df)
    count_summary = counts.groupby(["system_condition", "gender_condition", "ethnicity_condition", "age_condition"]).mean(numeric_only=True).reset_index()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    paired.to_csv(OUT_DIR / "paired_system_differences.csv", index=False)
    job_disparities.to_csv(OUT_DIR / "job_level_disparity_changes.csv", index=False)
    counterfactual.to_csv(OUT_DIR / "counterfactual_disparity_by_case.csv", index=False)
    counterfactual_summary.to_csv(OUT_DIR / "counterfactual_disparity_summary.csv", index=False)
    counterfactual_tests.to_csv(OUT_DIR / "counterfactual_disparity_paired_tests.csv", index=False)
    counts.to_csv(OUT_DIR / "text_output_counts_by_case.csv", index=False)
    count_summary.to_csv(OUT_DIR / "text_output_count_summary.csv", index=False)
    write_markdown(
        paired,
        job_disparities,
        counts,
        counterfactual_summary,
        counterfactual_tests,
        DOCS_DIR / "advanced_results_summary.md",
    )

    print(f"Saved: {OUT_DIR / 'paired_system_differences.csv'}")
    print(f"Saved: {OUT_DIR / 'job_level_disparity_changes.csv'}")
    print(f"Saved: {OUT_DIR / 'text_output_count_summary.csv'}")
    print(f"Saved: {DOCS_DIR / 'advanced_results_summary.md'}")
    print(paired)


if __name__ == "__main__":
    main()
