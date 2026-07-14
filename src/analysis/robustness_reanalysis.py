"""
Robustness re-analysis of the existing N=1,800 dataset, addressing four
methodological points raised in review: pseudoreplication (candidate-level
clustering), proportional (relative) counterfactual range, demographic
subgroup direction of disparity, and intersectional (2x2x2) breakdown.

This script uses only data already generated; it does not call any API
and does not need Azure/OpenAI credentials.

Usage:
    python src\\analysis\\robustness_reanalysis.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = PROJECT_DIR / "outputs" / "ai"
OUT_DIR = PROJECT_DIR / "outputs" / "analysis"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pd.read_csv(AI_DIR / "baseline_outputs.csv")
    r = pd.read_csv(AI_DIR / "rag_outputs.csv")
    return b, r


def candidate_level_test(b: pd.DataFrame, r: pd.DataFrame, metric: str) -> dict:
    b_cand = b.groupby("base_candidate_id")[metric].mean()
    r_cand = r.groupby("base_candidate_id")[metric].mean()
    common = b_cand.index.intersection(r_cand.index)
    t, p = stats.ttest_rel(r_cand[common], b_cand[common])
    diff = r_cand[common] - b_cand[common]
    dz = diff.mean() / diff.std()
    return {
        "metric": metric,
        "n_candidates": len(common),
        "baseline_mean": b_cand[common].mean(),
        "grounded_mean": r_cand[common].mean(),
        "mean_diff": diff.mean(),
        "t_statistic": t,
        "p_value": p,
        "cohens_dz": dz,
    }


def proportional_range(df: pd.DataFrame, metric: str) -> dict:
    df = df.copy()
    df["case_key"] = df["base_candidate_id"].astype(str) + "_" + df["job_id"].astype(str) + "_" + df["run_id"].astype(str)
    ranges = df.groupby("case_key")[metric].agg(lambda x: x.max() - x.min())
    mean_range = ranges.mean()
    mean_score = df[metric].mean()
    return {
        "metric": metric,
        "mean_absolute_range": mean_range,
        "mean_score": mean_score,
        "range_as_pct_of_mean": 100 * mean_range / mean_score,
    }


def subgroup_direction(df: pd.DataFrame, metric: str, condition: str) -> pd.DataFrame:
    return df.groupby(condition)[metric].agg(["mean", "std", "count"]).reset_index()


def intersectional_breakdown(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.groupby(["gender_condition", "ethnicity_condition", "age_condition"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def main() -> None:
    b, r = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = ["# Robustness Re-analysis (existing N=1,800 dataset)", ""]

    report_lines.append("## 1. Candidate-level analysis (addresses pseudoreplication)")
    report_lines.append("")
    for metric in ["role_alignment_index", "supportiveness_index"]:
        res = candidate_level_test(b, r, metric)
        report_lines.append(
            f"- **{metric}**: N={res['n_candidates']} candidates, baseline={res['baseline_mean']:.2f}, "
            f"grounded={res['grounded_mean']:.2f}, diff={res['mean_diff']:.3f}, "
            f"t={res['t_statistic']:.3f}, p={res['p_value']:.4g}, Cohen's dz={res['cohens_dz']:.3f}"
        )
    report_lines.append("")

    report_lines.append("## 2. Proportional (relative) counterfactual range")
    report_lines.append("")
    for metric in ["role_alignment_index", "supportiveness_index"]:
        b_prop = proportional_range(b, metric)
        r_prop = proportional_range(r, metric)
        report_lines.append(
            f"- **{metric}**: baseline range = {b_prop['mean_absolute_range']:.2f} "
            f"({b_prop['range_as_pct_of_mean']:.1f}% of mean); "
            f"grounded range = {r_prop['mean_absolute_range']:.2f} "
            f"({r_prop['range_as_pct_of_mean']:.1f}% of mean)"
        )
    report_lines.append("")

    report_lines.append("## 3. Subgroup direction of disparity (role_alignment_index)")
    report_lines.append("")
    for condition in ["gender_condition", "ethnicity_condition", "age_condition"]:
        report_lines.append(f"### {condition}")
        for label, df in [("baseline", b), ("grounded", r)]:
            sub = subgroup_direction(df, "role_alignment_index", condition)
            report_lines.append(f"**{label}**:")
            for _, row in sub.iterrows():
                report_lines.append(f"  - {row[condition]}: mean={row['mean']:.2f}, std={row['std']:.2f}, n={int(row['count'])}")
        report_lines.append("")

    report_lines.append("## 4. Intersectional breakdown (gender x ethnicity x age, role_alignment_index)")
    report_lines.append("")
    for label, df in [("baseline", b), ("grounded", r)]:
        report_lines.append(f"**{label}**:")
        inter = intersectional_breakdown(df, "role_alignment_index")
        for _, row in inter.iterrows():
            report_lines.append(
                f"  - {row['gender_condition']}/{row['ethnicity_condition']}/{row['age_condition']}: "
                f"mean={row['mean']:.2f}, std={row['std']:.2f}, n={int(row['count'])}"
            )
        report_lines.append("")

    report_path = OUT_DIR / "robustness_reanalysis.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()