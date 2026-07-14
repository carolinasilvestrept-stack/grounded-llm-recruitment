"""
Compares the original grounded RAG condition (rag_outputs.csv, retrieval
query included demographic cues) against the retrieval-fixed regeneration
(rag_outputs_query_fixed.csv, retrieval query built from core_resume_text
only), both against the same baseline, to quantify exactly what the
retrieval-query fix changed.

Usage (after generating rag_outputs_query_fixed.csv):
    python src\\analysis\\retrieval_fix_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = PROJECT_DIR / "outputs" / "ai"
OUT_DIR = PROJECT_DIR / "outputs" / "analysis"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bias_category_coverage import analyze_coverage, summarize_coverage  # noqa: E402


def coverage_by_category(df: pd.DataFrame) -> dict:
    cov = analyze_coverage(df)
    summary = summarize_coverage(cov)
    row = summary.iloc[0]
    return {col: row[col] for col in summary.columns if col != "system_condition"}


def case_range(df: pd.DataFrame, metric: str) -> float:
    df = df.copy()
    df["case_key"] = df["base_candidate_id"].astype(str) + "_" + df["job_id"].astype(str) + "_" + df["run_id"].astype(str)
    return df.groupby("case_key")[metric].agg(lambda x: x.max() - x.min()).mean()


def paired_test(baseline: pd.DataFrame, grounded: pd.DataFrame, metric: str) -> dict:
    keys = ["base_candidate_id", "job_id", "variant_id", "run_id"]
    b = baseline.set_index(keys)[metric]
    g = grounded.set_index(keys)[metric]
    common = b.index.intersection(g.index)
    diff = g.loc[common] - b.loc[common]
    t, p = stats.ttest_rel(g.loc[common], b.loc[common])
    return {
        "n_pairs": len(common),
        "baseline_mean": b.loc[common].mean(),
        "grounded_mean": g.loc[common].mean(),
        "mean_diff": diff.mean(),
        "t_statistic": t,
        "p_value": p,
        "cohens_dz": diff.mean() / diff.std(),
    }


def main() -> None:
    baseline_path = AI_DIR / "baseline_outputs.csv"
    original_path = AI_DIR / "rag_outputs.csv"
    fixed_path = AI_DIR / "rag_outputs_query_fixed.csv"

    if not fixed_path.exists():
        print(f"Missing {fixed_path}. Generate it first with run_rag_system.py --output-prefix rag_outputs_query_fixed.")
        return

    baseline = pd.read_csv(baseline_path)
    original = pd.read_csv(original_path)
    fixed = pd.read_csv(fixed_path)

    lines = ["# Retrieval-Query Fix: Before/After Comparison", ""]
    lines.append(f"N: baseline={len(baseline)}, original_grounded={len(original)}, fixed_grounded={len(fixed)}")
    lines.append("")

    lines.append("## Role-alignment index: paired comparison vs. baseline")
    for label, grounded_df in [("Original (query included demographic cues)", original), ("Fixed (query excludes demographic cues)", fixed)]:
        res = paired_test(baseline, grounded_df, "role_alignment_index")
        lines.append(
            f"- **{label}**: n={res['n_pairs']}, baseline={res['baseline_mean']:.2f}, grounded={res['grounded_mean']:.2f}, "
            f"diff={res['mean_diff']:.3f}, t={res['t_statistic']:.3f}, p={res['p_value']:.4g}, Cohen's dz={res['cohens_dz']:.3f}"
        )
    lines.append("")

    lines.append("## Counterfactual range (role_alignment_index): before vs. after the fix")
    lines.append(f"- Baseline: {case_range(baseline, 'role_alignment_index'):.3f}")
    lines.append(f"- Original grounded: {case_range(original, 'role_alignment_index'):.3f}")
    lines.append(f"- Fixed grounded: {case_range(fixed, 'role_alignment_index'):.3f}")
    lines.append("")

    lines.append("## Bias-keyword coverage (%): original vs. fixed")
    cov_orig = coverage_by_category(original)
    cov_fixed = coverage_by_category(fixed)
    for cat in cov_orig:
        lines.append(f"- {cat}: original={cov_orig[cat]:.1f}%, fixed={cov_fixed.get(cat, float('nan')):.1f}%")
    lines.append("")

    lines.append("## Interpretation")
    lines.append(
        "If the fixed condition's counterfactual range is similar to or smaller than the original's, this suggests "
        "the demographic-cue leakage in the retrieval query was a meaningful contributor to the RQ2 variance increase. "
        "If the numbers are nearly identical, the leakage's practical impact on this specific result was small, even "
        "though it remained a genuine methodological flaw worth fixing."
    )

    report_path = OUT_DIR / "retrieval_fix_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()