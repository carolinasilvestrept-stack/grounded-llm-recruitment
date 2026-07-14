"""
Three-way ablation comparison: baseline vs. ablation (baseline schema + bias
fields, no retrieval) vs. grounded (retrieval + bias fields), isolating
whether the RQ3 text-coverage jump is attributable to retrieval specifically
or to simply being asked to produce bias-related output fields.

Usage (after generating outputs/ai/ablation_fields_no_retrieval_outputs.csv
with src/generation/run_baseline_llm.py --prompt-file prompts/ablation_fields_no_retrieval_prompt.md):

    python src\\analysis\\ablation_comparison.py
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


def main() -> None:
    baseline_path = AI_DIR / "baseline_outputs.csv"
    ablation_path = AI_DIR / "ablation_fields_no_retrieval_outputs.csv"
    grounded_path = AI_DIR / "rag_outputs.csv"

    if not ablation_path.exists():
        print(f"Missing {ablation_path}.")
        print("Generate it first with:")
        print("  python src/generation/run_baseline_llm.py --prompt-file prompts/ablation_fields_no_retrieval_prompt.md "
              "--output-prefix ablation_fields_no_retrieval_outputs --runs 3 --api-provider azure")
        return

    b = pd.read_csv(baseline_path)
    a = pd.read_csv(ablation_path)
    r = pd.read_csv(grounded_path)

    lines = ["# Ablation Comparison: baseline vs. ablation (fields, no retrieval) vs. grounded", ""]
    lines.append(f"N: baseline={len(b)}, ablation={len(a)}, grounded={len(r)}")
    lines.append("")

    lines.append("## Role-alignment index means")
    lines.append(f"- baseline: {b['role_alignment_index'].mean():.2f}")
    lines.append(f"- ablation (fields, no retrieval): {a['role_alignment_index'].mean():.2f}")
    lines.append(f"- grounded (retrieval + fields): {r['role_alignment_index'].mean():.2f}")
    lines.append("")

    t_ba, p_ba = stats.ttest_ind(a["role_alignment_index"], b["role_alignment_index"], equal_var=False)
    t_ar, p_ar = stats.ttest_ind(r["role_alignment_index"], a["role_alignment_index"], equal_var=False)
    lines.append(f"Welch's t-test, ablation vs baseline (effect of fields/instructions alone): t={t_ba:.3f}, p={p_ba:.4g}")
    lines.append(f"Welch's t-test, grounded vs ablation (effect of adding retrieval, holding fields constant): t={t_ar:.3f}, p={p_ar:.4g}")
    lines.append("")

    lines.append("## Bias-related keyword coverage (%) — same categories and methodology as Table 4-3")
    cov_b = coverage_by_category(b)
    cov_a = coverage_by_category(a)
    cov_r = coverage_by_category(r)
    for cat in cov_b:
        lines.append(f"- {cat}: baseline={cov_b[cat]:.1f}%, ablation={cov_a.get(cat, float('nan')):.1f}%, grounded={cov_r.get(cat, float('nan')):.1f}%")

    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "If ablation coverage is close to grounded coverage (both far above baseline), the coverage increase is "
        "attributable mainly to the output schema/instructions, not to retrieval specifically. If ablation coverage "
        "is close to baseline (both far below grounded), retrieval is doing the work. A result in between indicates "
        "both factors contribute."
    )

    report_path = OUT_DIR / "ablation_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()