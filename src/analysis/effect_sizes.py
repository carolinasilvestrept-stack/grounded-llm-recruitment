from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = PROJECT_DIR / "outputs" / "ai"
OUT_DIR = PROJECT_DIR / "outputs" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def cohen_d(a, b):
    a = np.asarray(a.dropna(), dtype=float)
    b = np.asarray(b.dropna(), dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled

def cohen_d_unequal_var(a, b):
    """Effect size consistent with Welch's t-test, which does not assume
    equal variances. Uses the unweighted average of the two variances
    (Delacre, Lakens & Leys, 2017) rather than the pooled-SD formula, which
    implicitly assumes equal variances -- the assumption Welch's test exists
    to avoid."""
    a = np.asarray(a.dropna(), dtype=float)
    b = np.asarray(b.dropna(), dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    avg_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if avg_sd == 0:
        return np.nan
    return (a.mean() - b.mean()) / avg_sd

def ci_mean(series, confidence=0.95):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 2:
        return (np.nan, np.nan)
    mean = x.mean()
    sem = stats.sem(x)
    margin = sem * stats.t.ppf((1 + confidence) / 2, len(x)-1)
    return mean - margin, mean + margin

def main():
    baseline = pd.read_csv(AI_DIR / "baseline_outputs.csv")
    rag = pd.read_csv(AI_DIR / "rag_outputs.csv")
    df = pd.concat([baseline, rag], ignore_index=True)
    metrics = [m for m in ["role_alignment_index", "supportiveness_index"] if m in df.columns]

    rows = []
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

        for dim in ["system_condition", "gender_condition", "ethnicity_condition", "age_condition"]:
            groups = [(label, g[metric].dropna()) for label, g in df.groupby(dim)]
            if len(groups) == 2:
                a_label, a = groups[0]
                b_label, b = groups[1]
                t, p = stats.ttest_ind(a, b, equal_var=False)
                a_ci = ci_mean(a)
                b_ci = ci_mean(b)
                rows.append({
                    "metric": metric,
                    "comparison": dim,
                    "group_a": a_label,
                    "group_b": b_label,
                    "mean_a": a.mean(),
                    "mean_b": b.mean(),
                    "ci95_a_low": a_ci[0],
                    "ci95_a_high": a_ci[1],
                    "ci95_b_low": b_ci[0],
                    "ci95_b_high": b_ci[1],
                    "cohen_d": cohen_d(a, b),
                    "cohen_d_unequal_var": cohen_d_unequal_var(a, b),
                    "t_statistic": t,
                    "p_value": p,
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "effect_sizes_ai_outputs.csv", index=False)
    print(f"Saved: {OUT_DIR / 'effect_sizes_ai_outputs.csv'}")
    print(out)

if __name__ == "__main__":
    main()
