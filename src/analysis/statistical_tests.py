from pathlib import Path
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = PROJECT_DIR / "outputs" / "ai"
OUT_DIR = PROJECT_DIR / "outputs" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    baseline_path = AI_DIR / "baseline_outputs.csv"
    rag_path = AI_DIR / "rag_outputs.csv"
    if not baseline_path.exists() or not rag_path.exists():
        raise FileNotFoundError("Run generate_ai_outputs.py first.")

    df = pd.concat([pd.read_csv(baseline_path), pd.read_csv(rag_path)], ignore_index=True)
    metric = "role_alignment_index" if "role_alignment_index" in df.columns else "supportiveness_index"
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    rows = []
    systems = list(df["system_condition"].dropna().unique())
    if len(systems) == 2:
        a = df[df["system_condition"] == systems[0]][metric].dropna()
        b = df[df["system_condition"] == systems[1]][metric].dropna()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"test":"system_condition_welch_ttest","group_a":systems[0],"group_b":systems[1],
                     "mean_a":a.mean(),"mean_b":b.mean(),"t_statistic":t,"p_value":p})

        keys = [
            "variant_id", "base_candidate_id", "job_id", "job_title",
            "gender_condition", "ethnicity_condition", "age_condition", "run_id",
        ]
        keys = [key for key in keys if key in df.columns]
        wide = df.pivot_table(index=keys, columns="system_condition", values=metric, aggfunc="mean").dropna()
        if all(system in wide.columns for system in systems) and len(wide) >= 2:
            t, p = stats.ttest_rel(wide[systems[0]], wide[systems[1]])
            diff = wide[systems[1]] - wide[systems[0]]
            rows.append({
                "test": "system_condition_paired_ttest",
                "group_a": systems[0],
                "group_b": systems[1],
                "mean_a": wide[systems[0]].mean(),
                "mean_b": wide[systems[1]].mean(),
                "mean_difference_b_minus_a": diff.mean(),
                "t_statistic": t,
                "p_value": p,
            })

    for system, sdf in df.groupby("system_condition"):
        for dim in ["gender_condition", "ethnicity_condition", "age_condition"]:
            groups = [(label, g[metric].dropna()) for label, g in sdf.groupby(dim)]
            if len(groups) == 2:
                t, p = stats.ttest_ind(groups[0][1], groups[1][1], equal_var=False)
                rows.append({"test":f"{system}_{dim}_welch_ttest","group_a":groups[0][0],"group_b":groups[1][0],
                             "mean_a":groups[0][1].mean(),"mean_b":groups[1][1].mean(),
                             "t_statistic":t,"p_value":p})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "statistical_tests_ai_outputs.csv", index=False)
    print(f"Metric used: {metric}")
    print(out)

if __name__ == "__main__":
    main()
