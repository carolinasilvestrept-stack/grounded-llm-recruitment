from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "outputs" / "analysis"
AI_DIR = PROJECT_DIR / "outputs" / "ai"

COMBINED = OUT_DIR / "combined_ai_outputs.csv"
BASELINE = AI_DIR / "baseline_outputs.csv"
RAG = AI_DIR / "rag_outputs.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_COLS = [
    ["system_condition"],
    ["system_condition", "gender_condition"],
    ["system_condition", "ethnicity_condition"],
    ["system_condition", "age_condition"],
    ["system_condition", "gender_condition", "ethnicity_condition", "age_condition"],
]


def load_outputs():
    if BASELINE.exists() and RAG.exists():
        baseline = pd.read_csv(BASELINE)
        rag = pd.read_csv(RAG)
        return pd.concat([baseline, rag], ignore_index=True)

    if COMBINED.exists():
        return pd.read_csv(COMBINED)

    raise FileNotFoundError("Missing AI outputs. Run generate_ai_outputs.py first.")


def metric_column(df):
    if "role_alignment_index" in df.columns:
        return "role_alignment_index"
    if "supportiveness_index" in df.columns:
        return "supportiveness_index"
    raise ValueError(f"No metric found. Available columns: {list(df.columns)}")


def disparity_within_system(df, metric):
    records = []
    dimensions = ["gender_condition", "ethnicity_condition", "age_condition"]

    for system, sdf in df.groupby("system_condition"):
        for dim in dimensions:
            means = sdf.groupby(dim)[metric].mean().dropna()
            if len(means) >= 2:
                records.append({
                    "system_condition": system,
                    "dimension": dim,
                    "max_group_mean": means.max(),
                    "min_group_mean": means.min(),
                    "disparity_range": means.max() - means.min(),
                    "group_means": means.to_dict(),
                })

        inter_cols = ["gender_condition", "ethnicity_condition", "age_condition"]
        means = sdf.groupby(inter_cols)[metric].mean().dropna()
        if len(means) >= 2:
            records.append({
                "system_condition": system,
                "dimension": "intersectional_gender_ethnicity_age",
                "max_group_mean": means.max(),
                "min_group_mean": means.min(),
                "disparity_range": means.max() - means.min(),
                "group_means": {str(k): v for k, v in means.to_dict().items()},
            })

    return pd.DataFrame(records)


def main():
    df = load_outputs()
    metric = metric_column(df)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    summaries = []
    for cols in GROUP_COLS:
        grouped = df.groupby(cols)[metric].agg(["count", "mean", "std", "min", "max"]).reset_index()
        grouped["grouping"] = " + ".join(cols)
        summaries.append(grouped)

    summary = pd.concat(summaries, ignore_index=True)
    disparity = disparity_within_system(df, metric)

    summary.to_csv(OUT_DIR / "enhanced_metric_group_summary.csv", index=False)
    disparity.to_csv(OUT_DIR / "enhanced_disparity_ranges.csv", index=False)

    print(f"Metric used: {metric}")
    print(f"Saved: {OUT_DIR / 'enhanced_metric_group_summary.csv'}")
    print(f"Saved: {OUT_DIR / 'enhanced_disparity_ranges.csv'}")
    print(disparity)


if __name__ == "__main__":
    main()