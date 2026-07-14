"""
Backfills the bias_awareness_notes and fairness_guidance columns into
outputs/ai/ablation_fields_no_retrieval_outputs.csv (and the matching
.json/.jsonl), which build_output_record() in run_baseline_llm.py never
wrote to the ablation output even though the ablation prompt explicitly
asks the model for these two fields and the model's raw_response does
contain them. run_baseline_llm.py's output-record schema was written for
the plain baseline condition, which never needed these fields, and was
reused unmodified for the ablation condition.

Without this fix, bias_category_coverage.py's combined_text() silently
defaulted these two fields to "" for every ablation row (row.get(field, "")
does not error on a missing column), so the ablation bias-keyword coverage
percentages reported so far excluded exactly the two fields most likely to
contain bias-awareness language. This does not require any new API calls --
the data was always present in raw_response, just never extracted into its
own columns.

Usage:
    python src\\analysis\\backfill_ablation_bias_fields.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = PROJECT_DIR / "outputs" / "ai"


def serialize(value) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value) if value is not None else ""


def extract_field(raw_response, field: str) -> str:
    if pd.isna(raw_response):
        return ""
    try:
        parsed = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return ""
    return serialize(parsed.get(field, ""))


def main() -> None:
    csv_path = AI_DIR / "ablation_fields_no_retrieval_outputs.csv"
    json_path = AI_DIR / "ablation_fields_no_retrieval_outputs.json"
    jsonl_path = AI_DIR / "ablation_fields_no_retrieval_outputs.jsonl"

    if not csv_path.exists():
        print(f"Missing {csv_path}.")
        return

    df = pd.read_csv(csv_path)
    if "bias_awareness_notes" in df.columns and "fairness_guidance" in df.columns:
        print("bias_awareness_notes and fairness_guidance already present. Nothing to do.")
        return

    missing_raw = df["raw_response"].isna().sum()
    if missing_raw:
        print(f"Warning: {missing_raw} rows have no raw_response and will get empty backfilled fields.")

    df["bias_awareness_notes"] = df["raw_response"].apply(lambda r: extract_field(r, "bias_awareness_notes"))
    df["fairness_guidance"] = df["raw_response"].apply(lambda r: extract_field(r, "fairness_guidance"))

    empty_notes = (df["bias_awareness_notes"] == "").sum()
    empty_guidance = (df["fairness_guidance"] == "").sum()
    if empty_notes or empty_guidance:
        print(f"Warning: {empty_notes} rows ended up with empty bias_awareness_notes, {empty_guidance} with empty fairness_guidance.")

    df.to_csv(csv_path, index=False)
    records = df.to_dict(orient="records")
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    print(f"Backfilled bias_awareness_notes and fairness_guidance into {len(df)} rows.")
    print(f"Updated: {csv_path}")
    print(f"Updated: {json_path}")
    print(f"Updated: {jsonl_path}")


if __name__ == "__main__":
    main()
