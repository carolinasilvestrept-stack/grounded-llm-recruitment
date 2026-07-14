"""
Exports a random sample of grounded-condition records for a properly
documented manual Faithfulness coding exercise (addressing the review point
that the current explanation of the low Ragas Faithfulness score rests on a
single, undocumented example).

For each sampled record, this produces one row per claim/statement in the
bias_awareness_notes and fairness_guidance fields, with blank columns for a
human coder to mark whether that specific claim is traceable to the
retrieved context, a general principle, or neither. Coding a modest sample
(e.g. 20-30 records) properly, with a clear procedure, is enough to support
a documented claim in the thesis, unlike the current single-example basis.

Usage:
    python src\\analysis\\ragas_manual_coding_sample.py --sample-size 25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = PROJECT_DIR / "outputs" / "ai"
OUT_DIR = PROJECT_DIR / "outputs" / "analysis"


def extract_claims(value) -> list[str]:
    """bias_awareness_notes / fairness_guidance are stored as serialized
    JSON lists; split into individual claims for per-claim coding."""
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        return [str(value)]


def get_context_text(row: pd.Series) -> str:
    raw = row.get("retrieved_context_json")
    if pd.isna(raw):
        return ""
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    return "\n---\n".join(item.get("text", "") for item in items if item.get("text"))


def main(args: argparse.Namespace) -> None:
    rag_path = AI_DIR / "rag_outputs.csv"
    if not rag_path.exists():
        print(f"Missing {rag_path}.")
        return

    df = pd.read_csv(rag_path)
    sample = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed).reset_index(drop=True)

    rows = []
    for _, row in sample.iterrows():
        context_text = get_context_text(row)
        for field in ["bias_awareness_notes", "fairness_guidance"]:
            for claim in extract_claims(row.get(field)):
                rows.append({
                    "variant_id": row.get("variant_id"),
                    "job_id": row.get("job_id"),
                    "run_id": row.get("run_id"),
                    "source_field": field,
                    "claim_text": claim,
                    "retrieved_context_text": context_text,
                    # Blank columns for the human coder to fill in:
                    "coder_verdict": "",  # expected values: "traceable", "general_principle_only", "not_traceable"
                    "coder_notes": "",
                })

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ragas_manual_coding_sample.csv"
    out.to_csv(out_path, index=False)

    print(f"Sampled {len(sample)} records, producing {len(out)} individual claims to code.")
    print(f"Saved: {out_path}")
    print()
    print("Coding instructions:")
    print("For each row, read claim_text and retrieved_context_text, then set coder_verdict to one of:")
    print("  - traceable: the claim's specific content is directly supported by the retrieved context")
    print("  - general_principle_only: the claim restates a general fairness principle from context,")
    print("    but adds job- or candidate-specific detail not present in the retrieved text")
    print("  - not_traceable: the claim is not supported by the retrieved context and does not")
    print("    appear to be a general-principle restatement either")
    print("Once coded, compute the proportion in each category; report this distribution instead of")
    print("a single-example description in the thesis.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a random sample for manual Ragas Faithfulness coding.")
    parser.add_argument("--sample-size", type=int, default=25, help="Number of records to sample (each may yield multiple claim rows).")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())