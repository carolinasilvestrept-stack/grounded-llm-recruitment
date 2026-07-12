from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAG_OUTPUT = PROJECT_DIR / "outputs" / "ai" / "rag_outputs.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_similarities(value):
    if pd.isna(value):
        return []
    similarities = []
    for item in str(value).split(";"):
        item = item.strip()
        if not item or item.lower() in {"mandatory", "nan"}:
            continue
        similarities.append(float(item))
    return similarities

def main():
    if not RAG_OUTPUT.exists():
        raise FileNotFoundError(f"Missing {RAG_OUTPUT}")

    df = pd.read_csv(RAG_OUTPUT)
    rows = []
    for _, row in df.iterrows():
        sims = parse_similarities(row.get("retrieved_similarities", ""))
        if not sims:
            continue
        rows.append({
            "variant_id": row.get("variant_id"),
            "job_id": row.get("job_id"),
            "top_similarity": max(sims),
            "mean_similarity": sum(sims) / len(sims),
            "min_similarity": min(sims),
            "semantic_retrieved_count": len(sims),
            "total_retrieved_count": len([x for x in str(row.get("retrieved_chunk_ids", "")).split(";") if x.strip()]),
            "retrieved_chunk_ids": row.get("retrieved_chunk_ids", "")
        })

    out = pd.DataFrame(rows)
    summary = out[["top_similarity", "mean_similarity", "min_similarity", "semantic_retrieved_count", "total_retrieved_count"]].describe()

    out.to_csv(OUT_DIR / "retrieval_similarity_by_case.csv", index=False)
    summary.to_csv(OUT_DIR / "retrieval_similarity_summary.csv")

    print(f"Saved: {OUT_DIR / 'retrieval_similarity_by_case.csv'}")
    print(f"Saved: {OUT_DIR / 'retrieval_similarity_summary.csv'}")
    print(summary)

if __name__ == "__main__":
    main()
