"""
Ragas evaluation of the grounded RAG condition: Faithfulness and Context
Utilization, computed on a random sample of records from rag_outputs.csv.

This is a standalone analysis step, run AFTER the main generation pipeline
has produced a complete rag_outputs.csv. It does not touch baseline_outputs.csv
(Faithfulness/Context Utilization require retrieved context, which the
baseline condition does not have).

Usage (from the project root, after activating .venv):

    python src\\analysis\\ragas_evaluation.py --sample-size 150

Requires: pip install ragas litellm
Requires your .env to already have the Azure OpenAI variables set
(the same ones used for generation).

Cost note: this makes 2-4 additional LLM calls per sampled record
(one per metric, per claim/chunk it checks). A sample of 150 records
is a reasonable, defensible size for a thesis-level evaluation; the
full 1,800 would be considerably more expensive and slow for limited
extra insight.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

# ragas (as of 0.3.x/0.4.x) unconditionally imports langchain_community.chat_models.vertexai,
# a submodule that langchain-community removed during its "sunset" split into standalone
# integration packages. We never use Vertex AI (Azure via litellm only), so this stub just
# satisfies the import without pulling in a real (and version-incompatible) langchain-community.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # unused placeholder, only needed to satisfy ragas's import
        pass

    _vertexai_stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAG_OUTPUT = PROJECT_DIR / "outputs" / "ai" / "rag_outputs.csv"
CANDIDATES_PATH = PROJECT_DIR / "data" / "processed" / "controlled_candidate_profiles.csv"
JOBS_PATH = PROJECT_DIR / "data" / "processed" / "job_descriptions.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "analysis"


def format_job_description(row: pd.Series) -> str:
    """Matches format_job_description() in src/generation/run_rag_system.py."""
    return (
        f"Job Title: {row['job_title']}\nRole Summary: {row['role_summary']}\n"
        f"Main Responsibilities: {row['main_responsibilities']}\nRequired Qualifications: {row['required_qualifications']}\n"
        f"Required Skills: {row['required_skills']}\nPreferred Skills: {row['preferred_skills']}\nEvaluation Criteria: {row['evaluation_criteria']}"
    )


def require_env(names: list[str]) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        print("Set these in your .env file (same ones used for generation) before running this script.")
        sys.exit(1)


def build_llm():
    """Build the Ragas judge LLM from the same Azure OpenAI deployment used for generation."""
    require_env(["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT"])
    import litellm
    from ragas.llms import llm_factory

    litellm.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
    litellm.api_key = os.environ["AZURE_OPENAI_API_KEY"]
    litellm.api_version = os.environ["AZURE_OPENAI_API_VERSION"]

    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    return llm_factory(
        f"azure/{deployment}",
        provider="litellm",
        client=litellm.acompletion,
        max_tokens=4096,
    )


def build_question(row: pd.Series) -> str:
    """Reconstruct the exact retrieval query text the pipeline used at generation
    time: build_retrieval_query() in run_rag_system.py. Uses core_resume_text
    (job-relevant qualifications only, no name or demographic-cue labels), matching
    the fixed retrieval query, so that Faithfulness/Context Utilization are scored
    against the same text that was actually used to search the vector store.
    Requires the row to be pre-merged with controlled_candidate_profiles.csv and
    job_descriptions.csv, since rag_outputs.csv itself only carries job_title, not
    the full job description or resume text."""
    if pd.isna(row.get("core_resume_text")) or pd.isna(row.get("role_summary")):
        return "Evaluate this candidate for this role."
    return f"Fair structured hiring guidance. Job: {format_job_description(row)} Candidate: {row['core_resume_text']}"


def _join_fields(row: pd.Series, columns: list[str]) -> str:
    """Concatenate a set of (possibly JSON-encoded list) output columns into one string."""
    pieces = []
    for col in columns:
        if col not in row or pd.isna(row[col]):
            continue
        value = row[col]
        if isinstance(value, str) and value.strip().startswith(("[", "{")):
            try:
                parsed = json.loads(value)
                value = "; ".join(str(v) for v in parsed) if isinstance(parsed, list) else json.dumps(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
        pieces.append(str(value))
    return "\n".join(pieces)


def build_answer(row: pd.Series) -> str:
    """Assemble one 'answer' string from the model's structured JSON output fields, for
    Context Utilization. Only the evidence-bearing fields are included, since those are
    the claims that should be traceable back to the retrieved context."""
    return _join_fields(row, ["candidate_summary", "job_relevant_strengths", "job_relevant_concerns", "supporting_evidence"])


def build_faithfulness_answer(row: pd.Series) -> str:
    """Assemble the 'answer' used for Faithfulness specifically. Faithfulness checks
    whether response claims are grounded in the retrieved context, and the retrieved
    context here is fairness/hiring-criteria/counterfactual policy guidance, not
    candidate facts. bias_awareness_notes and fairness_guidance are the only output
    fields that are actually meant to draw from that policy guidance, so they are the
    only fields where Faithfulness is a meaningful signal; candidate-facts fields
    (candidate_summary, job_relevant_strengths, etc.) come from the resume, not
    retrieval, and would drag the score toward zero regardless of RAG quality."""
    return _join_fields(row, ["bias_awareness_notes", "fairness_guidance"])


def build_contexts(row: pd.Series) -> list[str]:
    """Extract the actual retrieved chunk text already saved per row."""
    raw = row.get("retrieved_context_json")
    if pd.isna(raw):
        return []
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [item.get("text", "") for item in items if item.get("text")]


def main(args: argparse.Namespace) -> None:
    rag_output = Path(args.rag_output) if args.rag_output else RAG_OUTPUT
    if not rag_output.exists():
        print(f"Missing {rag_output}. Run the generation pipeline first.")
        sys.exit(1)
    for path in (CANDIDATES_PATH, JOBS_PATH):
        if not path.exists():
            print(f"Missing {path}. Cannot reconstruct the original query text.")
            sys.exit(1)

    from ragas.metrics.collections import ContextUtilization, Faithfulness

    df = pd.read_csv(rag_output)
    candidates = pd.read_csv(CANDIDATES_PATH)[["variant_id", "core_resume_text"]]
    jobs = pd.read_csv(JOBS_PATH)
    df = df.merge(candidates, on="variant_id", how="left").merge(jobs, on="job_id", how="left", suffixes=("", "_job"))
    print(f"Loaded {len(df)} grounded RAG records.")

    random.seed(args.seed)
    sample_size = min(args.sample_size, len(df))
    sample = df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
    print(f"Evaluating a random sample of {sample_size} records (seed={args.seed}).")

    llm = build_llm()
    faithfulness_scorer = Faithfulness(llm=llm)
    context_scorer = ContextUtilization(llm=llm)

    results = []
    for i, row in sample.iterrows():
        question = build_question(row)
        answer = build_answer(row)
        faithfulness_answer = build_faithfulness_answer(row)
        contexts = build_contexts(row)

        record = {
            "variant_id": row.get("variant_id"),
            "job_id": row.get("job_id"),
            "run_id": row.get("run_id"),
            "num_contexts": len(contexts),
        }

        if not contexts:
            record["faithfulness"] = None
            record["context_utilization"] = None
            record["error"] = "empty contexts"
        else:
            if not faithfulness_answer.strip():
                record["faithfulness"] = None
                record["error"] = "empty faithfulness answer (no bias_awareness_notes/fairness_guidance)"
            else:
                try:
                    faithfulness_result = faithfulness_scorer.score(
                        user_input=question, response=faithfulness_answer, retrieved_contexts=contexts
                    )
                    record["faithfulness"] = faithfulness_result.value
                except Exception as exc:
                    record["faithfulness"] = None
                    record["error"] = f"faithfulness: {type(exc).__name__}: {exc}"

            if not answer.strip():
                record["context_utilization"] = None
                record.setdefault("error", "empty context_utilization answer")
            else:
                try:
                    context_result = context_scorer.score(
                        user_input=question, response=answer, retrieved_contexts=contexts
                    )
                    record["context_utilization"] = context_result.value
                except Exception as exc:
                    record["context_utilization"] = None
                    record.setdefault("error", f"context_utilization: {type(exc).__name__}: {exc}")

        results.append(record)
        print(f"[{i + 1}/{sample_size}] faithfulness={record.get('faithfulness')} "
              f"context_utilization={record.get('context_utilization')}", flush=True)

    stem = rag_output.stem
    tag = stem[len("rag_outputs"):].lstrip("_") if stem.startswith("rag_outputs") else stem
    suffix = f"_{tag}" if tag else ""

    results_df = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_case_path = OUT_DIR / f"ragas_evaluation_by_case{suffix}.csv"
    results_df.to_csv(by_case_path, index=False)

    valid = results_df.dropna(subset=["faithfulness", "context_utilization"], how="all")
    n_failed = len(results_df) - len(valid)

    summary_lines = [
        "# Ragas Evaluation Summary (Grounded RAG condition)",
        "",
        f"Sample size: {sample_size} of {len(df)} total grounded records (seed={args.seed}).",
        f"Records with a valid score: {len(valid)}. Records that failed or had empty content: {n_failed}.",
        "",
        "Faithfulness is computed against bias_awareness_notes + fairness_guidance only "
        "(the output fields meant to draw from the retrieved fairness/hiring-criteria/"
        "counterfactual policy documents). Context Utilization is computed against the "
        "full candidate-evaluation answer (candidate_summary, job_relevant_strengths, "
        "job_relevant_concerns, supporting_evidence), since it checks retrieval relevance "
        "to the query rather than groundedness of specific claims.",
        "",
    ]
    for metric in ["faithfulness", "context_utilization"]:
        series = pd.to_numeric(results_df[metric], errors="coerce").dropna()
        if len(series) == 0:
            summary_lines.append(f"- {metric}: no valid scores")
            continue
        summary_lines.append(
            f"- {metric}: mean={series.mean():.3f}, median={series.median():.3f}, "
            f"min={series.min():.3f}, max={series.max():.3f}, n={len(series)}"
        )

    summary_path = OUT_DIR / f"ragas_evaluation_summary{suffix}.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(summary_lines))
    print(f"\nSaved: {by_case_path}")
    print(f"Saved: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ragas Faithfulness and Context Utilization on a sample of the grounded RAG condition.")
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rag-output", default=None, help="Path to the grounded RAG output CSV to evaluate. Defaults to outputs/ai/rag_outputs.csv.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
