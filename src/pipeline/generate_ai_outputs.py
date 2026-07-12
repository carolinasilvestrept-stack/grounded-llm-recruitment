from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


PROJECT_DIR = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = PROJECT_DIR / "src" / "generation" / "run_baseline_llm.py"
RAG_SCRIPT = PROJECT_DIR / "src" / "generation" / "run_rag_system.py"
COMPARISON_SCRIPT = PROJECT_DIR / "src" / "analysis" / "compare_ai_outputs.py"


def run_command(command):
    print("Running:", " ".join(str(part) for part in command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_DIR)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def resolve_api_provider(api_provider: str) -> str:
    if api_provider != "auto":
        return api_provider
    azure_ready = all(os.getenv(name) for name in [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
    ])
    if azure_ready:
        return "azure"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise SystemExit(
        "No API credentials found. Set Azure variables "
        "(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION) "
        "or OPENAI_API_KEY."
    )


def resolve_model(model: str | None, api_provider: str) -> str:
    if model:
        return model
    if api_provider == "azure":
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise SystemExit("AZURE_OPENAI_DEPLOYMENT is not set. Example: gpt-5.2")
        return deployment
    return "gpt-4.1-mini"


def build_common_args(args):
    common = []
    if args.limit is not None:
        common.extend(["--limit", str(args.limit)])
    common.extend(["--runs", str(args.runs)])
    if args.use_sample:
        common.append("--use-sample")
    if args.dry_run:
        common.append("--dry-run")
    if args.resume:
        common.append("--resume")
    return common


def main(args):
    if not args.dry_run and args.embedding_provider == "hash":
        raise SystemExit(
            "Hash embeddings are for dry-run testing only. "
            "Use --embedding-provider sentence-transformers or --embedding-provider openai for final outputs."
        )
    if args.dry_run:
        args.api_provider = "openai" if args.api_provider == "auto" else args.api_provider
        args.model = args.model or "dry-run-model"
        if args.baseline_prefix == "baseline_outputs":
            args.baseline_prefix = "dry_run_baseline_outputs"
        if args.rag_prefix == "rag_outputs":
            args.rag_prefix = "dry_run_rag_outputs"
    else:
        args.api_provider = resolve_api_provider(args.api_provider)
        args.model = resolve_model(args.model, args.api_provider)
    print(f"Using API provider: {args.api_provider}")
    print(f"Using model/deployment: {args.model}")
    common = build_common_args(args)

    baseline_command = [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--model",
        args.model,
        "--api-provider",
        args.api_provider,
        "--output-prefix",
        args.baseline_prefix,
        *common,
    ]
    rag_command = [
        sys.executable,
        str(RAG_SCRIPT),
        "--model",
        args.model,
        "--api-provider",
        args.api_provider,
        "--embedding-provider",
        args.embedding_provider,
        "--output-prefix",
        args.rag_prefix,
        *common,
    ]
    if args.rebuild_rag_index:
        rag_command.append("--rebuild-index")

    run_command(baseline_command)
    run_command(rag_command)

    analysis_prefix = "dry_run_" if args.dry_run else ""
    analysis_command = [
        sys.executable,
        str(COMPARISON_SCRIPT),
        "--baseline-output",
        str(PROJECT_DIR / "outputs" / "ai" / f"{args.baseline_prefix}.csv"),
        "--rag-output",
        str(PROJECT_DIR / "outputs" / "ai" / f"{args.rag_prefix}.csv"),
        "--output-prefix",
        analysis_prefix,
    ]
    run_command(analysis_command)


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline and RAG AI experiment, then analyze outputs.")
    parser.add_argument("--model", default=None, help="Model name for OpenAI or deployment name for Azure. Defaults to AZURE_OPENAI_DEPLOYMENT when using Azure.")
    parser.add_argument("--api-provider", choices=["auto", "openai", "azure"], default="auto")
    parser.add_argument(
        "--embedding-provider",
        choices=["hash", "sentence-transformers", "openai"],
        default="hash",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-sample", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the output CSVs.")
    parser.add_argument("--rebuild-rag-index", action="store_true")
    parser.add_argument("--baseline-prefix", default="baseline_outputs")
    parser.add_argument("--rag-prefix", default="rag_outputs")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
