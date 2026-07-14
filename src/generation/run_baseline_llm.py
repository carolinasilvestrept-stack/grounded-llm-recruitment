from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)


def find_project_dir() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "data").exists() or (candidate / "prompts").exists() or (candidate / "baseline_prompt.md").exists():
            return candidate
    return here.parent


PROJECT_DIR = find_project_dir()
PROMPT_PATH = PROJECT_DIR / "prompts" / "baseline_prompt.md"
if not PROMPT_PATH.exists():
    PROMPT_PATH = PROJECT_DIR / "baseline_prompt.md"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ai"
DEFAULT_CANDIDATES_PATH = PROJECT_DIR / "data" / "processed" / "controlled_candidate_profiles.csv"
DEFAULT_JOBS_PATH = PROJECT_DIR / "data" / "processed" / "job_descriptions.csv"
SAMPLE_CANDIDATES_PATH = PROJECT_DIR / "data" / "sample" / "controlled_candidate_profiles_sample.csv"
SAMPLE_JOBS_PATH = PROJECT_DIR / "data" / "sample" / "job_descriptions_sample.csv"



def load_prompt(prompt_path: Path = None) -> str:
    return (prompt_path or PROMPT_PATH).read_text(encoding="utf-8")


def require_columns(frame: pd.DataFrame, columns: List[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def load_inputs(candidates_path: Path, jobs_path: Path, limit: Optional[int]) -> pd.DataFrame:
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidate profiles not found at {candidates_path}.")
    if not jobs_path.exists():
        raise FileNotFoundError(f"Job descriptions not found at {jobs_path}.")
    candidates = pd.read_csv(candidates_path)
    jobs = pd.read_csv(jobs_path)
    require_columns(candidates, ["variant_id", "base_candidate_id", "gender_condition", "ethnicity_condition", "age_condition", "controlled_resume_text"], candidates_path)
    require_columns(jobs, ["job_id", "job_title", "role_summary", "main_responsibilities", "required_qualifications", "required_skills", "preferred_skills", "evaluation_criteria"], jobs_path)
    candidates = candidates.head(limit) if limit else candidates
    candidates["_join_key"] = 1
    jobs["_join_key"] = 1
    return candidates.merge(jobs, on="_join_key").drop(columns=["_join_key"]).reset_index(drop=True)


def format_job_description(row: pd.Series) -> str:
    return (
        f"Job Title: {row['job_title']}\n"
        f"Role Summary: {row['role_summary']}\n"
        f"Main Responsibilities: {row['main_responsibilities']}\n"
        f"Required Qualifications: {row['required_qualifications']}\n"
        f"Required Skills: {row['required_skills']}\n"
        f"Preferred Skills: {row['preferred_skills']}\n"
        f"Evaluation Criteria: {row['evaluation_criteria']}"
    )


def build_user_message(row: pd.Series) -> str:
    return (
        "Provide AI-supported candidate-job alignment. Do not make a hiring decision.\n\n"
        "Candidate Resume:\n"
        f"{row['controlled_resume_text']}\n\n"
        "Job Description:\n"
        f"{format_job_description(row)}"
    )


def bounded_number(value, default: float = 50.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return round(max(0.0, min(100.0, number)), 2)


def as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_role_alignment_indicators(value) -> Dict[str, float]:
    if not isinstance(value, dict):
        value = {}
    return {
        "technical_match": bounded_number(value.get("technical_match", value.get("technical_or_role_skills", 50))),
        "experience_relevance": bounded_number(value.get("experience_relevance", 50)),
        "communication_evidence": bounded_number(value.get("communication_evidence", value.get("communication_or_collaboration", 50))),
        "education_relevance": bounded_number(value.get("education_relevance", value.get("education_or_training", 50))),
    }


def role_alignment_index(indicators: Dict[str, float]) -> float:
    if not isinstance(indicators, dict) or not indicators:
        return 50.0
    values = [bounded_number(v, 50.0) for v in indicators.values()]
    return round(sum(values) / len(values), 2)


def supportiveness_index(result: Dict[str, object]) -> float:
    strengths = " ".join(str(x) for x in result.get("job_relevant_strengths", []))
    concerns = " ".join(str(x) for x in result.get("job_relevant_concerns", []))
    observations = " ".join(str(v) for v in result.get("structured_observations", {}).values())
    text_blob = f"{strengths} {concerns} {observations}".lower()

    supportive_terms = ["strong", "relevant", "aligned", "demonstrates", "experience", "evidence", "matches", "supports"]
    concern_terms = ["limited", "gap", "unclear", "not enough", "missing", "uncertain", "weak", "insufficient"]

    supportive_count = sum(text_blob.count(term) for term in supportive_terms)
    concern_count = sum(text_blob.count(term) for term in concern_terms)
    total = supportive_count + concern_count

    if total == 0:
        return 50.0

    return round((supportive_count / total) * 100, 2)

def normalize_result(data: Dict[str, object]) -> Dict[str, object]:
    structured_observations = data.get("structured_observations", data.get("qualification_alignment", {}))
    if not isinstance(structured_observations, dict):
        structured_observations = {}

    indicators = normalize_role_alignment_indicators(data.get("role_alignment_indicators", {}))

    result = {
        "candidate_summary": str(data.get("candidate_summary", data.get("explanation", ""))).strip(),
        "job_relevant_strengths": as_list(data.get("job_relevant_strengths", data.get("relevant_evidence", []))),
        "job_relevant_concerns": as_list(data.get("job_relevant_concerns", data.get("development_areas", []))),
        "structured_observations": structured_observations,
        "role_alignment_indicators": indicators,
        "supporting_evidence": as_list(data.get("supporting_evidence", data.get("evidence_to_check", []))),
        "limits_of_assessment": as_list(data.get("limits_of_assessment", [])),
    }
    result["supportiveness_index"] = supportiveness_index(result)
    result["role_alignment_index"] = role_alignment_index(indicators)
    return result

def parse_json_response(text: str) -> Dict[str, object]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return normalize_result(json.loads(text))


def dry_run_evaluation(row: pd.Series) -> Dict[str, object]:
    result = {
        "candidate_summary": "Dry-run baseline decision-support summary of role-relevant qualifications.",
        "job_relevant_strengths": ["Dry-run role-relevant strength based on available resume evidence."],
        "job_relevant_concerns": ["Dry-run assessment limitation or potential evidence gap."],
        "structured_observations": {
            "technical_or_role_skills": "Evidence should be reviewed against the role requirements.",
            "experience_relevance": "Experience appears partially relevant based on available information.",
            "communication_or_collaboration": "Not enough evidence to assess fully.",
            "education_or_training": "Education/training evidence should be interpreted only where job-relevant."
        },
        "role_alignment_indicators": {
            "technical_match": 65,
            "experience_relevance": 60,
            "communication_evidence": 45,
            "education_relevance": 55
        },
        "supporting_evidence": ["Dry-run evidence item based on resume/job overlap."],
        "limits_of_assessment": ["Dry-run output cannot verify all resume claims."],
        "raw_response": "",
    }
    result["supportiveness_index"] = supportiveness_index(result)
    result["role_alignment_index"] = role_alignment_index(result["role_alignment_indicators"])
    return result

def resolve_api_provider(api_provider: str) -> str:
    """Resolve provider automatically from available environment variables."""
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
    raise RuntimeError(
        "No API credentials found. Set Azure variables "
        "(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION) "
        "or OPENAI_API_KEY."
    )


def resolve_model(model: str | None, api_provider: str) -> str:
    """For Azure, the model argument must be the Azure deployment name."""
    if model:
        return model
    if api_provider == "azure":
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is not set. Example: gpt-5.2")
        return deployment
    return "gpt-4.1-mini"


def create_openai_client(api_provider: str):
    api_provider = resolve_api_provider(api_provider)
    if api_provider == "azure":
        required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT"]
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"Missing Azure OpenAI environment variables: {', '.join(missing)}")
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    from openai import OpenAI
    return OpenAI()


def chat_completion_json(client, model: str, system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return response.choices[0].message.content


def call_openai(client, model: str, system_prompt: str, user_message: str, api_provider: str) -> Dict[str, object]:
    if api_provider == "azure":
        raw_text = chat_completion_json(client, model, system_prompt, user_message)
    else:
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                temperature=0,
            )
            raw_text = response.output_text
        except Exception:
            raw_text = chat_completion_json(client, model, system_prompt, user_message)
    parsed = parse_json_response(raw_text)
    parsed["raw_response"] = raw_text
    return parsed


def call_openai_with_retry(
    client,
    model: str,
    system_prompt: str,
    user_message: str,
    api_provider: str,
    max_retries: int = 5,
    base_sleep: float = 2.0,
) -> Tuple[Dict[str, object], int]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_openai(client, model, system_prompt, user_message, api_provider), attempt
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = any(
                marker in message
                for marker in [
                    "internal error",
                    "invalid_prompt",
                    "timeout",
                    "connection error",
                    "apiconnectionerror",
                    "connecterror",
                    "getaddrinfo",
                    "dns",
                    "network",
                    "temporarily",
                    "rate limit",
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "jsondecodeerror",
                    "unterminated string",
                    "expecting value",
                    "expecting property name",
                ]
            )
            if not retryable or attempt == max_retries:
                raise
            sleep_for = base_sleep * attempt
            print(f"API call failed on attempt {attempt}/{max_retries}; retrying in {sleep_for:.1f}s. Error: {exc}", flush=True)
            time.sleep(sleep_for)
    raise RuntimeError(f"API call failed after retries: {last_error}")


def test_api(model: str | None, api_provider: str) -> None:
    api_provider = resolve_api_provider(api_provider)
    model = resolve_model(model, api_provider)
    print(f"Testing API provider: {api_provider}")
    print(f"Testing model/deployment: {model}")
    client = create_openai_client(api_provider)
    result = call_openai(client, model, load_prompt(), "Return a minimal valid JSON candidate-job alignment output.", api_provider)
    print(json.dumps(result, indent=2))


def serialize(value) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value) if value is not None else ""


def build_output_record(row: pd.Series, result: Dict[str, object], model: str, run_id: int, attempts: int = 1, system_condition: str = "baseline_llm") -> Dict[str, object]:
    record = {
        "system_condition": system_condition,
        "model": model,
        "run_id": run_id,
        "generation_attempts": attempts,
        "variant_id": row["variant_id"],
        "base_candidate_id": row["base_candidate_id"],
        "job_id": row["job_id"],
        "job_title": row["job_title"],
        "gender_condition": row["gender_condition"],
        "ethnicity_condition": row["ethnicity_condition"],
        "age_condition": row["age_condition"],
        "candidate_summary": result["candidate_summary"],
        "job_relevant_strengths": serialize(result["job_relevant_strengths"]),
        "job_relevant_concerns": serialize(result["job_relevant_concerns"]),
        "structured_observations": serialize(result["structured_observations"]),
        "role_alignment_indicators": serialize(result["role_alignment_indicators"]),
        "supporting_evidence": serialize(result["supporting_evidence"]),
        "limits_of_assessment": serialize(result["limits_of_assessment"]),
        "supportiveness_index": result["supportiveness_index"],
        "role_alignment_index": result["role_alignment_index"],
        "raw_response": result.get("raw_response", ""),
    }
    # Present only when the prompt requests them (e.g. the grounded prompt,
    # or the ablation_fields_no_retrieval prompt); harmless no-op for the
    # standard baseline prompt, which does not include these fields.
    if "bias_awareness_notes" in result:
        record["bias_awareness_notes"] = serialize(result["bias_awareness_notes"])
    if "fairness_guidance" in result:
        record["fairness_guidance"] = serialize(result["fairness_guidance"])
    return record

def save_outputs(records: List[Dict[str, object]], output_prefix: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{output_prefix}.csv"
    json_path = OUTPUT_DIR / f"{output_prefix}.json"
    jsonl_path = OUTPUT_DIR / f"{output_prefix}.jsonl"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    print(f"CSV output: {csv_path}")
    print(f"JSON output: {json_path}")
    print(f"JSONL output: {jsonl_path}")


def output_paths(output_prefix: str) -> Tuple[Path, Path, Path]:
    return (
        OUTPUT_DIR / f"{output_prefix}.csv",
        OUTPUT_DIR / f"{output_prefix}.json",
        OUTPUT_DIR / f"{output_prefix}.jsonl",
    )


def write_outputs(records: List[Dict[str, object]], output_prefix: str, announce: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path, json_path, jsonl_path = output_paths(output_prefix)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    if announce:
        print(f"CSV output: {csv_path}")
        print(f"JSON output: {json_path}")
        print(f"JSONL output: {jsonl_path}")


def run_key(row: pd.Series, run_id: int) -> tuple[str, str, int]:
    return str(row["variant_id"]), str(row["job_id"]), int(run_id)


def load_existing_records(output_prefix: str) -> List[Dict[str, object]]:
    csv_path, _, _ = output_paths(output_prefix)
    if not csv_path.exists():
        return []
    existing = pd.read_csv(csv_path)
    required = {"variant_id", "job_id", "run_id"}
    if not required.issubset(existing.columns):
        raise SystemExit(f"Cannot resume from {csv_path}: missing one of {sorted(required)}")
    return existing.to_dict(orient="records")


def run(args: argparse.Namespace) -> None:
    if args.test_api:
        test_api(args.model, args.api_provider)
        return
    candidates_path = SAMPLE_CANDIDATES_PATH if args.use_sample else Path(args.candidates)
    jobs_path = SAMPLE_JOBS_PATH if args.use_sample else Path(args.jobs)
    system_prompt = load_prompt(Path(args.prompt_file) if args.prompt_file else None)
    inputs = load_inputs(candidates_path, jobs_path, args.limit)
    args.api_provider = resolve_api_provider(args.api_provider)
    args.model = resolve_model(args.model, args.api_provider)
    client = None if args.dry_run else create_openai_client(args.api_provider)
    records = load_existing_records(args.output_prefix) if args.resume else []
    completed_keys = {
        (str(record["variant_id"]), str(record["job_id"]), int(record["run_id"]))
        for record in records
    }
    if args.resume and records:
        print(f"Resuming from {len(records)} existing records in {output_paths(args.output_prefix)[0]}")
    total = len(inputs) * args.runs
    completed = len(completed_keys)
    for run_id in range(1, args.runs + 1):
        for _, row in inputs.iterrows():
            key = run_key(row, run_id)
            if key in completed_keys:
                continue
            if args.dry_run:
                result = dry_run_evaluation(row)
                attempts = 1
            else:
                result, attempts = call_openai_with_retry(client, args.model, system_prompt, build_user_message(row), args.api_provider)
                if args.sleep_seconds:
                    time.sleep(args.sleep_seconds)
            records.append(build_output_record(row, result, args.model, run_id, attempts, args.system_condition))
            completed_keys.add(key)
            completed += 1
            print(f"[{completed}/{total}] run {run_id} {row['variant_id']} / {row['job_id']} -> role_alignment_index={result['role_alignment_index']}")
            if len(records) % args.checkpoint_every == 0:
                write_outputs(records, args.output_prefix)
                print(f"Checkpoint saved: {len(records)} records", flush=True)
    write_outputs(records, args.output_prefix, announce=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline LLM recruitment decision-support outputs.")
    parser.add_argument("--model", default=None, help="Model name for OpenAI or deployment name for Azure. Defaults to AZURE_OPENAI_DEPLOYMENT when using Azure.")
    parser.add_argument("--api-provider", choices=["auto", "openai", "azure"], default="auto")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-prefix", default="baseline_outputs")
    parser.add_argument("--prompt-file", default=None, help="Override path to the system prompt file (used for ablation conditions).")
    parser.add_argument("--system-condition", default="baseline_llm", help="Label written to the system_condition column (override for ablation conditions).")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the output CSV.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-sample", action="store_true")
    parser.add_argument("--test-api", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
