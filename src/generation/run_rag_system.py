import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)


def find_project_dir() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "data").exists() or (candidate / "knowledge_base").exists() or (candidate / "grounded_prompt.md").exists():
            return candidate
    return here.parent


PROJECT_DIR = find_project_dir()
KNOWLEDGE_BASE_DIR = PROJECT_DIR / "knowledge_base"
if not KNOWLEDGE_BASE_DIR.exists():
    KNOWLEDGE_BASE_DIR = PROJECT_DIR
CONFIG_DIR = PROJECT_DIR / "config"
PROMPT_PATH = PROJECT_DIR / "prompts" / "grounded_prompt.md"
if not PROMPT_PATH.exists():
    PROMPT_PATH = PROJECT_DIR / "grounded_prompt.md"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ai"
VECTOR_STORE_DIR = PROJECT_DIR / "outputs" / "vector_store"
BIAS_WARNINGS_PATH = CONFIG_DIR / "bias_warnings.json"
if not BIAS_WARNINGS_PATH.exists():
    BIAS_WARNINGS_PATH = PROJECT_DIR / "bias_warnings.json"

DEFAULT_CANDIDATES_PATH = PROJECT_DIR / "data" / "processed" / "controlled_candidate_profiles.csv"
DEFAULT_JOBS_PATH = PROJECT_DIR / "data" / "processed" / "job_descriptions.csv"
SAMPLE_CANDIDATES_PATH = PROJECT_DIR / "data" / "sample" / "controlled_candidate_profiles_sample.csv"
SAMPLE_JOBS_PATH = PROJECT_DIR / "data" / "sample" / "job_descriptions_sample.csv"



def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_prompt() -> str:
    return read_text(PROMPT_PATH)


def load_bias_warning_config() -> Dict[str, object]:
    if not BIAS_WARNINGS_PATH.exists():
        return {}
    return json.loads(read_text(BIAS_WARNINGS_PATH))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = normalize_text(text).split()
    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_words]
        if len(chunk) >= 20:
            chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


def load_knowledge_chunks(chunk_words: int, overlap_words: int) -> pd.DataFrame:
    rows = []
    for doc_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        for index, chunk in enumerate(chunk_text(read_text(doc_path), chunk_words, overlap_words), start=1):
            rows.append({"chunk_id": f"{doc_path.stem}_{index:03d}", "source_document": doc_path.name, "chunk_index": index, "text": chunk})
    if not rows:
        raise RuntimeError(f"No knowledge chunks found in {KNOWLEDGE_BASE_DIR}")
    return pd.DataFrame(rows)


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
        f"Job Title: {row['job_title']}\nRole Summary: {row['role_summary']}\n"
        f"Main Responsibilities: {row['main_responsibilities']}\nRequired Qualifications: {row['required_qualifications']}\n"
        f"Required Skills: {row['required_skills']}\nPreferred Skills: {row['preferred_skills']}\nEvaluation Criteria: {row['evaluation_criteria']}"
    )


def build_retrieval_query(row: pd.Series) -> str:
    return f"Fair structured hiring guidance. Job: {format_job_description(row)} Candidate: {row['controlled_resume_text']}"


def hash_embedding(text: str, dimensions: int = 384) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float32)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def embed_hash(texts: List[str]) -> np.ndarray:
    return np.vstack([hash_embedding(text) for text in texts])


_SENTENCE_TRANSFORMER_CACHE: Dict[str, Any] = {}


def get_sentence_transformer(model_name: str):
    """Load each sentence-transformer model only once per script run."""
    if model_name not in _SENTENCE_TRANSFORMER_CACHE:
        from sentence_transformers import SentenceTransformer
        print(f"Loading sentence-transformer model once: {model_name}")
        _SENTENCE_TRANSFORMER_CACHE[model_name] = SentenceTransformer(model_name)
    return _SENTENCE_TRANSFORMER_CACHE[model_name]


def embed_sentence_transformers(texts: List[str], model_name: str) -> np.ndarray:
    model = get_sentence_transformer(model_name)
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def create_openai_client(api_provider: str = "openai"):
    if api_provider == "azure":
        required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"]
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"Missing Azure OpenAI environment variables: {', '.join(missing)}")
        from openai import AzureOpenAI
        return AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_version=os.environ["AZURE_OPENAI_API_VERSION"])
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    from openai import OpenAI
    return OpenAI()


def normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def embed_openai(texts: List[str], model_name: str) -> np.ndarray:
    client = create_openai_client()
    response = client.embeddings.create(model=model_name, input=texts)
    return normalize_matrix(np.asarray([item.embedding for item in response.data], dtype=np.float32))


def embed_texts(texts: List[str], args: argparse.Namespace) -> np.ndarray:
    if args.embedding_provider == "sentence-transformers":
        return embed_sentence_transformers(texts, args.sentence_model)
    if args.embedding_provider == "openai":
        return embed_openai(texts, args.embedding_model)
    return embed_hash(texts)


def build_or_load_vector_store(args: argparse.Namespace) -> Tuple[pd.DataFrame, np.ndarray, object]:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = VECTOR_STORE_DIR / f"chunks_{args.chunk_words}_{args.overlap_words}.csv"
    embeddings_path = VECTOR_STORE_DIR / f"embeddings_{args.embedding_provider}.npy"
    if not args.rebuild_index and chunks_path.exists() and embeddings_path.exists():
        chunks = pd.read_csv(chunks_path)
        embeddings = np.load(embeddings_path)
    else:
        chunks = load_knowledge_chunks(args.chunk_words, args.overlap_words)
        embeddings = embed_texts(chunks["text"].tolist(), args)
        chunks.to_csv(chunks_path, index=False)
        np.save(embeddings_path, embeddings)
    faiss_index = None
    try:
        import faiss
        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss_index.add(np.asarray(embeddings, dtype="float32"))
    except Exception:
        faiss_index = None
    return chunks, embeddings, faiss_index


def mandatory_fairness_chunks(chunks: pd.DataFrame, args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.disable_mandatory_fairness_context:
        return []
    mandatory = chunks[chunks["source_document"] == args.mandatory_fairness_source].head(args.mandatory_fairness_chunks)
    items = []
    for _, chunk in mandatory.iterrows():
        items.append(
            {
                "rank": len(items) + 1,
                "chunk_id": chunk["chunk_id"],
                "source_document": chunk["source_document"],
                "similarity": np.nan,
                "retrieval_type": "mandatory_fairness",
                "text": chunk["text"],
            }
        )
    return items


def retrieve_context(row: pd.Series, chunks: pd.DataFrame, embeddings: np.ndarray, faiss_index, args: argparse.Namespace) -> Tuple[str, List[Dict[str, object]]]:
    mandatory_items = mandatory_fairness_chunks(chunks, args)
    mandatory_ids = {item["chunk_id"] for item in mandatory_items}
    semantic_k = max(0, args.top_k - len(mandatory_items))
    query_embedding = embed_texts([build_retrieval_query(row)], args)
    if faiss_index is not None:
        search_k = min(len(chunks), max(args.top_k + len(mandatory_items), semantic_k))
        scores, indices = faiss_index.search(np.asarray(query_embedding, dtype="float32"), search_k)
        top = [(int(i), float(s)) for i, s in zip(indices[0].tolist(), scores[0].tolist())]
    else:
        scores = embeddings @ query_embedding[0]
        search_k = min(len(chunks), max(args.top_k + len(mandatory_items), semantic_k))
        top_indices = np.argsort(scores)[::-1][:search_k]
        top = [(int(index), float(scores[index])) for index in top_indices]

    retrieved = list(mandatory_items)
    context_parts = []
    for item in mandatory_items:
        context_parts.append(
            f"[Mandatory Fairness Context {item['rank']} | {item['chunk_id']} | {item['source_document']}]\n{item['text']}"
        )

    semantic_rank = len(retrieved) + 1
    for chunk_index, score in top:
        chunk = chunks.iloc[chunk_index]
        if chunk["chunk_id"] in mandatory_ids:
            continue
        if len(retrieved) >= args.top_k:
            break
        item = {
            "rank": semantic_rank,
            "chunk_id": chunk["chunk_id"],
            "source_document": chunk["source_document"],
            "similarity": score,
            "retrieval_type": "semantic",
            "text": chunk["text"],
        }
        retrieved.append(item)
        context_parts.append(f"[Retrieved Context {semantic_rank} | {item['chunk_id']} | {item['source_document']} | similarity={score:.4f}]\n{item['text']}")
        semantic_rank += 1
    return "\n\n".join(context_parts), retrieved


def build_bias_warnings(row: pd.Series, config: Dict[str, object]) -> List[Dict[str, str]]:
    warnings = []
    if str(row.get("gender_condition", "")).lower() not in ["", "nan", "unknown"]:
        warnings.append({"dimension": "Gender", "message": "Gender-related cues may unintentionally influence evaluation. Focus on role-relevant evidence."})
    if str(row.get("ethnicity_condition", "")).lower() not in ["", "nan", "unknown"]:
        warnings.append({"dimension": "Ethnicity/Nationality", "message": "Name or nationality cues may unintentionally influence perceived fit. Focus on documented qualifications."})
    if str(row.get("age_condition", "")).lower() not in ["", "nan", "unknown"]:
        warnings.append({"dimension": "Age", "message": "Age-related cues may unintentionally influence evaluation. Focus on current competencies and role requirements."})
    warnings.append({"dimension": "General fairness check", "message": "Apply the same structured criteria to every candidate."})
    return warnings


def format_bias_warnings(warnings: List[Dict[str, str]]) -> str:
    return "\n".join(f"- {w['dimension']}: {w['message']}" for w in warnings)


def build_grounded_user_message(row: pd.Series, retrieved_context: str, warnings: List[Dict[str, str]]) -> str:
    return (
        "Provide grounded, bias-aware recruitment decision support. Do not make an autonomous hiring decision.\n\n"
        f"Bias-Awareness Warnings:\n{format_bias_warnings(warnings)}\n\n"
        f"Retrieved Context:\n{retrieved_context}\n\n"
        f"Candidate Resume:\n{row['controlled_resume_text']}\n\n"
        f"Job Description:\n{format_job_description(row)}"
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
        "supporting_evidence": as_list(data.get("supporting_evidence", data.get("relevant_evidence", []))),
        "bias_awareness_notes": as_list(data.get("bias_awareness_notes", data.get("bias_awareness_guidance", []))),
        "fairness_guidance": as_list(data.get("fairness_guidance", [])),
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
) -> Dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_openai(client, model, system_prompt, user_message, api_provider)
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = any(
                marker in message
                for marker in [
                    "internal error",
                    "invalid_prompt",
                    "timeout",
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


def dry_run_evaluation(row: pd.Series, retrieved_context: str, warnings: List[Dict[str, str]]) -> Dict[str, object]:
    result = {
        "candidate_summary": "Dry-run grounded decision-support summary of role-relevant qualifications.",
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
        "bias_awareness_notes": [w["message"] for w in warnings if w["dimension"] != "General fairness check"],
        "fairness_guidance": ["Use structured criteria and avoid demographic assumptions."],
        "limits_of_assessment": ["Dry-run output cannot verify all resume claims."],
        "raw_response": "",
    }
    result["supportiveness_index"] = supportiveness_index(result)
    result["role_alignment_index"] = role_alignment_index(result["role_alignment_indicators"])
    return result

def serialize(value) -> str:


    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value) if value is not None else ""


def build_output_record(row: pd.Series, result: Dict[str, object], retrieved: List[Dict[str, object]], warnings: List[Dict[str, str]], model: str, embedding_provider: str, run_id: int) -> Dict[str, object]:
    return {
        "system_condition": "grounded_rag",
        "model": model,
        "embedding_provider": embedding_provider,
        "run_id": run_id,
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
        "bias_awareness_notes": serialize(result["bias_awareness_notes"]),
        "fairness_guidance": serialize(result["fairness_guidance"]),
        "supportiveness_index": result["supportiveness_index"],
        "role_alignment_index": result["role_alignment_index"],
        "bias_warning_dimensions": "; ".join(w["dimension"] for w in warnings),
        "bias_warning_messages": " ".join(w["message"] for w in warnings),
        "bias_warnings_json": json.dumps(warnings),
        "retrieved_chunk_ids": "; ".join(item["chunk_id"] for item in retrieved),
        "retrieved_sources": "; ".join(item["source_document"] for item in retrieved),
        "retrieved_similarities": "; ".join(
            "mandatory" if pd.isna(item.get("similarity")) else f"{item['similarity']:.4f}"
            for item in retrieved
        ),
        "retrieval_types": "; ".join(str(item.get("retrieval_type", "semantic")) for item in retrieved),
        "retrieved_context_json": json.dumps(retrieved),
        "raw_response": result.get("raw_response", ""),
    }

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


def write_outputs(records: List[Dict[str, object]], output_prefix: str, announce: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{output_prefix}.csv"
    json_path = OUTPUT_DIR / f"{output_prefix}.json"
    jsonl_path = OUTPUT_DIR / f"{output_prefix}.jsonl"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    if announce:
        print(f"CSV output: {csv_path}")
        print(f"JSON output: {json_path}")
        print(f"JSONL output: {jsonl_path}")


def output_paths(output_prefix: str) -> tuple[Path, Path, Path]:
    return (
        OUTPUT_DIR / f"{output_prefix}.csv",
        OUTPUT_DIR / f"{output_prefix}.json",
        OUTPUT_DIR / f"{output_prefix}.jsonl",
    )


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


def test_retrieval(args: argparse.Namespace) -> None:
    chunks, embeddings, faiss_index = build_or_load_vector_store(args)
    sample = pd.Series({"job_title": "Data Analyst", "role_summary": "Analyze data.", "main_responsibilities": "Build dashboards", "required_qualifications": "Bachelor's degree", "required_skills": "SQL; Excel", "preferred_skills": "Python", "evaluation_criteria": "Skills; Experience", "controlled_resume_text": "Candidate has SQL and Excel experience."})
    context, retrieved = retrieve_context(sample, chunks, embeddings, faiss_index, args)
    print(context)
    print(json.dumps(retrieved, indent=2))


def run(args: argparse.Namespace) -> None:
    if args.test_retrieval:
        test_retrieval(args)
        return
    candidates_path = SAMPLE_CANDIDATES_PATH if args.use_sample else Path(args.candidates)
    jobs_path = SAMPLE_JOBS_PATH if args.use_sample else Path(args.jobs)
    system_prompt = load_prompt()
    warning_config = load_bias_warning_config()
    inputs = load_inputs(candidates_path, jobs_path, args.limit)
    chunks, embeddings, faiss_index = build_or_load_vector_store(args)
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
            retrieved_context, retrieved = retrieve_context(row, chunks, embeddings, faiss_index, args)
            warnings = build_bias_warnings(row, warning_config)
            if args.dry_run:
                result = dry_run_evaluation(row, retrieved_context, warnings)
            else:
                result = call_openai_with_retry(client, args.model, system_prompt, build_grounded_user_message(row, retrieved_context, warnings), args.api_provider)
                if args.sleep_seconds:
                    time.sleep(args.sleep_seconds)
            records.append(build_output_record(row, result, retrieved, warnings, args.model, args.embedding_provider, run_id))
            completed_keys.add(key)
            completed += 1
            print(f"[{completed}/{total}] run {run_id} {row['variant_id']} / {row['job_id']} -> role_alignment_index={result['role_alignment_index']}")
            if len(records) % args.checkpoint_every == 0:
                write_outputs(records, args.output_prefix)
                print(f"Checkpoint saved: {len(records)} records", flush=True)
    write_outputs(records, args.output_prefix, announce=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grounded RAG recruitment decision-support outputs.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--api-provider", choices=["openai", "azure"], default="openai")
    parser.add_argument("--embedding-provider", choices=["hash", "sentence-transformers", "openai"], default="hash")
    parser.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--mandatory-fairness-source", default="counterfactual_fairness_checklist.md")
    parser.add_argument("--mandatory-fairness-chunks", type=int, default=1)
    parser.add_argument("--disable-mandatory-fairness-context", action="store_true")
    parser.add_argument("--chunk-words", type=int, default=120)
    parser.add_argument("--overlap-words", type=int, default=25)
    parser.add_argument("--output-prefix", default="rag_outputs")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the output CSV.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-sample", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--test-retrieval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
