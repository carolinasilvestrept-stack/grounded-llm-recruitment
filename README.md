# Grounded LLM Recruitment Decision-Support Prototype

This repository contains the code for the thesis experiment on grounded LLM-based recruitment decision support.

The system compares two conditions:

- **Baseline LLM**: evaluates a controlled candidate profile against a job description.
- **Grounded RAG**: evaluates the same case with retrieved fairness, debiasing, hiring criteria, and counterfactual consistency guidance.

The prototype is a decision-support system only. It does not make hire/reject decisions, rank candidates, or replace human judgment.

## Main Structure

```text
config/             demographic variants, jobs, warning settings
data/processed/     prepared controlled candidate profiles and job descriptions
docs/               generated human-readable result summaries
knowledge_base/     grounding documents for the RAG condition
outputs/            generated model outputs, vector store, and analysis tables
prompts/            baseline and grounded prompts
src/generation/     baseline and grounded RAG generation scripts
src/pipeline/       main workflow runner
src/analysis/       computational analysis scripts, including the Ragas evaluation
src/diagnostics/    validation and readiness checks
```

## Setup

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:

```powershell
copy .env.example .env
```

Verify the connection works before running anything expensive:

```powershell
python check_setup.py
```

## Run The Main Experiment

```powershell
python src\pipeline\run_thesis_pipeline.py --api-provider azure --embedding-provider sentence-transformers --runs 3 --rebuild-rag-index --resume
```

This generates:

- `outputs/ai/baseline_outputs.csv`
- `outputs/ai/rag_outputs.csv`
- analysis tables in `outputs/analysis/`
- RAG vector-store files in `outputs/vector_store/`

## Quick Smoke Test

```powershell
python src\pipeline\run_thesis_pipeline.py --dry-run --runs 1 --limit 2
```

## Ragas Evaluation (Grounded RAG condition)

Faithfulness and Context Utilization on a random sample of `outputs/ai/rag_outputs.csv`, scored by the same Azure deployment used for generation:

```powershell
python src\analysis\ragas_evaluation.py --sample-size 150
```

Faithfulness is computed against `bias_awareness_notes` + `fairness_guidance` only, since those are the output fields meant to draw from the retrieved fairness/hiring-criteria/counterfactual policy documents (the candidate-facts fields come from the resume, not retrieval, and aren't a meaningful faithfulness target). Context Utilization is computed against the full evaluation output, since it checks retrieval relevance to the query rather than groundedness of specific claims. Results are saved to `outputs/analysis/ragas_evaluation_by_case.csv` and `ragas_evaluation_summary.md`.

## Useful Checks

Check the environment and required files/data are all in place:

```powershell
python src\diagnostics\check_project_health.py
```

Validate AI outputs:

```powershell
python src\diagnostics\validate_ai_outputs.py
```

Create a readiness summary:

```powershell
python src\diagnostics\results_readiness_report.py
```
