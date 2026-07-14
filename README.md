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

Use `--rag-output <path>` to evaluate a different grounded-condition CSV (e.g. before/after a retrieval fix) without overwriting the default results — output filenames get a suffix derived from the input filename automatically.

For a properly documented manual check of the Faithfulness numbers (rather than a single anecdotal example), export a sample of individual claims to code by hand:

```powershell
python src\analysis\ragas_manual_coding_sample.py --sample-size 25
```

This splits each sampled record's `bias_awareness_notes`/`fairness_guidance` into individual claims alongside the retrieved context, with a blank `coder_verdict` column (`traceable` / `general_principle_only` / `not_traceable`) to fill in by hand. Saved to `outputs/analysis/ragas_manual_coding_sample.csv`.

## Ablation Study: Does Retrieval or Just the Prompt Schema Drive the Bias-Mitigation Effect?

`prompts/ablation_fields_no_retrieval_prompt.md` asks for the same output schema as the grounded condition (including `bias_awareness_notes`/`fairness_guidance`) but with no retrieval at all, isolating whether the RAG condition's improvements come from retrieval specifically or just from prompting the model to produce bias-aware output fields:

```powershell
python src\generation\run_baseline_llm.py --prompt-file prompts\ablation_fields_no_retrieval_prompt.md --output-prefix ablation_fields_no_retrieval_outputs --system-condition ablation_fields_no_retrieval --runs 3 --api-provider azure --model gpt-5.2
python src\analysis\ablation_comparison.py
```

`--system-condition` labels the `system_condition` column correctly for this condition (it defaults to `baseline_llm`, which would otherwise mislabel the ablation data as the real baseline). `ablation_comparison.py` reports role-alignment index and bias-keyword coverage across all three conditions (baseline / ablation / grounded) with paired significance tests, saved to `outputs/analysis/ablation_comparison.md`.

If `run_baseline_llm.py` output ever lacks `bias_awareness_notes`/`fairness_guidance` (they're not part of the plain-baseline output schema this script was originally written for), recover them from the already-generated `raw_response` field rather than regenerating:

```powershell
python src\analysis\backfill_ablation_bias_fields.py
```

## Robustness Re-Analysis and Retrieval-Query Fix

`robustness_reanalysis.py` re-analyzes the existing baseline/grounded dataset with no new API calls: candidate-level paired tests (addressing pseudoreplication across the 1,800 rows), proportional (relative) counterfactual range, subgroup direction of disparity, and a full 2x2x2 intersectional breakdown:

```powershell
python src\analysis\robustness_reanalysis.py
```

`retrieval_fix_comparison.py` quantifies the effect of the retrieval-query fix (`build_retrieval_query()` in `run_rag_system.py` now searches on `core_resume_text`, which excludes the candidate's name and demographic-cue labels, instead of `controlled_resume_text`, which included them — the LLM's evaluation prompt still sees the full demographic cues, since that's the counterfactual manipulation itself) by comparing the pre-fix and post-fix grounded conditions against the same baseline:

```powershell
python src\analysis\retrieval_fix_comparison.py
```

Run all three of the above (robustness re-analysis, ablation comparison if its data is ready, and the Ragas manual-coding sample export) in one pass:

```powershell
python src\analysis\run_all_new_analyses.py
```

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
