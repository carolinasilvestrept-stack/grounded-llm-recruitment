# Advanced Results Summary

This file summarizes additional analyses that make the current computational results easier to interpret.

## Paired System Differences

Because the same candidate-job cases are evaluated by both systems, paired comparisons are methodologically useful.

| Metric | Baseline mean | Grounded RAG mean | Mean difference | Paired p-value | Cohen dz |
|---|---:|---:|---:|---:|---:|
| role_alignment_index | 23.95 | 28.60 | 4.65 | 0 | 1.192 |
| supportiveness_index | 95.07 | 93.48 | -1.59 | 5.246e-13 | -0.171 |

The role-alignment paired comparison indicates that grounded RAG produced higher role-alignment values for the same candidate-job cases. This should not be interpreted as bias reduction by itself.

## Job-Level Disparity Changes

Across role-alignment job-by-dimension comparisons, disparity decreased in 11 cases, increased in 9 cases, and was unchanged in 0 cases.

The detailed job-level table is saved as `outputs/analysis/job_level_disparity_changes.csv`.

## Counterfactual Disparity

For each base resume, job, and run, this metric calculates the range across the 8 demographic variants. Lower values indicate more consistent treatment of otherwise equivalent profiles.

| System condition | Metric | Mean range | Median range | Max range |
|---|---|---:|---:|---:|
| baseline_llm | role_alignment_index | 5.66 | 5.00 | 16.25 |
| baseline_llm | supportiveness_index | 12.98 | 12.50 | 37.50 |
| grounded_rag | role_alignment_index | 6.25 | 6.25 | 16.25 |
| grounded_rag | supportiveness_index | 16.14 | 14.29 | 44.44 |

Paired counterfactual comparisons:

| Metric | Baseline mean range | Grounded RAG mean range | Mean change | Paired p-value | Cohen dz |
|---|---:|---:|---:|---:|---:|
| role_alignment_index | 5.66 | 6.25 | 0.59 | 0.008509 | 0.177 |
| supportiveness_index | 12.98 | 16.14 | 3.16 | 2.544e-07 | 0.354 |

Detailed case-level counterfactual ranges are saved as `outputs/analysis/counterfactual_disparity_by_case.csv`.

## Text Output Counts

Average text-structure counts by system condition:

| System condition | Strengths | Concerns | Evidence items | Limits |
|---|---:|---:|---:|---:|
| baseline_llm | 3.07 | 4.20 | 5.10 | 3.88 |
| grounded_rag | 2.87 | 4.18 | 4.50 | 3.09 |

These counts help identify whether one condition produces more strengths, concerns, evidence items, or assessment limitations. They are diagnostic indicators rather than direct fairness metrics.

## Interpretation

The additional analyses strengthen the current interpretation: grounded RAG changes output behavior and improves traceability, but the fairness effect is mixed. In the current run, ethnicity and intersectional role-alignment disparities are lower, while gender and age disparities are higher.
