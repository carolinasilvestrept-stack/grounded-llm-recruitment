# Results Readiness Report

## Data Preparation
- OK: Controlled candidate profiles: 120 rows
- OK: Base resumes: 15
- OK: Source datasets: huggingface_resume_atlas_1
- OK: Job descriptions: 5

## AI Outputs
- OK: baseline rows: 1800
- OK: baseline gender_condition groups: female, male
- OK: baseline ethnicity_condition groups: majority, minority
- OK: baseline age_condition groups: older, younger
- OK: baseline role_alignment_index valid values: 1800/1800
- OK: grounded RAG rows: 1800
- OK: grounded RAG gender_condition groups: female, male
- OK: grounded RAG ethnicity_condition groups: majority, minority
- OK: grounded RAG age_condition groups: older, younger
- OK: grounded RAG role_alignment_index valid values: 1800/1800
- OK: Expected rows per run: 600
- OK: baseline row coverage: 1800 rows across 3 run(s)
- OK: grounded RAG row coverage: 1800 rows across 3 run(s)

## Analysis Tables
- OK: combined_ai_outputs.csv: present
- OK: system_comparison.csv: present
- OK: disparity_summary.csv: present
- OK: intersectional_disparity_summary.csv: present
- OK: enhanced_metric_group_summary.csv: present
- OK: enhanced_disparity_ranges.csv: present
- OK: retrieval_similarity_summary.csv: present
- OK: statistical_tests_ai_outputs.csv: present
- OK: effect_sizes_ai_outputs.csv: present
- OK: counterfactual_disparity_summary.csv: present
- OK: bias_category_coverage_summary.csv: present
- OK: ai_output_schema_report.csv: present

## Research Question Readiness
- OK: RQ1-RQ3 (Section 2.8) can be tested when baseline and grounded RAG outputs are present.
