# Ablation Comparison: baseline vs. ablation (fields, no retrieval) vs. grounded

N: baseline=1800, ablation=1800, grounded=1800

## Role-alignment index means
- baseline: 23.95
- ablation (fields, no retrieval): 23.38
- grounded (retrieval + fields): 28.13

Welch's t-test, ablation vs baseline (effect of fields/instructions alone): t=-1.062, p=0.2885
Welch's t-test, grounded vs ablation (effect of adding retrieval, holding fields constant): t=8.945, p=5.826e-19

## Bias-related keyword coverage (%) — same categories and methodology as Table 4-3
- gender_bias: baseline=0.0%, ablation=99.9%, grounded=100.0%
- ethnicity_name_bias: baseline=5.0%, ablation=99.9%, grounded=100.0%
- age_graduation_bias: baseline=3.6%, ablation=99.9%, grounded=100.0%
- education_prestige_bias: baseline=99.3%, ablation=99.2%, grounded=98.3%
- career_gap_bias: baseline=9.2%, ablation=20.3%, grounded=18.4%
- counterfactual_consistency: baseline=0.5%, ablation=100.0%, grounded=99.8%

## Interpretation
If ablation coverage is close to grounded coverage (both far above baseline), the coverage increase is attributable mainly to the output schema/instructions, not to retrieval specifically. If ablation coverage is close to baseline (both far below grounded), retrieval is doing the work. A result in between indicates both factors contribute.