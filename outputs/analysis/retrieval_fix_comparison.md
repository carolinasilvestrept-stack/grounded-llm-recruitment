# Retrieval-Query Fix: Before/After Comparison

N: baseline=1800, original_grounded=1800, fixed_grounded=1800

## Role-alignment index: paired comparison vs. baseline
- **Original (query included demographic cues)**: n=1800, baseline=23.95, grounded=28.60, diff=4.652, t=50.557, p=0, Cohen's dz=1.192
- **Fixed (query excludes demographic cues)**: n=1800, baseline=23.95, grounded=28.13, diff=4.181, t=43.876, p=1.522e-286, Cohen's dz=1.034

## Counterfactual range (role_alignment_index): before vs. after the fix
- Baseline: 5.660
- Original grounded: 6.253
- Fixed grounded: 5.976

## Bias-keyword coverage (%): original vs. fixed
- gender_bias: original=100.0%, fixed=100.0%
- ethnicity_name_bias: original=100.0%, fixed=100.0%
- age_graduation_bias: original=100.0%, fixed=100.0%
- education_prestige_bias: original=98.3%, fixed=98.3%
- career_gap_bias: original=17.3%, fixed=18.4%
- counterfactual_consistency: original=99.3%, fixed=99.8%

## Interpretation
If the fixed condition's counterfactual range is similar to or smaller than the original's, this suggests the demographic-cue leakage in the retrieval query was a meaningful contributor to the RQ2 variance increase. If the numbers are nearly identical, the leakage's practical impact on this specific result was small, even though it remained a genuine methodological flaw worth fixing.