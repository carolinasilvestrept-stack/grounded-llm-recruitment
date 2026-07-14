# Bias Category Coverage

This diagnostic estimates how often generated outputs explicitly mention fairness-relevant bias categories. It is a text-coverage indicator, not a direct fairness metric.

| System condition | Gender | Ethnicity/name | Age/graduation | Education prestige | Career gap | Counterfactual consistency |
|---|---:|---:|---:|---:|---:|---:|
| baseline_llm | 0.0% | 5.0% | 3.6% | 99.3% | 9.2% | 0.5% |
| grounded_rag | 100.0% | 100.0% | 100.0% | 98.3% | 18.4% | 99.8% |

Interpretation: higher coverage means the system more frequently surfaces a category in its written output. It does not prove that the evaluation is fair, but it helps show whether grounding makes fairness risks more visible to users.
