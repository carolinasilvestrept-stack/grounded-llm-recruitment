# Ragas Evaluation Summary (Grounded RAG condition)

Sample size: 150 of 1800 total grounded records (seed=42).
Records with a valid score: 150. Records that failed or had empty content: 0.

Faithfulness is computed against bias_awareness_notes + fairness_guidance only (the output fields meant to draw from the retrieved fairness/hiring-criteria/counterfactual policy documents). Context Utilization is computed against the full candidate-evaluation answer (candidate_summary, job_relevant_strengths, job_relevant_concerns, supporting_evidence), since it checks retrieval relevance to the query rather than groundedness of specific claims.

- faithfulness: mean=0.207, median=0.192, min=0.042, max=0.500, n=150
- context_utilization: mean=0.887, median=1.000, min=0.250, max=1.000, n=150
