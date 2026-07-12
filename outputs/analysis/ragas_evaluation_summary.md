# Ragas Evaluation Summary (Grounded RAG condition)

Sample size: 150 of 1800 total grounded records (seed=42).
Records with a valid score: 149. Records that failed or had empty content: 1.

Faithfulness is computed against bias_awareness_notes + fairness_guidance only (the output fields meant to draw from the retrieved fairness/hiring-criteria/counterfactual policy documents). Context Utilization is computed against the full candidate-evaluation answer (candidate_summary, job_relevant_strengths, job_relevant_concerns, supporting_evidence), since it checks retrieval relevance to the query rather than groundedness of specific claims.

- faithfulness: mean=0.195, median=0.188, min=0.053, max=0.436, n=148
- context_utilization: mean=0.959, median=1.000, min=0.417, max=1.000, n=148
