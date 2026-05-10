"""Shopping agents — same multi-agent architecture as the investigation pipeline,
applied to consumer product discovery (the JOSEPHA-style use case).

Specialists:
    - product_researcher    → fetches product specs and candidate matches
    - review_aggregator     → summarizes professional and user reviews
    - price_comparator      → compares current/historical pricing across retailers
    - recommendation_generator → synthesizes a grounded buy recommendation

Reuses: supervisor pattern, GraphState shape, evaluator (citation/hallucination),
LLM factory, Langfuse callback.
"""
