"""Module 10 lab: a deep-research agent over an offline web snapshot, plus local-deployment maths.

Research pipeline: planner -> search (BM25) -> credibility -> reader (sanitise, quote, verify)
-> note store (dedup, supersession, compaction) -> reflection -> writer -> grader.
Deployment: memory estimator, quantisation error, batching simulator, KV paging, cost.
"""
