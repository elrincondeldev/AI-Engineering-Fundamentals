"""Module 06 lab: a grounded customer-support chatbot for Hearthwise.

The package is small on purpose. Read it in this order:

1. ``kb.py`` and ``sanitize.py``  load the help centre and clean retrieved text.
2. ``retriever.py``               BM25 with metadata and access-control filters.
3. ``prompting.py``               the grounded RAG prompt.
4. ``responder.py``               the offline stand-in model (extractive, cites, abstains).
5. ``guardrails.py`` and ``bot.py``  routing, guardrails and handoff.
6. ``metrics.py``, ``evaluate.py`` and ``judge.py``  the evaluation harness.
7. ``raft.py``                    a RAFT-style fine-tuning dataset builder and validator.
"""
