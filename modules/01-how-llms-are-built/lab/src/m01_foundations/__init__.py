"""Module 01 lab: how LLMs are built, from a raw web crawl to a transformer block.

Sub-modules:

- ``ngram``       a counting next-word predictor (lesson 1.1)
- ``cleaning``    a pre-training data pipeline over ``datasets/web`` (lesson 1.2)
- ``dedup``       exact hashing and MinHash near-duplicate detection (lesson 1.2)
- ``bpe``         a byte-level BPE tokenizer trained from scratch (lesson 1.3)
- ``transformer`` causal self-attention and a decoder block in numpy (lesson 1.4)
- ``neural``      a tiny neural next-token model trained by gradient descent (lesson 1.4)
- ``params``      parameter, memory and compute counting, dense and MoE (lesson 1.5)
"""
