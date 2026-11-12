"""Figures from public papers that the lessons quote, each with its source.

They are not lab results: they are recorded here once so the lessons, the ``lab facts`` command and
the tests all use the same values. Check them against the linked papers yourself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    topic: str
    claim: str
    value: float
    unit: str
    source: str


QWEN, DSV3 = "https://arxiv.org/abs/2412.15115", "https://arxiv.org/abs/2412.19437"
MIXTRAL, GEMMA3 = "https://arxiv.org/abs/2401.04088", "https://arxiv.org/abs/2503.19786"

FACTS: list[Fact] = [
    Fact("GPT-3", "parameters", 175, "billion", "https://arxiv.org/abs/2005.14165"),
    Fact("Chinchilla", "parameters (same compute budget as the 280B Gopher)", 70, "billion",
         "https://arxiv.org/abs/2203.15556"),
    Fact("Chinchilla", "Gopher parameters", 280, "billion", "https://arxiv.org/abs/2203.15556"),
    Fact("RefinedWeb", "tokens extracted from Common Crawl after filtering", 5, "trillion",
         "https://arxiv.org/abs/2306.01116"),
    Fact("RefinedWeb", "tokens in the public extract", 600, "billion", "https://arxiv.org/abs/2306.01116"),
    Fact("Dolma", "tokens", 3, "trillion", "https://arxiv.org/abs/2402.00159"),
    Fact("FineWeb", "tokens", 15, "trillion", "https://arxiv.org/abs/2406.17557"),
    Fact("FineWeb", "Common Crawl snapshots used", 96, "snapshots", "https://arxiv.org/abs/2406.17557"),
    Fact("Qwen2.5", "pre-training tokens (up from 7 trillion)", 18, "trillion", QWEN),
    Fact("Qwen2.5", "regular tokens in the byte-level BPE vocabulary", 151_643, "tokens",
         QWEN),
    Fact("Qwen2.5", "smallest open-weight dense model", 0.5, "billion", QWEN),
    Fact("Qwen2.5", "largest open-weight dense model", 72, "billion", QWEN),
    Fact("DeepSeek-V3", "total parameters", 671, "billion", DSV3),
    Fact("DeepSeek-V3", "parameters activated per token", 37, "billion", DSV3),
    Fact("DeepSeek-V3", "pre-training tokens", 14.8, "trillion", DSV3),
    Fact("DeepSeek-V3", "routed experts per MoE layer", 256, "experts", DSV3),
    Fact("DeepSeek-V3", "shared experts per MoE layer", 1, "experts", DSV3),
    Fact("DeepSeek-V3", "routed experts activated per token", 8, "experts", DSV3),
    Fact("Mixtral 8x7B", "experts per layer", 8, "experts", MIXTRAL),
    Fact("Mixtral 8x7B", "experts per token", 2, "experts", MIXTRAL),
    Fact("Mixtral 8x7B", "parameters each token has access to", 47, "billion", MIXTRAL),
    Fact("Mixtral 8x7B", "active parameters per token", 13, "billion", MIXTRAL),
    Fact("Gemma 3", "smallest model", 1, "billion", GEMMA3),
    Fact("Gemma 3", "largest model", 27, "billion", GEMMA3),
    Fact("Gemma 3", "context length (at least)", 128, "thousand tokens", GEMMA3),
    Fact("Gemma 3", "local attention layers per global layer", 5, "layers", GEMMA3),
    Fact("Gemma 3", "local attention window", 1024, "tokens", GEMMA3),
]


def fact(topic: str, claim_start: str) -> float:
    """Look up one value, for example ``fact("DeepSeek-V3", "total")``."""
    for f in FACTS:
        if f.topic == topic and f.claim.startswith(claim_start):
            return f.value
    raise KeyError(f"{topic}: {claim_start}")
