🌐 English · [Español](resources.es.md)

# Free resources

Everything here is free to read or use. Start with the ⭐ item of each section; the rest is for when
you want depth. Papers are linked to their arXiv abstract pages, where the PDF is one click away.

## General

- ⭐ [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — build language models
  from scratch in Python, step by step, in video form.
- [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
  — university course with public slides and assignments.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) — practical course on
  Transformers, tokenizers and fine-tuning with the Hugging Face libraries.
- [uv documentation](https://docs.astral.sh/uv/) — the tool every lab uses.

## Module 01 — How LLMs are built

- ⭐ [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the clearest
  visual walk-through of attention.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original Transformer paper.
- [minbpe](https://github.com/karpathy/minbpe) — a minimal, readable BPE tokenizer.
- [FineWeb](https://arxiv.org/abs/2406.17557), [RefinedWeb](https://arxiv.org/abs/2306.01116) and
  [Dolma](https://arxiv.org/abs/2402.00159) — how large web datasets are built and cleaned.
- [Common Crawl](https://commoncrawl.org) — the open web crawl most pre-training data starts from.
- [Scaling laws](https://arxiv.org/abs/2001.08361) and [Chinchilla](https://arxiv.org/abs/2203.15556)
  — how model size, data and compute trade off.

## Module 02 — From base model to assistant

- ⭐ [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155)
  — the reference description of SFT followed by RLHF.
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — why top-p
  sampling exists.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — preference tuning without a
  reward model.
- [Chatbot Arena](https://arxiv.org/abs/2403.04132) and [LLM-as-a-judge](https://arxiv.org/abs/2306.05685)
  — human and model-based evaluation.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the open-source tool
  many teams use to run benchmarks.

## Module 03 — LLM playground

- ⭐ [Ollama](https://ollama.com) — run open models locally behind a simple API.
- [OpenAI API documentation](https://platform.openai.com/docs) — the chat-completions format that
  many servers imitate.

## Module 04 — Prompting, context and fine-tuning

- ⭐ [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- [Chain-of-thought prompting](https://arxiv.org/abs/2201.11903) and
  [zero-shot reasoners](https://arxiv.org/abs/2205.11916).
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — where in the context models pay attention.
- [QLoRA](https://arxiv.org/abs/2305.14314) and [adapters](https://arxiv.org/abs/1902.00751).
- [Hugging Face PEFT](https://huggingface.co/docs/peft) — the library most people use for LoRA.

## Module 05 — RAG I: retrieval

- ⭐ [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
- [Faiss](https://github.com/facebookresearch/faiss) and [the Faiss library paper](https://arxiv.org/abs/2401.08281).
- [HNSW](https://arxiv.org/abs/1603.09320) — the graph index behind many vector databases.
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) and [Sentence Transformers](https://www.sbert.net)
  — practical embedding models.
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) and [ColBERT](https://arxiv.org/abs/2004.12832).

## Module 06 — RAG II: the support chatbot

- ⭐ [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131).
- [RAGAS](https://arxiv.org/abs/2309.15217) — reference-free metrics for RAG.
- [Indirect prompt injection](https://arxiv.org/abs/2302.12173) — why retrieved text must be treated
  as data.

## Module 07 — Workflows, tools and MCP

- ⭐ [Model Context Protocol](https://modelcontextprotocol.io) — the specification and guides.
- [Toolformer](https://arxiv.org/abs/2302.04761) — models that learn to use tools.
- [AutoGen](https://arxiv.org/abs/2308.08155) — multi-agent conversation framework.

## Module 08 — The Ask-the-Web agent

- ⭐ [ReAct](https://arxiv.org/abs/2210.03629) — reasoning and acting in a loop.
- [Reflexion](https://arxiv.org/abs/2303.11366), [ReWOO](https://arxiv.org/abs/2305.18323) and
  [LATS](https://arxiv.org/abs/2310.04406).
- [WebGPT](https://arxiv.org/abs/2112.09332) — browsing with citations.
- [Agent2Agent (A2A) protocol](https://github.com/a2aproject/A2A).

## Module 09 — Reasoning models

- ⭐ [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — an openly documented reasoning-model recipe.
- [Self-consistency](https://arxiv.org/abs/2203.11171), [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
  and [Self-Refine](https://arxiv.org/abs/2303.17651).
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — process reward models.
- [Scaling LLM test-time compute](https://arxiv.org/abs/2408.03314).
- [STaR](https://arxiv.org/abs/2203.14465), [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
  and [Meta-CoT](https://arxiv.org/abs/2501.04682).

## Module 10 — Deep research and local deployment

- ⭐ [llama.cpp](https://github.com/ggml-org/llama.cpp) — run quantised models on CPUs and GPUs.
- [vLLM documentation](https://docs.vllm.ai) and [PagedAttention](https://arxiv.org/abs/2309.06180).
- [GPTQ](https://arxiv.org/abs/2210.17323) and [AWQ](https://arxiv.org/abs/2306.00978) — weight
  quantisation methods.
- [Speculative decoding](https://arxiv.org/abs/2211.17192) and
  [FlashAttention](https://arxiv.org/abs/2205.14135).

## Templates and practice

- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — the idea behind our
  [model card template](../templates/model-card.md).

---

[🏠 Course home](../README.md) · [Progress journal →](progress-journal.md)
