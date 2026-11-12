🌐 English · [Español](glossary.es.md)

# Glossary · Glosario

Every key term in the course, with its Spanish equivalent. Terms are sorted alphabetically by the
English name. When Spanish speakers usually keep the English word, the Spanish column says so.

## A

| Term | Español | Meaning |
|---|---|---|
| A2A (Agent2Agent protocol) | protocolo A2A | An open protocol for agents built by different teams to discover each other and exchange tasks. |
| Abstention | abstención | Answering "I don't know" (or handing off) instead of guessing when the evidence is missing. |
| Adapter | adaptador | A small trainable module inserted into a frozen model to adapt it to a task. |
| Agent | agente | A system where a model decides which steps and tools to use, in a loop, to reach a goal. |
| Agentic system | sistema agéntico | Any system that uses an LLM with some autonomy: from fixed workflows to fully autonomous agents. |
| Approximate nearest neighbour (ANN) | vecino más cercano aproximado | Search that trades a little recall for much faster vector search (IVF, HNSW, PQ). |
| Attention | atención | The Transformer operation that lets each token weigh information from other tokens. |

## B

| Term | Español | Meaning |
|---|---|---|
| Base model | modelo base | A model after pre-training only: it continues text but does not follow instructions reliably. |
| Beam search | búsqueda en haz (*beam search*) | Decoding that keeps the `k` most likely partial sequences at each step. |
| Best-of-N | mejor de N (*best-of-N*) | Generate N answers and keep the one a verifier or reward model scores highest. |
| BM25 | BM25 | A classic full-text ranking function based on term frequency, rarity and document length. |
| Byte-pair encoding (BPE) | codificación por pares de bytes (BPE) | A tokenizer that repeatedly merges the most frequent pair of symbols into a new token. |

## C

| Term | Español | Meaning |
|---|---|---|
| Chain of thought (CoT) | cadena de pensamiento | Intermediate reasoning steps written before the final answer. |
| Chat template | plantilla de chat | The exact text format that turns a list of messages into the model's input. |
| Chunking | fragmentación (*chunking*) | Splitting documents into retrievable pieces (chunks). |
| Citation | cita | A reference from an answer to the source passage that supports it. |
| Compaction | compactación | Shrinking context (summaries, pruning, deduplication) to fit a token budget. |
| Concept drift | deriva de concepto | A change over time in the relationship between inputs and labels. |
| Context engineering | ingeniería de contexto | Deciding what goes into the model's context, in what order and how much. |
| Context window | ventana de contexto | The maximum number of tokens a model can read in one call. |
| Contamination | contaminación | Evaluation data that leaked into training data, inflating scores. |
| Cross-entropy | entropía cruzada | The loss used to train language models: how surprised the model is by the true next token. |

## D

| Term | Español | Meaning |
|---|---|---|
| Data leakage | fuga de datos (*data leakage*) | Information in the inputs that would not be available at prediction time, often the label itself. |
| Decoder-only | solo decodificador (*decoder-only*) | The Transformer variant used by most LLMs: it predicts each token from the previous ones. |
| Decoding | decodificación | Turning the model's next-token probabilities into text (greedy, beam, sampling). |
| Deduplication | deduplicación | Removing exact and near-duplicate documents. |
| Deep research | investigación profunda (*deep research*) | An agent that plans, searches, reads and writes a cited report over many sources. |
| Direct Preference Optimization (DPO) | optimización directa de preferencias (DPO) | Training on preference pairs without a separate reward model or RL loop. |
| Distillation | destilación | Training a smaller model to imitate a larger one's outputs. |

## E

| Term | Español | Meaning |
|---|---|---|
| Embedding | *embedding* (vector de representación) | A vector that represents text so that similar meanings are close together. |
| Evaluation harness | arnés de evaluación | Code that runs a system on a fixed dataset and reports metrics reproducibly. |
| Exact match (EM) | coincidencia exacta | A metric: 1 if the normalised answer equals the reference, else 0. |

## F

| Term | Español | Meaning |
|---|---|---|
| FAISS | FAISS | An open-source library for efficient vector similarity search. |
| Faithfulness | fidelidad | Whether every claim in an answer is supported by the provided context. |
| Few-shot prompting | *prompting* con pocos ejemplos (*few-shot*) | Including a few solved examples in the prompt. |
| Fine-tuning | ajuste fino (*fine-tuning*) | Continuing training of a pre-trained model on task-specific data. |

## G

| Term | Español | Meaning |
|---|---|---|
| GGUF | GGUF | A file format for (often quantised) model weights used by llama.cpp and Ollama. |
| Greedy decoding | decodificación voraz (*greedy*) | Always picking the most likely next token. |
| Grounding | fundamentación (*grounding*) | Basing an answer on supplied sources rather than on the model's memory. |
| Guardrail | barrera de protección (*guardrail*) | A check before or after the model that blocks or fixes unsafe input or output. |

## H

| Term | Español | Meaning |
|---|---|---|
| Hallucination | alucinación | Fluent output that is not supported by facts or sources. |
| HNSW | HNSW | Hierarchical Navigable Small World: a graph-based ANN index. |
| Hybrid search | búsqueda híbrida | Combining keyword and vector search results, for example with RRF. |

## I

| Term | Español | Meaning |
|---|---|---|
| Inference | inferencia | Running a trained model to produce outputs. |
| Inference-time scaling | escalado en tiempo de inferencia | Spending more compute when answering (more samples, search, longer reasoning) to improve accuracy. |
| Instruction tuning | ajuste por instrucciones | Supervised fine-tuning on instruction–response pairs. |
| IVF (inverted file index) | índice de archivo invertido (IVF) | An ANN index that clusters vectors and searches only the closest clusters. |

## J

| Term | Español | Meaning |
|---|---|---|
| JSON Schema | JSON Schema | A standard for describing the shape of JSON data; used to define tool parameters. |

## K

| Term | Español | Meaning |
|---|---|---|
| Knowledge graph | grafo de conocimiento | Facts stored as entity–relation–entity triples. |
| KV cache | caché KV | Stored attention keys and values that avoid recomputing past tokens during generation. |

## L

| Term | Español | Meaning |
|---|---|---|
| Latency | latencia | Time from request to response; for LLMs, often split into time to first token and time per token. |
| LLM (large language model) | modelo de lenguaje grande (LLM) | A neural network trained on large text corpora to predict the next token. |
| LLM-as-a-judge | LLM como juez | Using a model to grade outputs; useful but biased, so it must be calibrated. |
| Logits | *logits* | The raw scores a model outputs for each token before softmax. |
| LoRA (low-rank adaptation) | LoRA (adaptación de bajo rango) | Fine-tuning that learns a small low-rank update instead of changing all weights. |
| Loss masking | enmascarado de la pérdida | Training only on some tokens (for example the assistant's reply) in SFT. |

## M

| Term | Español | Meaning |
|---|---|---|
| MCP (Model Context Protocol) | MCP (Model Context Protocol) | An open protocol to connect AI applications to tools, resources and prompts through servers. |
| MinHash | MinHash | A technique to estimate Jaccard similarity quickly, used for near-duplicate detection. |
| Mixture of Experts (MoE) | mezcla de expertos (MoE) | An architecture where each token activates only some expert sub-networks. |
| Multi-agent system | sistema multiagente | Several agents with different roles that cooperate on a task. |

## N

| Term | Español | Meaning |
|---|---|---|
| Near-duplicate | casi duplicado | A document almost identical to another (small edits, different boilerplate). |
| Next-token prediction | predicción del siguiente token | The training objective of language models. |

## O

| Term | Español | Meaning |
|---|---|---|
| Orchestrator-worker | orquestador-trabajadores | A workflow where one model splits a task and delegates parts to worker calls. |
| Outcome reward model (ORM) | modelo de recompensa de resultado (ORM) | A model that scores only the final answer. |

## P

| Term | Español | Meaning |
|---|---|---|
| Parameter-efficient fine-tuning (PEFT) | ajuste fino eficiente en parámetros (PEFT) | Methods that train a small fraction of parameters (adapters, LoRA). |
| Perplexity | perplejidad | The exponential of the average cross-entropy; lower means the model is less surprised. |
| PII (personally identifiable information) | datos personales (PII) | Data that identifies a person: names, emails, phone numbers, addresses. |
| Post-training | post-entrenamiento | Everything after pre-training that turns a base model into an assistant (SFT, RLHF, DPO). |
| PPO (Proximal Policy Optimization) | PPO (optimización de política proximal) | The RL algorithm used in classic RLHF. |
| Pre-training | preentrenamiento | Training on huge amounts of text to predict the next token. |
| Process reward model (PRM) | modelo de recompensa de proceso (PRM) | A model that scores each reasoning step, not just the answer. |
| Prompt | *prompt* | The input text given to a model. |
| Prompt chaining | encadenamiento de *prompts* | Splitting a task into a fixed sequence of model calls. |
| Prompt injection | inyección de *prompts* | Text that tries to override the system's instructions, directly or hidden in retrieved content. |

## Q

| Term | Español | Meaning |
|---|---|---|
| Quantization | cuantización | Storing weights with fewer bits (8, 4...) to save memory and speed up inference. |

## R

| Term | Español | Meaning |
|---|---|---|
| RAFT | RAFT | Retrieval-augmented fine-tuning: training a model to answer from documents with distractors. |
| RAG (retrieval-augmented generation) | generación aumentada por recuperación (RAG) | Retrieving relevant documents and giving them to the model as context. |
| ReAct | ReAct | An agent loop that interleaves reasoning, tool actions and observations. |
| Reasoning model | modelo de razonamiento | A model trained to think in long chains before answering. |
| Recall@k | *recall@k* (exhaustividad en k) | The share of queries whose relevant item appears in the top k results. |
| Reciprocal rank fusion (RRF) | fusión por rango recíproco (RRF) | Merging ranked lists by summing 1 / (k + rank). |
| Reflection | reflexión | A pattern where a model critiques and improves its own output. |
| Reranking | reordenación (*reranking*) | Re-scoring the top retrieved items with a more precise (and slower) model. |
| Retrieval | recuperación | Finding the documents most relevant to a query. |
| ReWOO | ReWOO | An agent pattern that plans all tool calls first, then executes them, then answers. |
| Reward hacking | explotación de la recompensa (*reward hacking*) | A model maximising a reward in ways that do not reflect the intended goal. |
| Reward model | modelo de recompensa | A model trained on human preferences to score outputs. |
| RLHF | aprendizaje por refuerzo con retroalimentación humana (RLHF) | Reinforcement learning from human feedback. |
| Routing | enrutamiento | Classifying an input and sending it to a specialised path. |

## S

| Term | Español | Meaning |
|---|---|---|
| Sampling | muestreo | Picking the next token at random according to (adjusted) probabilities. |
| Self-attention | autoatención | Attention where queries, keys and values come from the same sequence. |
| Self-consistency | autoconsistencia | Sampling several reasoning paths and taking a majority vote on the answer. |
| SFT (supervised fine-tuning) | ajuste fino supervisado (SFT) | Training on example inputs and ideal outputs. |
| Speculative decoding | decodificación especulativa | A small draft model proposes tokens that the large model verifies in parallel. |
| STaR (Self-Taught Reasoner) | STaR | Bootstrapping reasoning by fine-tuning on the model's own correct rationales. |
| System prompt | *prompt* de sistema | Instructions that set the assistant's role and rules for a conversation. |

## T

| Term | Español | Meaning |
|---|---|---|
| Temperature | temperatura | A divisor of the logits: low values make output more deterministic, high values more random. |
| Time to first token (TTFT) | tiempo hasta el primer token (TTFT) | Latency until the first output token arrives. |
| Token | *token* | The unit of text a model reads and writes: a word, part of a word or a byte. |
| Tokenizer | tokenizador | The component that converts text to tokens and back. |
| Tool calling | llamada a herramientas | The model asks the application to run a function with structured arguments. |
| Top-k sampling | muestreo *top-k* | Sampling only among the k most likely tokens. |
| Top-p (nucleus) sampling | muestreo *top-p* (de núcleo) | Sampling among the smallest set of tokens whose probabilities add up to p. |
| Transformer | Transformer | The neural-network architecture behind modern LLMs, built on attention. |
| Tree of Thoughts (ToT) | árbol de pensamientos (ToT) | Exploring several partial reasoning paths as a tree, with evaluation and pruning. |

## V

| Term | Español | Meaning |
|---|---|---|
| Vector database | base de datos vectorial | A store that indexes embeddings for similarity search. |
| Verifier | verificador | A program or model that checks whether an answer or step is correct. |

## Z

| Term | Español | Meaning |
|---|---|---|
| Zero-shot prompting | *prompting* sin ejemplos (*zero-shot*) | Asking the model to do a task with instructions only, no examples. |

---

[🏠 Course home](../README.md) · [Resources →](resources.md)
