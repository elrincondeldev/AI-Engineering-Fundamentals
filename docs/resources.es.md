🌐 [English](resources.md) · Español

# Recursos gratuitos

Todo lo que hay aquí se puede leer o usar gratis. Empieza por el elemento ⭐ de cada sección; el resto
es para cuando quieras profundizar. Los artículos enlazan a su página de resumen en arXiv, desde donde
el PDF está a un clic. La mayoría de los recursos están en inglés.

## General

- ⭐ [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — construye modelos de
  lenguaje desde cero en Python, paso a paso, en vídeo.
- [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
  — curso universitario con diapositivas y ejercicios públicos.
- [Curso de LLM de Hugging Face](https://huggingface.co/learn/llm-course) — curso práctico sobre
  Transformers, tokenizadores y *fine-tuning* con las bibliotecas de Hugging Face.
- [Documentación de uv](https://docs.astral.sh/uv/) — la herramienta que usan todos los labs.

## Módulo 01 — Cómo se construyen los LLM

- ⭐ [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — el recorrido
  visual más claro por la atención.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — el artículo original del
  Transformer.
- [minbpe](https://github.com/karpathy/minbpe) — un tokenizador BPE mínimo y fácil de leer.
- [FineWeb](https://arxiv.org/abs/2406.17557), [RefinedWeb](https://arxiv.org/abs/2306.01116) y
  [Dolma](https://arxiv.org/abs/2402.00159) — cómo se construyen y limpian los grandes datasets web.
- [Common Crawl](https://commoncrawl.org) — el rastreo abierto de la web del que parten casi todos los
  datos de preentrenamiento.
- [Leyes de escala](https://arxiv.org/abs/2001.08361) y [Chinchilla](https://arxiv.org/abs/2203.15556)
  — cómo se equilibran tamaño del modelo, datos y cómputo.

## Módulo 02 — Del modelo base al asistente

- ⭐ [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155)
  — la descripción de referencia de SFT seguido de RLHF.
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — por qué existe el
  muestreo *top-p*.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — ajuste por preferencias sin
  modelo de recompensa.
- [Chatbot Arena](https://arxiv.org/abs/2403.04132) y [LLM como juez](https://arxiv.org/abs/2306.05685)
  — evaluación humana y evaluación con modelos.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — la herramienta de
  código abierto que muchos equipos usan para ejecutar *benchmarks*.

## Módulo 03 — Playground de LLM

- ⭐ [Ollama](https://ollama.com) — ejecuta modelos abiertos en local detrás de una API sencilla.
- [Documentación de la API de OpenAI](https://platform.openai.com/docs) — el formato de *chat
  completions* que imitan muchos servidores.

## Módulo 04 — Prompting, contexto y fine-tuning

- ⭐ [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- [Chain-of-thought prompting](https://arxiv.org/abs/2201.11903) y
  [razonamiento sin ejemplos](https://arxiv.org/abs/2205.11916).
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — a qué parte del contexto prestan atención
  los modelos.
- [QLoRA](https://arxiv.org/abs/2305.14314) y [adapters](https://arxiv.org/abs/1902.00751).
- [Hugging Face PEFT](https://huggingface.co/docs/peft) — la biblioteca que casi todo el mundo usa
  para LoRA.

## Módulo 05 — RAG I: recuperación

- ⭐ [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
- [Faiss](https://github.com/facebookresearch/faiss) y [el artículo de la biblioteca Faiss](https://arxiv.org/abs/2401.08281).
- [HNSW](https://arxiv.org/abs/1603.09320) — el índice en grafo detrás de muchas bases de datos
  vectoriales.
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) y [Sentence Transformers](https://www.sbert.net)
  — modelos de *embeddings* prácticos.
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) y [ColBERT](https://arxiv.org/abs/2004.12832).

## Módulo 06 — RAG II: el chatbot de soporte

- ⭐ [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131).
- [RAGAS](https://arxiv.org/abs/2309.15217) — métricas para RAG sin respuestas de referencia.
- [Inyección indirecta de *prompts*](https://arxiv.org/abs/2302.12173) — por qué el texto recuperado
  debe tratarse como datos.

## Módulo 07 — Workflows, herramientas y MCP

- ⭐ [Model Context Protocol](https://modelcontextprotocol.io) — la especificación y sus guías.
- [Toolformer](https://arxiv.org/abs/2302.04761) — modelos que aprenden a usar herramientas.
- [AutoGen](https://arxiv.org/abs/2308.08155) — marco de conversación multiagente.

## Módulo 08 — El agente Ask-the-Web

- ⭐ [ReAct](https://arxiv.org/abs/2210.03629) — razonar y actuar en bucle.
- [Reflexion](https://arxiv.org/abs/2303.11366), [ReWOO](https://arxiv.org/abs/2305.18323) y
  [LATS](https://arxiv.org/abs/2310.04406).
- [WebGPT](https://arxiv.org/abs/2112.09332) — navegación con citas.
- [Protocolo Agent2Agent (A2A)](https://github.com/a2aproject/A2A).

## Módulo 09 — Modelos de razonamiento

- ⭐ [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — una receta de modelo de razonamiento
  documentada abiertamente.
- [Autoconsistencia](https://arxiv.org/abs/2203.11171), [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
  y [Self-Refine](https://arxiv.org/abs/2303.17651).
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — modelos de recompensa de proceso.
- [Escalar el cómputo en inferencia](https://arxiv.org/abs/2408.03314).
- [STaR](https://arxiv.org/abs/2203.14465), [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
  y [Meta-CoT](https://arxiv.org/abs/2501.04682).

## Módulo 10 — Deep research y despliegue local

- ⭐ [llama.cpp](https://github.com/ggml-org/llama.cpp) — ejecuta modelos cuantizados en CPU y GPU.
- [Documentación de vLLM](https://docs.vllm.ai) y [PagedAttention](https://arxiv.org/abs/2309.06180).
- [GPTQ](https://arxiv.org/abs/2210.17323) y [AWQ](https://arxiv.org/abs/2306.00978) — métodos de
  cuantización de pesos.
- [Decodificación especulativa](https://arxiv.org/abs/2211.17192) y
  [FlashAttention](https://arxiv.org/abs/2205.14135).

## Plantillas y práctica

- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — la idea detrás de nuestra
  [plantilla de *model card*](../templates/model-card.es.md).

---

[🏠 Inicio del curso](../README.es.md) · [Diario de progreso →](progress-journal.es.md)
