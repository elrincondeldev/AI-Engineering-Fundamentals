🌐 English · [Español](model-card.es.md)

# Model card: <model or system name>

> A model card documents what a model or LLM system is for, how it was evaluated and where it fails,
> following the idea in [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993). Fill it
> in for every model or prompt-plus-model system you ship, including fine-tuned adapters.

## Model details

| Field | Value |
|---|---|
| Name and version | |
| Type | *for example: LoRA adapter on an open-weights model, RAG system over a model API* |
| Base model and provider | |
| Owner and contact | |
| Date | |
| License | |

## Intended use

- **Primary use:** *who uses it and for what*
- **Users:** *customers, support agents, analysts...*
- **Out of scope:** *uses it must not be used for (for example medical, legal or payment decisions)*

## Data

| Data | Source | Size | Notes (licence, personal data, known biases) |
|---|---|---|---|
| Training / adaptation | | | |
| Retrieval corpus | | | |
| Evaluation | | | |

## Evaluation

| Metric | Dataset / slice | Result | Launch bar |
|---|---|---|---|
| | Overall | | |
| | *English vs Spanish* | | |
| | *Unanswerable questions* | | |

*Say how results were produced (command, commit, provider and model) so anyone can reproduce them.*

## Limitations and known failure modes

- *For example: facts only documented in Spanish are often missed for English questions.*

## Safety, privacy and ethics

- **Personal data:** *what is redacted, what is stored, for how long*
- **Prompt injection and misuse:** *mitigations and residual risk*
- **Fairness:** *slices checked, gaps found*

## Monitoring and maintenance

*What is monitored in production, who reviews it, and what triggers re-evaluation or rollback.*

---

[🏠 Course home](../README.md) · [Runbook template →](runbook.md)
