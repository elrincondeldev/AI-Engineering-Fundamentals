🌐 English · [Español](README.es.md)

# Lab 01 — From a raw web crawl to a Transformer block

> Clean Hearthwise's web crawl and measure it against labels, train a byte-level BPE tokenizer,
> run causal self-attention and gradient descent in numpy, and size real model families.
> ⏱️ about 3 hours.

## 🎯 What you will build

| Command | You build | Lesson |
|---|---|---|
| `uv run lab ngram`, `uv run lab ask` | A counting next-word predictor; base-model continuation vs assistant | [1.1](../01-llm-overview.md) |
| `uv run lab clean`, `uv run lab minhash` | A pre-training cleaning pipeline scored with precision and recall; MinHash near-duplicate detection | [1.2](../02-pretraining-data.md) |
| `uv run lab bpe` | A byte-level BPE tokenizer, and the cost of Spanish in tokens | [1.3](../03-tokenization.md) |
| `uv run lab attention`, `uv run lab train` | Causal multi-head self-attention, a decoder block, RoPE, and gradient descent | [1.4](../04-neural-networks-and-transformers.md) |
| `uv run lab params`, `uv run lab facts` | A parameter, memory, compute and KV-cache calculator, dense and MoE | [1.5](../05-model-families.md) |

Everything runs offline on a CPU in seconds. The only dependency besides the standard library is
`numpy`. The code lives in `src/m01_foundations/`:

| File | What it contains |
|---|---|
| `ngram.py` | `NGramModel` with backoff, `coverage()` |
| `cleaning.py` | `normalise_url`, `html_to_text`, `detect_language`, `quality_signals`, `redact_pii`, decontamination, `run_pipeline`, `evaluate` |
| `dedup.py` | `content_hash`, `shingles`, `MinHasher`, `lsh_candidates`, `UnionFind` |
| `bpe.py` | `pretokenize`, `pair_counts`, `merge_pair`, `train_bpe`, `BPETokenizer` |
| `transformer.py` | `softmax`, `layer_norm`, `causal_mask`, `rope`, `scaled_dot_product_attention`, `decoder_block`, `forward`, `cross_entropy` |
| `neural.py` | `NeuralBigram` with a hand-written gradient, `train_neural_bigram` |
| `params.py` | `ModelShape`, `count_params`, `weight_memory_gb`, `kv_cache_bytes`, `training_flops`, presets |
| `published.py` | Figures quoted from papers, each with its source URL |
| `providers.py`, `data.py` | The course's shared provider layer and dataset loader (copied unchanged) |

**How to use this lab.** Each step has a 🛠️ **Build it** task. The reference implementation is
already in the file so the commands work from the start. To learn it, delete the body of the named
function (keep the signature and docstring), write your own, and run the named test file until it
passes. If you get stuck, `git restore <file>` brings the reference back.

## ⚙️ Setup

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 for you.

```bash
cd modules/01-how-llms-are-built/lab
uv sync
uv run pytest
```

Expected summary (the time varies by machine):

```text
110 passed
```

`uv run lab --help` lists every command.

## 👣 Steps

### Step 1 — Set up and run the tests

Run the setup above, then list the commands:

```bash
uv run lab --help
```

```text
usage: lab [-h] {ngram,ask,clean,minhash,bpe,attention,train,params,facts} ...

Module 01 lab: how LLMs are built.
```

Open `tests/test_cleaning.py` and read two tests. Tests are the specification you will code against
in every Build-it task.

### Step 2 — Predict the next word

```bash
uv run lab ngram
```

```text
Trained a 4-gram counting model on 77 cleaned English pages.

Prompt: 'Lower the setpoint by one degree in the'  (matched the last 3 word(s))
  living       0.364
  bedroom      0.273
  bathroom     0.091
  hallway      0.091
  home         0.091
  greedy continuation: living room and see if anyone notices . a thermostat near the

Prompt: 'The Ember T3'  (matched the last 3 word(s))
  costs        1.000
  greedy continuation: costs € 149 and launched on 2023 - 05 - 22 .

Held-out n-grams already seen in training (61 train / 16 held-out pages):
  n=1: 0.973
  n=2: 0.855
  n=3: 0.735
  n=4: 0.645
  n=5: 0.576
```

The launch date is wrong (the Ember T3 launched on 2023-10-02). Lesson 1.1 explains why. Then compare
a base-style continuation with an assistant reply:

```bash
uv run lab ask
```

```text
Question: How do I reset my thermostat?

Base-model style (4-gram continuation): ...user loopback : i have a 6 m hallway and the sensor misses people at
Assistant via LLM_PROVIDER=fake (fake-1): [fake:81d8b57f] You said: How do I reset my thermostat?
```

🛠️ **Build it:** rewrite `NGramModel.next_word_probs` in `ngram.py` (longest context first, then back
off). Check with `uv run pytest tests/test_ngram.py`.

### Step 3 — Clean the crawl

```bash
uv run lab clean
```

```text
Crawl: 134 pages (128 with HTTP 200, 6 errors; 131 text/html, 3 other types). Labels say keep 83.

Dropped by stage:
  fetch          -9   -> 125
  url_dedup      -4   -> 121
  url_filter     -0   -> 121
  extract        -0   -> 121
  language       -8   -> 113
  quality        -13  -> 100
  exact_dup      -8   -> 92
  near_dup       -8   -> 84
  contamination  -2   -> 82

Kept 82 pages: 48,932 characters of text from 192,610 characters of raw HTML (25.4%).
Languages kept: en=77, es=5
PII redacted on 4 pages: ADDRESS=1, EMAIL=2, PHONE=2

Against labels.jsonl (positive class = keep):
  TP=82  FP=0  FN=1  TN=51
  precision=1.000  recall=0.988  F1=0.994
  dropped but should keep: https://corvid-home.example/thermo-x  (quality:repetitive)

Wrote outputs/clean_corpus.jsonl
```

Now measure two variants and fill the table in lesson 1.2 yourself:

```bash
uv run lab clean --min-words 50
uv run lab clean --skip near_dup
```

Expected key lines: `precision=1.000  recall=0.687  F1=0.814` and
`precision=0.911  recall=0.988  F1=0.948`.

🛠️ **Build it:** rewrite `normalise_url`, `quality_signals` and `redact_pii` in `cleaning.py`. Check
with `uv run pytest tests/test_cleaning.py`, then rerun `uv run lab clean` and confirm the same
numbers.

### Step 4 — Measure near-duplicates with MinHash

```bash
uv run lab minhash
```

```text
MinHash with 128 hash functions; LSH with 16 bands of 8 rows (50% candidate threshold ~ 0.707)

pair                                                        exact J  MinHash
lighting-lighting-scenes-that-actually-get-used               0.969    0.977
lighting-smart-lighting-ideas-for-every-room-guide            0.970    0.969
standby-hidden-energy-what-your-gadgets-use-when-idle-gu      0.969    0.984
lighting-choosing-the-right-colour-temperature-guide          0.968    0.992
lighting-smart-lighting-ideas-for-every-room                  0.969    0.977
standby-a-weekend-project-to-find-vampire-loads-guide         0.973    0.969
standby-how-to-cut-standby-power-at-home                      0.969    0.992
standby-a-weekend-project-to-find-vampire-loads               0.975    0.953

Most similar pair of different articles: J=0.417
  https://techtips.example/heating/comfort-without-waste-in-winter-guide
  https://homeideas.example/heating/comfort-without-waste-in-winter
```

Try `uv run lab minhash --num-perm 32`: the estimates get noisier. Fewer hash functions mean a
cheaper but less precise estimate.

🛠️ **Build it:** rewrite `MinHasher.signature` and `lsh_candidates` in `dedup.py`. Check with
`uv run pytest tests/test_dedup.py`.

### Step 5 — Train a BPE tokenizer

```bash
uv run lab bpe
```

```text
Trained byte-level BPE on 82 documents (48,999 bytes): 767 merges + 256 bytes + 1 special = vocab 1024

First 10 merges:
  #0        '·' + 't'      -> ' t'       (seen 1003 times)
  #1        'h' + 'e'      -> 'he'       (seen 955 times)
  #2        '·' + 'a'      -> ' a'       (seen 900 times)
  #3        '·' + 's'      -> ' s'       (seen 797 times)
  #4        'i' + 'n'      -> 'in'       (seen 672 times)
  #5       '·t' + 'he'     -> ' the'     (seen 657 times)
  #6        'e' + 'r'      -> 'er'       (seen 623 times)
  #7        'o' + 'n'      -> 'on'       (seen 525 times)
  #8        'o' + 'r'      -> 'or'       (seen 506 times)
  #9        '·' + 'c'      -> ' c'       (seen 416 times)

'How long does shipping take?'
  12 tokens: H | ow | ·lon | g | ·does | ·sh | i | pping | ·t | a | ke | ?

'¿Cuánto tarda el envío?'
  19 tokens: <C2> | <BF> | C | u | <C3> | <A1> | nt | o | ·t | ar | d | a | ·el | ·en | v | <C3> | <AD> | o | ?

Special token: 'Hola<|endoftext|>Hello' -> [72, 316, 97, 1023, 72, 744, 111]

Tokens per word on Hearthwise tickets (846 en / 378 es):
   vocab      en      es   es/en
     257   5.566   5.726   1.029
     384   3.864   4.193   1.085
     512   3.519   3.907   1.110
     768   3.139   3.565   1.136
    1024   2.932   3.345   1.141
```

Then rebalance the training data: `uv run lab bpe --es-weight 5`. The last row becomes
`1024   2.993   3.231   1.080`.

🛠️ **Build it:** rewrite `merge_pair`, `train_bpe` and `BPETokenizer._encode_chunk` in `bpe.py`.
Check with `uv run pytest tests/test_bpe.py`. The round-trip tests on emoji and Chinese text are the
ones that catch most bugs.

### Step 6 — Run causal self-attention

```bash
uv run lab attention
```

```text
Text: 'Lower the setpoint by one degree'
Tokens (7): L | ower | ·the | ·setpoint | ·by | ·one | ·degree

Config: vocab=1024 d_model=32 heads=4 d_head=8 layers=2 d_ff=128
  token ids          (7,)
  embeddings x       (7, 32)
  queries per head   (4, 7, 8)   (heads, tokens, d_head)
  scores Q.K^T       (4, 7, 7)
  attention weights  (4, 7, 7)
  logits             (7, 1024)   (one score per vocabulary entry, per position)

Layer 0, head 0 attention weights (row = query token, column = key token):
                L   ower   ·the ·setpo    ·by   ·one ·degre
L            1.00   0.00   0.00   0.00   0.00   0.00   0.00
ower         0.50   0.50   0.00   0.00   0.00   0.00   0.00
·the         0.34   0.34   0.33   0.00   0.00   0.00   0.00
·setpoint    0.24   0.25   0.26   0.25   0.00   0.00   0.00
·by          0.20   0.20   0.20   0.20   0.20   0.00   0.00
·one         0.16   0.16   0.17   0.17   0.17   0.17   0.00
·degree      0.14   0.14   0.14   0.14   0.15   0.15   0.14

Every row sums to 1: True; weights above the diagonal: 0.0
Next-token loss with random weights: 6.905 (uniform guess: ln 1024 = 6.931)
```

🛠️ **Build it:** rewrite `scaled_dot_product_attention`, `rope` and `decoder_block` in
`transformer.py`. Check with `uv run pytest tests/test_transformer.py`. The tests prove causality
end to end and the relative-position property of RoPE.

### Step 7 — Train with gradient descent

```bash
uv run lab train
```

```text
Neural bigram: embedding 1024x32 -> logits 32x1024; 11,204 train / 2,596 validation tokens; lr=10.0

  step    train    valid
     0    6.932    6.932
   200    4.237    4.600
   400    2.832    3.376
   600    2.297    3.036
   800    2.042    2.986
  1000    1.914    2.985

Uniform guess:            6.931
Counting bigram (smoothed): 3.065
```

🛠️ **Build it:** rewrite `NeuralBigram.step` in `neural.py`. `test_gradient_matches_finite_differences`
compares your gradient with a numerical estimate; it is the standard way to debug a backward pass.

### Step 8 — Count parameters, dense and MoE

```bash
uv run lab params
```

```text
preset                         total          active  bf16 GB  KV MB/1k tok
lab-tiny                      57,984          57,984     0.00          0.26
classic-small            124,439,808     124,439,808     0.25         36.86
modern-1b                852,559,872     852,559,872     1.71         32.77
moe-8x-top2            4,728,358,912   1,406,470,144     9.46         32.77
moe-fine-64x-top6      4,868,605,952     854,657,024     9.74         32.77

Breakdown of moe-8x-top2:
  embeddings          65,536,000    1.4%
  positions                    0    0.0%
  attention          167,772,160    3.5%
  feed_forward     4,429,185,024   93.7%
  router                 262,144    0.0%
  norms                   67,584    0.0%
  lm_head             65,536,000    1.4%
  non-embedding    4,597,286,912  (the 12*L*d^2 rule only fits dense MLPs with d_ff = 4*d)
  weights: fp32 18.91 GB, bf16 9.46 GB, int4 2.36 GB
  training compute for 1e+12 tokens ~ 6*N*D = 8.44e+21 FLOPs
  KV cache at 8,192 tokens: 0.268 GB all-global, 0.063 GB with 5 local (1024-token window) per global layer

Published mixture-of-experts sizes (sources: uv run lab facts):
  DeepSeek-V3    total 671B, active 37B per token -> 5.5% active
  Mixtral 8x7B   total 47B, active 13B per token -> 27.7% active
```

Run `uv run lab params --show classic-small` to compare the exact count with Kaplan's rule, and
`uv run lab facts` to see every published figure the lessons quote, with its source.

🛠️ **Build it:** rewrite `count_params` in `params.py`. `test_counter_matches_the_numpy_model_exactly`
checks your counter against the real arrays of the numpy model from step 6.

## 💥 Break it

**1. Stop stripping `utm_` parameters.** In `cleaning.py`, change `TRACKING_PARAMS` to `("fbclid",)`
and run `uv run lab clean`. Does precision change? Which stages now catch the copies?

<details><summary>What you should observe</summary>

Precision and recall do not change (82 kept). `url_dedup` drops 0 pages; `exact_dup` now drops 11
instead of 8 and `quality` drops 14 instead of 13, because the tracking copies have the same text as
their originals and one of them is a spam page. Layered stages give defence in depth, but each later
stage does more expensive work for the same result. Restore the original value.

</details>

**2. Lower the near-duplicate threshold.** The two independent heating guides have J = 0.417. Run
the pipeline with a verification threshold of 0.3 and count the kept pages:

```bash
uv run python -c "from m01_foundations.cleaning import *; print(clean_crawl(PipelineConfig(near_dup_threshold=0.3))[1].kept)"
```

Are the two guides merged?

<details><summary>What you should observe</summary>

Nothing changes: still 82 kept. LSH with 16 bands of 8 rows only proposes pairs as candidates when a
whole band matches, which is likely only above a Jaccard similarity of about 0.707. A pair at 0.417
almost never becomes a candidate, so the verification threshold never sees it. To change what counts
as a near-duplicate you must change the banding too.

</details>

**3. Apply merges in the wrong order.** In `BPETokenizer._encode_chunk`, replace `min(ranked)` with
`max(ranked)` and measure tokens per word on the English tickets.

<details><summary>What you should observe</summary>

Round trips still pass, because decoding only concatenates bytes. But English tickets need 3.360
tokens per word instead of 2.932: late merges grab pieces first and block the frequent merges that
training intended. Correct decoding does not prove correct encoding.

</details>

**4. Mask after the softmax.** In `scaled_dot_product_attention`, compute the softmax first and then
zero the masked weights. Run `uv run pytest tests/test_transformer.py`.

<details><summary>What you should observe</summary>

Two tests fail: `test_attention_never_looks_at_the_future` and `test_forward_is_causal_end_to_end`.
The weights above the diagonal are still zero, yet changing only the last token changes the outputs
of earlier tokens: the softmax normalised every row over the future scores before you zeroed them.
`uv run lab attention` now prints `Every row sums to 1: False`. The fix is to mask the scores with
minus infinity before the softmax. Restore the original code.

</details>

**5. Explode the learning rate.** `uv run lab train --lr 100`.

<details><summary>What you should observe</summary>

The loss is `nan` from step 200 on, and numpy warns about invalid values in `softmax`. Steps that are
too large push the logits to huge values until the arithmetic overflows. Real training runs watch
for `nan` and gradient spikes and stop or roll back.

</details>

## 🚀 Extend it

**1. Fix the false drop.** Make the repetition rule apply only to texts of at least 30 words, then
rerun `uv run lab clean`.

<details><summary>Hint and solution</summary>

In `quality_failures`, guard the repetition check with the word count:

```python
if s["top_trigram_share"] > cfg.max_top_trigram_share and s["words"] >= 30:
    fails.append("repetitive")
```

All spam pages have 45 or more words, so they are still caught. Result: 83 kept,
`precision=1.000  recall=1.000`. Remember the warning from lesson 1.2: you tuned this on the same
labels you score on.

</details>

**2. Measure the redactor.** Write `pii_report(docs, labels)` that compares "the pipeline redacted
something" with the `has_pii` label on kept pages, and prints precision and recall.

<details><summary>Hint and solution</summary>

Among kept pages, 4 are labelled `has_pii`; the redactor touches exactly those 4 and no others, so
precision and recall are both 1.000 on this crawl. That only shows the regexes fit these formats; try
adding a sentence like "write to lucia at mail dot example" to a test to see a miss.

</details>

**3. Pick a language weight with a cost model.** About 30% of Hearthwise tickets are Spanish. Train
with `--es-weight` 1, 3, 5 and 10 and compute a traffic-weighted cost
`0.7 * en + 0.3 * es` tokens per word for each.

<details><summary>Hint</summary>

Reuse `default_tokenizer(vocab_size, es_weight)` from `cli.py` and `ticket_bodies()`. Look for the
weight that minimises the weighted cost, and check how much English gets worse. This is the same
trade-off multilingual model builders make when they set language sampling weights.

</details>

**4. Add grouped-query attention to the numpy model.** Give `TinyConfig` an `n_kv_heads` field, make
`wk` and `wv` project to `n_kv_heads * d_head`, and repeat each key/value head for its group of query
heads.

<details><summary>Hint</summary>

Use `np.repeat(k, n_heads // n_kv_heads, axis=0)` after `split_heads`. Then extend
`test_counter_matches_the_numpy_model_exactly` with `ModelShape(..., n_kv_heads=2)` and check that
`count_params` and the arrays still agree.

</details>

**5. Sample instead of always taking the top word.** Add a `temperature` argument to
`NGramModel.generate` that samples from the next-word distribution with a seeded
`random.Random`.

<details><summary>Hint</summary>

Raise each probability to the power 1/temperature, renormalise, and use `rng.choices`. Low
temperatures behave like greedy decoding; high ones produce more varied (and more wrong) text. Module
02 covers decoding strategies properly.

</details>

## ✅ Definition of Done

- [ ] `uv run pytest` passes.
- [ ] You rebuilt at least one function per step from 2 to 8 and its test file passes.
- [ ] You can explain the one false drop of `uv run lab clean` and the recall of `--min-words 50`.
- [ ] You can explain why `uv run lab bpe` gives Spanish more tokens per word, and what `--es-weight` changes.
- [ ] You can state every shape printed by `uv run lab attention` and why the upper triangle is zero.
- [ ] You can compute total and active parameters of an MoE configuration by hand and match `uv run lab params`.
- [ ] You completed at least two Break-it and one Extend-it items.

## 🧯 Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` | Install uv from [docs.astral.sh/uv](https://docs.astral.sh/uv/) and open a new terminal. |
| `FileNotFoundError: Could not find the repo's datasets/ folder` | Run commands from inside the cloned repository, or set `COURSE_DATASETS_DIR` to the `datasets` folder. |
| Different numbers from the ones shown here | Make sure `datasets/` is unmodified: `python3 datasets/generate.py --check` from the repository root. |
| `RuntimeWarning: invalid value encountered` | Expected with `--lr 100`; the loss has diverged to `nan`. |
| Where did `outputs/` come from? | `uv run lab clean` writes the cleaned corpus there. It is ignored by Git. |

## 🔌 Using a real model

Only `uv run lab ask` talks to a model provider; everything else is self-contained. The command
uses the course's provider layer (`providers.py`), so you can point it at a real assistant with
environment variables:

```bash
# Local model with Ollama (https://ollama.com): start the server and pull a small model first
ollama pull llama3.2:1b
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab ask "How do I reset my thermostat?"

# Any OpenAI-compatible endpoint
LLM_PROVIDER=openai OPENAI_BASE_URL=https://your-endpoint/v1 OPENAI_API_KEY=... OPENAI_MODEL=... \
  uv run lab ask "How do I reset my thermostat?"
```

The base-style line stays the same (it is the counting model); the assistant line comes from the real
model. Expect a step-by-step answer that may or may not match the Ember T3's real reset procedure,
because the model has never seen Hearthwise's help centre. That gap is what modules 05 and 06 close.

Honest status: these live paths were **not run** when this lab was written (no Ollama server was
running and no OpenAI-compatible key was used). The tests only check, offline, that the provider
payloads are built correctly; live tests would be marked `ollama` or `openai` and run only with
`RUN_LIVE_TESTS=1`. If something fails, `uv run pytest
tests/test_providers.py` and the error message from `ProviderError` are the places to start.

---

[← Previous: 1.5 Model families](../05-model-families.md) · [🏠 Module 01](../README.md) · [Next: Quiz →](../quiz.md)
