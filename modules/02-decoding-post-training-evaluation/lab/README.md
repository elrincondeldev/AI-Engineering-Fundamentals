🌐 English · [Español](README.es.md)

# Lab 02 — Decoding, post-training data and evaluation from scratch

> Build every decoding strategy on a Hearthwise language model, clean an SFT dataset, train a reward
> model that falls into the length trap, and evaluate models with metrics, a contamination detector, a
> leaderboard and a judge audit. ⏱️ about 3 hours.

## 🎯 What you will build

| File | What you build | Lesson |
|---|---|---|
| `src/m02_post_training/ngram.py` | A trigram language model that returns one log-probability per vocabulary entry | 2.1, 2.4 |
| `src/m02_post_training/decoding.py` | Softmax with temperature, greedy, beam search, top-k, top-p, penalties, stop sequences, seeds, diversity metrics | 2.1 |
| `src/m02_post_training/sft.py` | A chat template, a loss mask and a data cleaner scored against labels | 2.2 |
| `src/m02_post_training/reward.py` | A verifiable reward, a Bradley–Terry reward model, a hack test, the DPO loss and the KL-penalised reward | 2.3 |
| `src/m02_post_training/evaluation.py` | EM, F1, BLEU, ROUGE-L, perplexity, a contamination detector, Elo and Bradley–Terry ratings, a judge swap test | 2.4 |
| `src/m02_post_training/cli.py` | The `uv run lab <command>` entry point | all |

`providers.py` and `data.py` are the shared course files, copied unchanged. Everything runs offline on a
CPU with a fixed seed, so your numbers must match the ones below exactly.

**How to work through it.** The package contains a complete reference implementation, so the tests pass
from the start. Each step names the functions to rebuild. For each one: read the tests that cover it,
delete the function body (keep the signature and docstring), write your own version, and run the tests
and the step command until the output matches again. If you get stuck, `git diff` shows what you
changed and `git checkout -- <file>` brings the reference back.

## ⚙️ Setup

```bash
cd modules/02-decoding-post-training-evaluation/lab
uv sync
uv run pytest
```

Expected summary (the time is an example and varies by machine):

```text
........................................................................ [ 80%]
.................                                                        [100%]
89 passed in 8.09s
```

`uv run lab --help` lists every command.

## 👣 Steps

### Step 1 — Train the base model and look at its logits

Rebuild `NgramLM.next_logprobs` in `ngram.py` and `softmax` in `decoding.py`. The model mixes trigram,
bigram and word-frequency estimates with weights 0.6, 0.3 and 0.1; when a context was never seen, its
weight goes back to the lower orders.

```bash
uv run lab corpus
uv run lab next-token --prompt "to reset the"
```

```text
documents: 95 (17 help-centre pages, 78 web pages)
training tokens (including <eos>): 12544
vocabulary (words seen at least twice + 3 special tokens): 797
prompt: 'to reset the'
T=0.5  'device' 0.996  'router' 0.000  'app' 0.000  'thermostat' 0.000  'home' 0.000
T=1.0  'device' 0.612  'router' 0.011  'app' 0.009  'thermostat' 0.009  'home' 0.008
T=2.0  'device' 0.063  'router' 0.009  'app' 0.008  'thermostat' 0.007  'home' 0.007
tokens kept by top-p (p=0.5: 1, p=0.9: 79) out of 797
```

### Step 2 — Compare decoding strategies

Rebuild `top_k_filter`, `top_p_filter`, `apply_repetition_penalty`, `beam_search`, `distinct_n` and
`repetition_rate` in `decoding.py`. Then trigger a greedy loop, break it, and run the benchmark.

```bash
uv run lab generate --strategy greedy --prompt "returns are" --max-tokens 50
uv run lab generate --strategy greedy --prompt "returns are" --max-tokens 50 --repetition-penalty 1.3
uv run lab generate --top-p 0.9 --seed 7
uv run lab decode-bench
```

```text
returns are -> hard to debug. avoid bright cold light late in the app with the appliance, not the lamp, so voice commands stay simple. warm white works best in the app with the appliance, not the lamp, so voice commands stay simple. warm white works
[finish_reason=length tokens=50 logprob=-29.91 rep-3=0.38]
returns are -> hard to debug. avoid bright cold light late in the app with the appliance, not the lamp, so voice commands stay simple. warm white works best in the bedroom and see if anyone notices. a smart plug can switch on when the last person leaves
[finish_reason=length tokens=50 logprob=-34.15 rep-3=0.00]
to reset the -> device to support 5 ghz wi-fi) battery sensors last longer than wi-fi ones. name your gadgets a week before adding more; future you will thank you. how do? tap forgot password on. open the
[finish_reason=length tokens=40 logprob=-48.08 rep-3=0.00]
6 prompts x 5 seeds, up to 50 new tokens each
strategy               distinct-2  rep-3  unique  logp/tok  ended
greedy                      0.130  0.062    6/30     -0.78   0.83
beam (4 beams)              0.140  0.000    6/30     -0.85   1.00
sample T=1.0                0.540  0.021   30/30     -2.00   0.03
sample T=0.7                0.532  0.006   30/30     -1.12   0.40
sample T=1.5                0.623  0.000   30/30     -5.77   0.40
top-k k=10                  0.616  0.008   30/30     -1.26   0.27
top-p p=0.9                 0.635  0.001   30/30     -1.38   0.37
top-p 0.9 + rep 1.3         0.655  0.003   30/30     -1.32   0.23
```

Run the seed-7 command twice: the text must be identical. Then try `--seed 8`.

### Step 3 — Render chat templates and loss masks

Rebuild `render_chat` and `loss_mask` in `sft.py`.

```bash
uv run lab chat-template --id s002
```

```text
<|system|>
You are Hearthwise's support assistant. Answer briefly, accurately and kindly.<|end|>
<|user|>
How many devices can the Hearth Hub handle? Thanks.<|end|>
<|assistant|>
The Hearth Hub connects up to 64 HomeMesh devices.<|end|>
--- loss mask ([x] = trained) ---
<|system|> You are Hearthwise ' s support assistant . Answer briefly , accurately and kindly . <|end|> <|user|> How many devices can the Hearth Hub handle ? Thanks . <|end|> <|assistant|> [The] [Hearth] [Hub] [connects] [up] [to] [64] [HomeMesh] [devices] [.] [<|end|>]
tokens: 42  trained: 11  (26%)
```

### Step 4 — Clean the SFT data

Rebuild `is_low_quality`, `has_pii`, `ContaminationIndex.matches` and `clean` in `sft.py`.

```bash
uv run lab sft-clean
```

```text
rows: 261  kept: 108  dropped: 153
check                    labelled  flagged  correct precision  recall
contaminated_rag_eval           5       11        5      0.45    1.00
empty_response                  1        1        1      1.00    1.00
low_quality                     1        1        1      1.00    1.00
outdated_policy                 4        4        4      1.00    1.00
pii_in_response                 3        3        3      1.00    1.00
wrong_language                  1        1        1      1.00    1.00
duplicate                dropped 132 exact repeats; labelled duplicates kept exactly once: 6/6
kept rows that still ask an eval question word for word: 5
distinct responses among kept rows: 25
tokens before cleaning: 13874 (42% trained)  after: 5969 (42% trained)
```

Then open the 6 rows that the contamination check flags but the labels do not, and decide who is right.

### Step 5 — Train a reward model and catch the length trap

Rebuild `fit_reward_model`, `pairwise_accuracy`, `prompt_disjoint` and `hack_rate` in `reward.py`.

```bash
uv run lab reward
```

```text
'longer answer wins' baseline: train 0.787  test 0.233
padding used for the hack test: 'Thank you so much for reaching out to Hearthwise, we really appreciate you taking the time to contact us.'

official split (same prompts in train and test)
reward model               pairs feats  train   test   hack
length only                  240     1  0.787  0.233  0.767
words + length               240   302  1.000  1.000  0.000

new prompts (train and test questions do not overlap)
reward model               pairs feats  train   test   hack
length only                  125     1  0.784  0.250  0.750
words + length               125   217  1.000  0.714  0.357
words + length, balanced      54   217  1.000  0.821  0.250

words + length (new prompts), largest weights:
  + or 0.76, <length> 0.76, 30 0.48, orders 0.47, no 0.47
  - yes -0.62, 5 -0.38, works -0.38, 14 -0.37, 1-year -0.31
```

### Step 6 — Compute DPO and PPO quantities

Rebuild `verifiable_reward`, `dpo_loss` and `kl_penalized_reward` in `reward.py`.

```bash
uv run lab verify
uv run lab dpo
```

```text
r001: A customer buys 3 x Spark Mini at €19 each, 1 x Ember T3 at €149 each. A 15% discount code applies to the products (not to shipping). Shipping is free when the products cost more than €49 after any discount; otherwise it costs €4.95. What is the total in euros?
gold answer: 175.10
reward 1.0 <- '206 x 0.85 = 175.10, shipping is free, so the total is 175.10 euros.'
reward 1.0 <- 'The total is €175.1'
reward 0.0 <- 'Subtotal 206, minus 15% is 175.10, plus €4.95 shipping = 180.05'
reference model: log p(chosen) = -20.0, log p(rejected) = -18.0
policy scenario                            margin loss b=0.1 loss b=0.5
policy = reference (start of training)        0.0      0.693      0.693
chosen up 1, rejected down 1                  2.0      0.598      0.313
chosen down 2, rejected down 5                3.0      0.554      0.201
chosen down 1, rejected up 1 (wrong way)     -2.0      0.798      1.313

PPO-style reward: r=2.0, log pi=-12.0, log pi_ref=-15.0, beta=0.1 -> 1.70
```

### Step 7 — Score answers with traditional metrics

Rebuild `token_f1`, `bleu` and `rouge_l` in `evaluation.py` and `NgramLM.perplexity` in `ngram.py`.

```bash
uv run lab metrics
```

```text
reference answer (rag_eval q001): '30 days from delivery.'
candidate                  correct    EM    F1  BLEU ROUGE-L
copy of the reference          yes  1.00  1.00  1.00    1.00
correct paraphrase             yes  0.00  0.46  0.18    0.53
correct, different words       yes  0.00  0.36  0.14    0.40
wrong (old policy)              no  0.00  0.75  0.75    0.80
correct but verbose            yes  0.00  0.33  0.14    0.34

perplexity of a trigram model trained on 90% of the corpus
text                       tokens  <unk> perplexity
training documents          11043     4%       2.54
held-out documents           1406     6%       7.01
greedy outputs                219     0%       2.66
top-p 0.9 outputs             128     0%       9.80
T=1.5 outputs                  97     0%     674.91
Spanish help-centre pages     266    80%      20.30
```

### Step 8 — Detect benchmark contamination

Rebuild `item_in_page` and `find_contamination` in `evaluation.py`.

```bash
uv run lab contamination
```

```text
web pages labelled as contaminated: 2; eval questions copied word for word: 12
  n share pages flagged correct items flagged correct
  3   0.5             2       2            46      12
  8   0.5             2       2            45      12
  8   1.0             2       2            12      12
 13   1.0             2       2            12      12
https://studyhub.example/math-practice-with-solutions: r001, r002, r003, r004, r005, r006
https://studyhub.example/hearthwise-support-quiz-answers: q001, q002, q003, q004, q005, q006
```

### Step 9 — Build a leaderboard and audit a judge

Rebuild `elo_ratings`, `bradley_terry_ratings` and `judge_swap_test` in `evaluation.py`. The votes are
simulated from hidden strengths so you can check that a method recovers them; the judge is a scripted
stand-in with a built-in length preference and a five-word bonus for the answer shown first.

```bash
uv run lab arena
uv run lab judge
```

```text
600 simulated votes between 4 models
model     truth    Elo Elo rev     BT  BT 95% interval
model-A    1113   1105    1087   1105     1075 to 1146
model-B    1026   1021    1078   1026      997 to 1061
model-C     991    986     985    993      959 to 1022
model-D     870    887     850    875      844 to  903
Elo ranking:          model-A > model-B > model-C > model-D
Elo ranking reversed: model-A > model-B > model-C > model-D
BT ranking:           model-A > model-B > model-C > model-D
judge provider: fake
pairs                          n agree AB agree BA consistent  first longer
test pairs as rated           60    0.233    0.233      1.000  0.500  1.000
test pairs, filler removed    60    1.000    0.667      0.667  0.667  0.833
```

## 💥 Break it

1. **Remove context from the model.** In `cli.build_lm`, create `NgramLM(lambdas=(1.0, 0.0, 0.0))` and
   run `uv run lab generate --strategy greedy --max-tokens 12`. What does a "language model" that only
   knows word frequencies say?

   <details><summary>What you should see</summary>

   A row of full stops. With only word frequencies, the most common token is always the most likely
   next token, so greedy decoding picks it forever. Context is what turns frequencies into language.
   Put the default lambdas back afterwards.

   </details>

2. **An off-by-one in top-p.** In `top_p_filter`, remove the `+ 1` from `n_keep` and run
   `uv run lab generate --top-p 0.5 --seed 1`.

   <details><summary>What you should see</summary>

   A crash: `ValueError: Probabilities contain NaN`. After "to reset the", "device" alone already passes 0.5,
   so the search position is 0 and the filter keeps zero tokens. Every logit becomes minus infinity and
   softmax divides zero by zero. The tests `test_top_p_keeps_smallest_set_reaching_p` catch this too.

   </details>

3. **Forget to train the end marker.** In `loss_mask`, append 0 instead of 1 for `END` inside an
   assistant turn and run `uv run pytest tests/test_sft.py`.

   <details><summary>What you should see</summary>

   `test_loss_mask_trains_only_assistant_tokens_and_end_marker` fails. A model trained with this mask
   never practises emitting `<|end|>`, so it would not know when to stop: the rambling from lesson 2.1.

   </details>

4. **Hide the length feature.** In `reward._run`, build the featurizer with `use_length=False` and run
   `uv run lab reward`. Does the hack rate on new prompts drop to zero?

   <details><summary>What you should see</summary>

   No. Test accuracy and hack rate barely move, because the filler words ("thank", "appreciate",
   "wonderful") appear almost only in long answers and carry the same bias. Removing the explicit
   length feature does not remove length bias; balancing the data does more.

   </details>

5. **Remove the judge's position bonus.** In `cli.cmd_judge`, pass `position_bonus=0` to
   `biased_judge_responder` and rerun `uv run lab judge`.

   <details><summary>What you should see</summary>

   On the filler-free pairs, consistency rises sharply and the "picked first" rate moves close to 0.5.
   A small position effect remains, because ties in length still go to answer A (`>=`). That is the
   swap test doing its job: it measures exactly the bias you planted.

   </details>

## 🚀 Extend it

1. **Add a min-p filter.** Keep only tokens whose probability is at least `m` times the probability of
   the most likely token. Add it to `DecodeConfig` and `generate`, add a strategy to `BENCH_STRATEGIES`,
   and compare it with top-p in `decode-bench`.

   <details><summary>Hint and solution</summary>

   ```python
   def min_p_filter(logits: np.ndarray, m: float, temperature: float = 1.0) -> np.ndarray:
       """Keep tokens with probability >= m * (largest probability)."""
       if m <= 0:
           return logits
       probs = softmax(logits, temperature)
       return np.where(probs >= m * probs.max(), logits, NEG_INF)
   ```

   Unlike top-p, min-p ignores how the tail adds up, so a long flat tail is cut even when it carries a
   lot of mass in total.

   </details>

2. **Add a length penalty to beam search.** Choose the final beam by `score / len(seq) ** alpha` instead
   of `score`, and see how the beam answer to "to reset the" changes for `alpha` = 0.5 and 1.0.

   <details><summary>Hint</summary>

   Only the final `max(pool, ...)` needs to change, plus a new `length_penalty` field in `DecodeConfig`.
   Guard against empty sequences (`max(len(seq), 1)`).

   </details>

3. **Deduplicate by response only.** Add a `dedup_by_response` option to `sft.clean` and count how many
   rows survive. Is a paraphrased greeting ("Hi! ...", "... Thanks.") a new example or a duplicate?

   <details><summary>Hint</summary>

   Use `" ".join(tokenize(row["response"]))` as the key. The answer to the question is a judgement call:
   greeting variants teach robustness to phrasing but repeat the same target text. Write your decision
   in a comment.

   </details>

4. **How many votes are enough?** Call `evaluation.simulate_votes(n_votes=...)` with larger counts and
   find the smallest count for which the bootstrap intervals of model-B and model-C stop overlapping.

   <details><summary>Hint</summary>

   Try doubling: 600, 1200, 2400. With 2,400 votes the two intervals no longer overlap. This is the
   calculation to do before you ask Alex's agents for their time.

   </details>

5. **Count only consistent verdicts.** Add a field to `JudgeReport` with the agreement computed only over
   pairs where both orders gave the same winner, and the share of pairs that were dropped.

   <details><summary>Hint</summary>

   Keep two counters inside the loop: one for consistent pairs, one for consistent pairs whose winner is
   the chosen answer.

   </details>

## ✅ Definition of Done

- [ ] `uv run pytest` passes with your own implementations of the functions named in steps 1 to 9.
- [ ] Every step command prints exactly the expected output above.
- [ ] You can explain the greedy loop and how the repetition penalty breaks it.
- [ ] You looked at the 6 unlabelled contamination hits in the SFT data and can say why they are leaks.
- [ ] You can explain why the reward model's 1.000 test accuracy on the official split is misleading.
- [ ] You completed at least two "Break it" experiments and one "Extend it" task.
- [ ] You wrote down which evaluation you would use for Hearthwise's model choice, and why.

## 🧯 Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Install uv from the [official documentation](https://docs.astral.sh/uv/). |
| `FileNotFoundError: Could not find the repo's datasets/ folder` | Run the commands from inside the repository, or set `COURSE_DATASETS_DIR` to the `datasets` folder. |
| Numbers differ slightly from this page | Check that you did not change a default (seeds, lambdas, thresholds) and that `uv sync` installed the locked versions. Restore a file with `git checkout -- <file>` to compare. |
| `ValueError: Probabilities contain NaN` | A filter removed every token; see "Break it" item 2. |
| `RuntimeError: Network access is disabled in tests` | A test tried to reach a server. Only tests marked `ollama` or `openai` may do that, and only with `RUN_LIVE_TESTS=1`. |

## 🔌 Using a real model

Most of this lab is about algorithms you can only see inside a small model, so it needs no real model.
The one place where a real model makes sense is the **judge** in step 9: `lab judge` sends its prompts
through `providers.py`, so it can call a local or hosted model instead of the scripted stand-in.

```bash
# Local model with Ollama (https://ollama.com)
ollama pull llama3.2:1b
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab judge

# Any OpenAI-compatible endpoint
LLM_PROVIDER=openai OPENAI_BASE_URL=https://your-endpoint/v1 OPENAI_API_KEY=... OPENAI_MODEL=... uv run lab judge
```

The judge sends 240 requests (60 pairs, two orders, two sets of pairs), so a hosted endpoint will cost
money. A small local model may not answer with a single letter; `parse_verdict` then returns `None`,
which the report counts as a vote for answer B. Look at a few raw replies before trusting the numbers.

You can also send the decoding settings from lesson 2.1 to a real model. `providers.py` passes
temperature, max tokens, seed and stop sequences; Ollama's options also accept `top_k`, `top_p` and
`repeat_penalty`, which you can add to `OllamaProvider.build_chat_payload` as an extension.

**Not tested:** the Ollama and OpenAI-compatible paths were not run when this lab was written. Only the
offline provider contract tests in `tests/test_providers.py` run in CI; no test in this lab calls a live
model.

---

[← Previous: 2.4 Evaluating LLMs](../04-evaluating-llms.md) · [🏠 Module 02](../README.md) · [Next: Quiz →](../quiz.md)
