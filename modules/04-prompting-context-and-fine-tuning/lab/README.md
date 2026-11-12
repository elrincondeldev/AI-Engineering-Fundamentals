🌐 English · [Español](README.es.md)

# Lab 04 — An intent router, adapted three ways

> Route Hearthwise support tickets by intent with zero- and few-shot prompts, a budgeted context
> assembler and LoRA fine-tuning, all scored on an honest temporal split. ⏱️ about 2.5 hours.

## 🎯 What you will build

Alex wants every incoming ticket sent to the right queue among eight intents. Dani wants every claim
backed by an evaluation that cannot fool itself. You will build:

- a **data audit** that finds duplicates, personal data, a leaky column and concept drift;
- an **evaluation harness** with a temporal split, per-slice accuracy (Spanish, camera) and token cost;
- a **prompt builder** with label definitions, few-shot example selection, user context, JSON output,
  validation and retries;
- a **context assembler** that fits a token budget, compacts the text and reports what it dropped,
  plus a prefix-cache simulation;
- **LoRA from scratch in numpy**, compared with full fine-tuning, with a forgetting test and a
  data-size curve;
- four small functions of your own in `src/m04_adaptation/exercises.py`.

The default "LLM" is a **toy in-context learner** (`icl.py`) plugged into the course's `FakeProvider`.
It reads the label definitions and the examples in the prompt and votes by word overlap, so every
answer can be explained and every run is identical. It is not a real LLM: use it to build and debug
the harness, then point the same harness at a real model (see the last section).

| File | What it does |
|---|---|
| `tickets.py` | Load, clean, redact, deduplicate and split the tickets |
| `textfeat.py` | Word stems, Jaccard similarity, a stable hash |
| `icl.py` | The toy in-context learner (a `FakeProvider` responder) |
| `prompts.py` | Label definitions, `PromptConfig`, example selection, message building |
| `structured.py` | Parse and validate JSON answers, retry once with the error |
| `evaluation.py` | Accuracy by slice, majority baseline, single-column leak audit |
| `context.py` | Context items, budgeted assembly, ordering, deduplication, prefix cache |
| `lora.py` | Hashed features, softmax classifier, full fine-tuning, LoRA, parameter counts |
| `reports.py` | Every experiment quoted in the lessons (asserted in `tests/test_documented_numbers.py`) |
| `exercises.py` | Your four functions (they raise `NotImplementedError` until you write them) |
| `providers.py`, `data.py` | Copied unchanged from `tools/lab_skeleton` |

## ⚙️ Setup

From the repository root:

```bash
cd modules/04-prompting-context-and-fine-tuning/lab
uv sync
uv run pytest
```

Expected summary (the time is an example and varies by machine):

```text
81 passed, 5 skipped in 16.18s
```

Four skipped tests belong to the exercises you have not written yet; the fifth is a live-model test
that only runs with `RUN_LIVE_TESTS=1`. List the commands with `uv run lab --help`.

## 👣 Steps

### Step 1 — Audit the tickets

Lesson: [4.1](../01-adaptation-techniques.md). Count duplicates and personal data, look for a column
that knows the answer, and measure drift after the Lookout C2 launch.

```bash
uv run lab audit
```

```text
Ticket audit
------------
rows: 1224 · resubmitted copies: 24 · repeated bodies (any customer): 51
after removing resubmissions: 1200 · temporal split: train 800 (before 2025-09-01) / test 400
personal data and noise in bodies: phone_number 82, email_address 0, signature 205, quoted_reply 142, html_entity 23

Can one column predict the intent? (rule learned on train, scored on test)
--------------------------------------------------------------------------
column           values  accuracy  known when the ticket arrives?
channel               3     0.120  yes
language              2     0.125  yes
product               5     0.125  yes
customer_email      763     0.113  yes
resolution_code       8     1.000  NO  <- leak: filled in after the ticket is solved
majority-class baseline on test: 0.120

Drift after the Lookout C2 launch
---------------------------------
intent                before   after
order_status           0.181   0.120
return_request         0.155   0.095
warranty_claim         0.081   0.087
connectivity_issue     0.139   0.278
billing_question       0.133   0.120
account_access         0.111   0.110
product_question       0.138   0.125
cancel_subscription    0.062   0.065
Lookout C2 share of test tickets: 0.395
unseen word stems vs Jan-Jul vocabulary: August 0.000 · after launch 0.186 · camera tickets 0.465
```

Read `find_resubmissions` and `clean_text` in `tickets.py` and `single_column_accuracy` in
`evaluation.py` before moving on.

### Step 2 — Compare evaluation splits

Lesson: [4.1](../01-adaptation-techniques.md). The same prompt, scored with four evaluation designs
and once with the leaky field pasted into the prompt.

```bash
uv run lab splits
```

```text
Same prompt (bilingual descriptions + 8 similar examples), different evaluation designs
---------------------------------------------------------------------------------------
split                          train  test overlap accuracy
random 80/20, raw rows           979   245      16    0.959
random 80/20, deduplicated       960   240       6    0.921
temporal, raw rows               814   410      13    0.820
temporal, deduplicated           800   400      11    0.828
temporal + resolution_code       800   400      11    0.920
overlap = test tickets whose exact text also appears in train
```

**Your turn (exercise 1).** Implement `flag_leaky_columns(audits, threshold)` in `exercises.py`: return
the columns that are unknown at prediction time or predict the label alone with accuracy at or above
the threshold. Until you write it the test is skipped; once it is correct you see (times vary):

```bash
uv run pytest tests/test_exercises.py -k leaky
```

```text
.                                                                        [100%]
1 passed, 3 deselected in 0.14s
```

<details><summary>Reference solution</summary>

```python
def flag_leaky_columns(audits: list[ColumnAudit], threshold: float = 0.95) -> list[str]:
    return [a.column for a in audits if not a.known_at_prediction_time or a.accuracy >= threshold]
```

</details>

### Step 3 — Read one routing prompt

Lesson: [4.2](../02-prompt-engineering.md). Print every message the router sends for Alex's misrouted
camera ticket (T00843) with two similar examples, plus the model's raw answer.

```bash
uv run lab show-prompt
```

The output starts with the full system message and ends with:

```text
[user]
Ticket: Hey, ¿El Glow A19 es compatible con el wifi de 5 GHz? Thank you!

[assistant]
{"intent": "product_question"}

[user]
Ticket: Buenas, ¿El Glow A19 es compatible con el wifi de 5 GHz? Urgent!

[assistant]
{"intent": "product_question"}

[user]
Ticket: Buenas, La cámara se queda en 'conectando' desde que la pasé a 5 GHz. Thanks.

[model answer]
{"intent": "product_question"}

-> predicted: product_question · true: connectivity_issue · attempts: 1 · prompt tokens: 415
```

Try other flags: `--k 0`, `--selection random`, `--user-context`, `--reasoning`,
`--descriptions names` and `--descriptions bilingual_v2`, and `--ticket` with any test ticket ID.

### Step 4 — Compare prompting strategies

Lesson: [4.2](../02-prompt-engineering.md). Dani's table: one row per prompt variant.

```bash
uv run lab prompts
```

```text
Prompting strategies on the temporal test set (provider: fake)
--------------------------------------------------------------
prompt                                acc  camera spanish   tokens
label names only, 0-shot            0.270   0.228   0.125    130.5
English descriptions, 0-shot        0.618   0.443   0.333    276.1
bilingual descriptions, 0-shot      0.723   0.519   0.683    350.9
bilingual + 1 random example        0.728   0.519   0.692    389.0
bilingual + 4 random examples       0.735   0.525   0.708    504.3
bilingual + 8 random examples       0.728   0.532   0.692    656.8
bilingual + 8 balanced examples     0.740   0.557   0.708    654.3
bilingual + 1 similar example       0.782   0.595   0.750    388.8
bilingual + 4 similar examples      0.828   0.633   0.775    502.6
bilingual + 8 similar examples      0.828   0.633   0.775    654.0
8 similar + user context            0.845   0.677   0.833    790.5
n = 400 tickets · tokens = mean prompt tokens per ticket
```

**Your turn (exercise 2).** Similar selection often returns near-copies of the same ticket. Implement
`select_diverse` with maximal marginal relevance (MMR): prefer candidates that are similar to the
query but not to the examples already chosen.

```bash
uv run pytest tests/test_exercises.py -k diverse
```

```text
.                                                                        [100%]
1 passed, 3 deselected in 0.16s
```

<details><summary>Reference solution</summary>

```python
def select_diverse(selector: ExampleSelector, query: Ticket, k: int, diversity: float = 0.7) -> list[Ticket]:
    from .prompts import user_message
    from .textfeat import jaccard, stems

    candidates = selector.select(query, 4 * k, "similar")[::-1]  # most similar first
    feats = {t.ticket_id: stems(user_message(t, selector.cfg)) for t in candidates}
    q = stems(user_message(query, selector.cfg))
    chosen: list[Ticket] = []
    while candidates and len(chosen) < k:
        def mmr(c: Ticket) -> float:
            redundancy = max((jaccard(feats[c.ticket_id], feats[s.ticket_id]) for s in chosen), default=0.0)
            return (1 - diversity) * jaccard(q, feats[c.ticket_id]) - diversity * redundancy
        best = max(candidates, key=mmr)
        chosen.append(best)
        candidates.remove(best)
    return chosen
```

</details>

### Step 5 — Validate structured output

Lesson: [4.2](../02-prompt-engineering.md). How often does the first answer break the JSON format,
what do a lenient parser and one retry recover, and what does step-by-step prompting cost?

```bash
uv run lab format
```

```text
Structured output on 400 test tickets (bilingual + 8 similar, JSON)
-------------------------------------------------------------------
first answer is strict, valid JSON:        379
first answer readable by a lenient parser: 390
needed a retry:                            10
still invalid after one retry:             0
accuracy without retry: 0.810 · with one retry: 0.828
step-by-step prompt: accuracy 0.828 · completion tokens 9.4 -> 35.7
```

Read `parse_answer` and `classify` in `structured.py`: they are the code you would keep unchanged
when switching to a real model.

### Step 6 — Assemble context under a budget

Lesson: [4.3](../03-context-engineering.md). Fit T00822 into Sam's 500-token budget and read the drop
report, then sweep budgets and compaction policies.

```bash
uv run lab context
uv run lab context --policy raw
uv run lab budget
```

The first command prints the kept and dropped pieces (the full output is in lesson 4.3); its first
lines are:

```text
Context for T00822 · budget 500 tokens · policy clean+dedupe
------------------------------------------------------------
offered 884 tokens · sent 487 tokens · examples kept 4
```

With `--policy raw` the same budget fits a single example:

```text
Context for T00822 · budget 500 tokens · policy raw
---------------------------------------------------
offered 1351 tokens · sent 447 tokens · examples kept 1
```

The budget sweep:

```text
Token budget sweep (16 candidate examples, profile, query; 400 test tickets)
----------------------------------------------------------------------------
budget policy        accuracy examples  tokens
   400 raw              0.790     0.71   388.4
   400 clean            0.797     0.89   390.5
   400 clean+dedupe     0.797     0.86   389.6
   500 raw              0.820     3.21   485.9
   500 clean            0.828     3.50   486.0
   500 clean+dedupe     0.830     3.26   478.3
   700 raw              0.815     8.17   684.4
   700 clean            0.843     8.96   685.1
   700 clean+dedupe     0.830     6.07   583.0
  1000 raw              0.807    14.64   957.4
  1000 clean            0.843    15.67   933.2
  1000 clean+dedupe     0.830     6.28   592.0

More examples are not always better
-----------------------------------
prompt                                acc  camera spanish   tokens
4 similar examples                  0.828   0.633   0.775    502.6
8 similar examples                  0.828   0.633   0.775    654.0
16 similar examples                 0.840   0.665   0.792    955.0
32 similar examples                 0.825   0.639   0.783   1555.6

Tickets with a quoted earlier e-mail (n = 47)
---------------------------------------------
0-shot: raw 0.681 -> cleaned 0.809
8-shot: raw 0.596 -> cleaned 0.830
```

**Your turn (exercise 3).** Implement `truncate_middle(text, max_tokens)`: keep the beginning and the
end of an over-long text and cut the middle, marking the cut with `[...]`.

```bash
uv run pytest tests/test_exercises.py -k truncate
```

```text
.                                                                        [100%]
1 passed, 3 deselected in 0.14s
```

<details><summary>Reference solution</summary>

```python
def truncate_middle(text: str, max_tokens: int, marker: str = " [...] ") -> str:
    from .providers import count_tokens

    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    for keep in range(len(words) - 1, 1, -1):
        head, tail = words[: (keep + 1) // 2], words[len(words) - keep // 2 :]
        candidate = " ".join(head) + marker + " ".join(tail)
        if count_tokens(candidate) <= max_tokens:
            return candidate
    return marker.strip()
```

</details>

### Step 7 — Cache the stable prefix

Lesson: [4.3](../03-context-engineering.md). Replay the 400 test requests through a simulated prefix
cache, once with a stable system prompt and once with a request ID and date on its first line.

```bash
uv run lab cache
```

```text
Prefix cache simulation over the 400 test requests
--------------------------------------------------
stable system prompt     hits 399 · misses   1 · hit rate 0.998 · cached share of prompt tokens 0.496
request id + date first  hits   0 · misses 400 · hit rate 0.000 · cached share of prompt tokens 0.000
```

### Step 8 — Adapt to drift without training

Lesson: [4.4](../04-fine-tuning-peft-and-lora.md). From here on the months are split four ways:
January–July (704 tickets) for the pool or the base model, August (96) as the "old" test, September
(97) for adaptation and October–December (303) as the "new" test.

```bash
uv run lab drift
```

```text
Prompt-side adaptation (old = August, new = October-December)
-------------------------------------------------------------
approach                         labelled    old    new  camera  tokens
0-shot, original descriptions           0  0.760  0.710   0.496   352.7
0-shot, descriptions updated            0  0.760  0.851   0.851   430.9
8 similar, pool Jan-Jul               704  0.917  0.825   0.620   656.0
8 similar, pool Jan-Jul + Sep         801  0.917  0.901   0.810   658.7
```

The updated definitions are `CAMERA_UPDATE` in `prompts.py`. Read them and ask yourself which
October tickets they would miss.

### Step 9 — Fine-tune with LoRA

Lesson: [4.4](../04-fine-tuning-peft-and-lora.md). Train a linear intent classifier on January–July,
freeze it, and adapt it to September with full fine-tuning and with LoRA at ranks 1, 2, 4 and 8.
Numpy results may differ by about one ticket on another machine; the tests allow for that.

```bash
uv run lab lora
uv run lab lora-params
```

```text
Adapting the frozen classifier with 97 September tickets (LoRA: mean of 5 seeds)
--------------------------------------------------------------------------------
model                      trainable  share    old    new  camera  spread
base (frozen, no update)           0  0.000  0.917  0.828   0.628   0.000
full fine-tuning                8200  1.000  0.906  0.927   0.893   0.000
LoRA r=1                        1032  0.126  0.906  0.870   0.770   0.042
LoRA r=2                        2064  0.252  0.904  0.893   0.825   0.046
LoRA r=4                        4128  0.503  0.904  0.905   0.868   0.020
LoRA r=8                        8256  1.007  0.904  0.906   0.874   0.013

Forgetting: train longer on September only (old = August accuracy)
------------------------------------------------------------------
steps  full old  full new  r=4 old  r=4 new replay old replay new
  100     0.906     0.927    0.906    0.901      0.906      0.927
 1000     0.875     0.904    0.885    0.881      0.906      0.914
 4000     0.792     0.855    0.812    0.848      0.865      0.888

How many labelled September tickets? (accuracy on October-December)
-------------------------------------------------------------------
labelled   full  LoRA r=4
      12  0.865     0.867
      25  0.871     0.881
      50  0.911     0.908
      97  0.927     0.905
```

The `lora-params` output is shown in lesson 4.4. Read `train_lora` in `lora.py` next to the maths in
the lesson: every line maps to one equation.

**Your turn (exercise 4).** Implement `lora_logits`: the LoRA forward pass that never builds the
`(C, D)` update matrix. With large matrices this is how adapters run before they are merged.

```bash
uv run pytest tests/test_exercises.py -k lora
```

```text
.                                                                        [100%]
1 passed, 3 deselected in 0.13s
```

<details><summary>Reference solution</summary>

```python
def lora_logits(X, W, b, A, B, alpha):
    r = A.shape[0]
    return X @ W.T + b + (alpha / r) * (X @ A.T) @ B.T
```

`X @ A.T` has shape `(n, r)`, so the low-rank path costs `n·r·(D + C)` multiplications instead of
building a `C × D` matrix first.

</details>

### Step 10 — Write the recommendation

Run the whole suite once more. With your four exercises in place nothing is skipped except the
live-model test:

```bash
uv run pytest
```

```text
85 passed, 1 skipped in 16.40s
```

Then write Alex and Dani a short recommendation (half a page) in your progress journal. Use this
outline and only numbers you produced:

```text
Decision: <prompt only | prompt now + LoRA adapter later | fine-tune now>
Evidence (new = Oct-Dec, old = August, camera = Lookout C2 tickets):
  - best prompt without training: ...
  - LoRA adapter (rank, trainable numbers): ...
  - forgetting check on August: ...
Cost per ticket: prompt tokens ... versus ...
Risks: label noise, drift after the next launch, personal data in tickets
What we monitor and what triggers retraining: ...
```

## 💥 Break it

**1. Take the few-shot examples from the test set.** Run this in `uv run python`. Accuracy jumps from
0.828 to 0.943. Why is that number worthless?

```python
from m04_adaptation.evaluation import evaluate_prompt
from m04_adaptation.reports import BEST_PROMPT, standard_split

train, test = standard_split()
print(evaluate_prompt("leaky pool", test, test, BEST_PROMPT).accuracy)  # pool = test set
```

<details><summary>What is going on</summary>

The most similar example for each ticket is now the ticket itself, labelled with its own answer. The
prompt contains the answer key, exactly like the `resolution_code` leak in lesson 4.1. In production
the pool can only contain tickets solved before the new one arrives. It does not reach 1.000 because
the label definitions still pull some tickets elsewhere.

</details>

**2. Change the order of the intents in the system prompt.** In `system_prompt` (`prompts.py`),
iterate over `reversed(descriptions(cfg.descriptions).items())` and rerun `uv run lab prompts`. The
zero-shot bilingual row moves from 0.723 to 0.755 even though no word of the prompt changed.

<details><summary>What is going on</summary>

When two intents score the same, the toy picks the one listed first, and when nothing matches at all
it falls back to the first intent. Label order is a hidden default. Real models show their own order
and position preferences, so treat the order of labels and examples as a variable to test, not as
decoration.

</details>

**3. Turn the LoRA scale up.** Run this in `uv run python`. Accuracy on October–December collapses to
0.122. Print `abs(model.B).max()` to see why.

```python
from m04_adaptation.lora import train_lora
from m04_adaptation.reports import base_model, lora_data

d = lora_data()
model = train_lora(base_model(), d.X_adapt, d.y_adapt, rank=1, alpha=8)  # scale alpha/r = 8
print(model.accuracy(d.X_new, d.y_new))
```

<details><summary>What is going on</summary>

The update is `(alpha/r)·B·A`, and both `A` and `B` are trained, so a scale of 8 multiplies the
effective step size many times over. Plain gradient descent overshoots, the entries of `A` and `B`
grow without bound, and almost every ticket ends up with the same label. This is why the lab keeps
`alpha = r` (scale 1), and why changing the rank without thinking about alpha and the learning rate
can break training.

</details>

**4. Stop redacting personal data.** In `clean_text` (`tickets.py`), delete the line
`text = redact_pii(text)` and run `uv run pytest`. Which test fails? Do any of the lab's numbers
change?

<details><summary>What is going on</summary>

`test_clean_text_removes_quotes_signatures_entities_and_pii` fails, because a phone number written in
the body of a ticket now survives cleaning. None of the documented numbers change: in this dataset
every phone number sits in a signature, and signatures are already stripped. That is the point. A
privacy control can break without moving any metric, so only a dedicated unit test catches it. Without
it, the next ticket that says "call me on ..." would carry the number into prompts, example pools and
adapter training data, which is exactly what Noor asked you to prevent in module 03.

</details>

## 🚀 Extend it

**1. Build a position probe.** Measure whether the position of the most useful example matters: for
each ticket take seven random examples plus the single most similar one (the "needle"), insert the
needle at position 0 to 7, and compute accuracy per position. Run it with
`ToyInContextLearner(position_bias=0.0)` (the line must be flat) and `position_bias=0.6`.

<details><summary>Hint and solution</summary>

```python
from m04_adaptation.evaluation import make_provider
from m04_adaptation.icl import ToyInContextLearner
from m04_adaptation.prompts import ExampleSelector, answer_message, build_messages, user_message
from m04_adaptation.providers import Message
from m04_adaptation.reports import BEST_PROMPT, standard_split
from m04_adaptation.structured import classify
from m04_adaptation.tickets import INTENTS

train, test = standard_split()
cfg, selector = BEST_PROMPT, ExampleSelector(train, BEST_PROMPT)
provider = make_provider(ToyInContextLearner(position_bias=0.6))
for pos in range(8):
    hits = 0
    for t in test:
        examples = selector.select(t, 7, "random")
        examples.insert(pos, selector.select(t, 1, "similar")[0])
        base = build_messages(t, cfg)  # [system, query]
        msgs = [base[0]]
        for ex in examples:
            msgs += [Message("user", user_message(ex, cfg)), Message("assistant", answer_message(ex.intent, cfg))]
        hits += classify(provider, msgs + [base[-1]], INTENTS).label == t.intent
    print(pos, round(hits / len(test), 3))
```

With `position_bias=0` every position gives the same accuracy, which validates the probe. With the
bias switched on, accuracy changes with position, and the direction may surprise you: at full weight
one similar example sometimes overrides correct label definitions. A probe's job is to detect
sensitivity; run it on a real model to learn that model's shape.

</details>

**2. Let the router abstain.** A misrouted ticket costs more than a ticket sent to human triage.
After each call, read `learner.last_scores` (create `learner = ToyInContextLearner()` and pass it to
`make_provider(learner)`), compute the margin between the best and the second-best score, and send
tickets below a threshold to a human. Report coverage and accuracy on the tickets you kept.

<details><summary>Hint</summary>

Sort `learner.last_scores.values()` in descending order and take `top[0] - top[1]`. Sweep thresholds
such as 0, 0.5, 1 and 2. Accuracy on the kept tickets should rise while coverage falls; the right
threshold is a business decision for Alex (how much triage work is acceptable), not a technical one.

</details>

**3. Sweep the replay size.** In `forgetting_table`, the replay sample has 100 old tickets. Try 0,
25, 50, 100 and 200 at 1,000 steps and plot August against October–December accuracy. How small can
the replay sample be before forgetting returns?

<details><summary>Hint</summary>

Copy the body of `forgetting_table` into a script, loop over the sample size, and keep the seed fixed
so that the only change is the size. Report both test sets for each size: replay that protects August
but hurts October–December is not a free lunch.

</details>

**4. Give LoRA a matrix where it pays off.** Add a hidden layer (1,024 inputs → 64 hidden units →
8 intents) to the classifier, pre-train it, freeze it and apply LoRA to the 64 × 1,024 matrix.
Compare trainable parameters and accuracy with full fine-tuning of both layers.

<details><summary>Hint</summary>

The forward pass is `h = relu(X @ W1.T + b1)`, `logits = h @ W2.T + b2`. Backpropagate the error
signal through `W2` and the ReLU mask to get the gradient for `W1`, then apply the same two lines as
`train_lora` to get the gradients for `A` and `B`. With `d_in = 1,024` and `d_out = 64`, rank 4 trains
`4 · (1,024 + 64)` numbers against 65,536 for the full matrix, so the saving is real this time. Add a
finite-difference gradient check (see `tests/test_lora.py`) before trusting any result.

</details>

## ✅ Definition of Done

- [ ] `uv run pytest` passes, and after your exercises it reports 85 passed and 1 skipped.
- [ ] You implemented the four functions in `exercises.py` without copying the reference solutions.
- [ ] You can explain why the same prompt scores 0.959 on a random split and 0.828 on the temporal one.
- [ ] You can name the leaky column, show the evidence, and list every path by which it could enter
      a prompt.
- [ ] You reproduced Dani's prompt table and can say which variant you would ship and what it costs
      in prompt tokens.
- [ ] You can read the drop report for T00822 and explain each "near-duplicate" and "over budget"
      line.
- [ ] You compared prompt-side adaptation with LoRA on new *and* old tickets, and wrote the Step 10
      recommendation with your own numbers.
- [ ] You completed at least one "Break it" and one "Extend it" item.

## 🧯 Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` | Install uv from [docs.astral.sh/uv](https://docs.astral.sh/uv/), then open a new terminal. |
| `FileNotFoundError: Could not find the repo's datasets/ folder` | Run the lab from inside the repository, or set `COURSE_DATASETS_DIR` to the `datasets` folder. |
| `lab: command not found` | Use `uv run lab ...` from the `lab` folder, so uv activates the environment. |
| LoRA numbers differ from the lesson by about 0.01 | Expected on some machines: a different numpy/BLAS build can flip a borderline prediction. The tests allow 0.011. |
| Accuracy collapses or `RuntimeWarning: overflow encountered` | Training diverged: lower the learning rate or keep `alpha = rank` (see Break it 3). |
| A prompting number differs from the lesson | Prompting results are pure Python and exact. Check that you did not edit `prompts.py`, `icl.py` or the datasets, and that `LLM_PROVIDER` is unset. |
| A test in `test_exercises.py` fails instead of skipping | Your function no longer raises `NotImplementedError` but returns a wrong result. Read the assertion message. |

## 🔌 Using a real model

Everything above runs offline with the toy learner. The provider layer (`providers.py`, copied from
the course skeleton) can also talk to a local model through Ollama or to any OpenAI-compatible
endpoint, and two commands are designed for that:

```bash
# Local model with Ollama (https://ollama.com), after `ollama pull llama3.2:1b`
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab show-prompt --k 4
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab prompts --limit 40

# Any OpenAI-compatible endpoint (a hosted API, vLLM, LM Studio, llama.cpp server)
LLM_PROVIDER=openai OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_MODEL=<model> uv run lab prompts --limit 40

# The optional live test (3 tickets through Ollama)
RUN_LIVE_TESTS=1 uv run pytest -m ollama
```

With a real provider, `prompts` scores only the first N test tickets (40 by default), because 11 prompt
variants times 400 tickets is a lot of calls. The `splits`, `format`, `budget` and `drift` commands
replay hundreds of calls per configuration, so they always use the toy learner and say so.

Be clear about what has and has not been checked:

- **Not tested here.** These real-model paths were not run when this lab was written. Ollama was
  installed but not running, and no API key was used. The commands above are wired through the
  same `providers.py` that the other labs use, and the offline tests check that the request payloads
  are built correctly, but no real model output has been verified.
- **Expect different numbers.** A real LLM understands label names, synonyms and Spanish far better
  than the toy, so zero-shot will be much stronger and the gaps between rows will change. It will
  not slip on the format on a fixed schedule, it may ignore or obey the step-by-step instruction, and
  its scores will vary with the model and its version. That is the point: the harness stays the same
  and tells you what *your* model does.
- **Mind the data.** Prompts contain cleaned, PII-redacted ticket text. Before sending them to an
  external API, check with your privacy lead (Noor, in our story) that this is allowed.
- **LoRA on a real LLM** needs a GPU and a library such as Hugging Face PEFT; it is outside this lab.
  The parameter arithmetic in `lora-params` applies unchanged.

---

[← Previous: 4.4 Fine-tuning, PEFT and LoRA](../04-fine-tuning-peft-and-lora.md) · [🏠 Module 04](../README.md) · [Next: Quiz →](../quiz.md)
