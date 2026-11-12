🌐 English · [Español](README.es.md)

# Lab 09 — Reasoning at answer time and at training time, measured offline

> Run direct, chain-of-thought and thinking answers, then build and measure self-consistency,
> verifiers, Tree of Thoughts, PRM beam search, a STaR loop and a GRPO-style RL toy against a seeded
> simulated reasoner. ⏱️ about 3 hours.

## 🎯 What you will build

Every command talks to the same chat interface (`FakeProvider` from the course's provider layer). By
default, the "model" behind it is a **simulated reasoner**: a seeded program written for this lab, not
a language model. It writes step-by-step solutions to the 160 problems in
`datasets/reasoning/problems.jsonl` in the same format as their reference steps, and makes three kinds
of mistakes on purpose: slips (a wrong number in one step), traps (a consistent misreading of the
question, such as counting Wi-Fi bulbs as hub devices) and lucky answers (right answer, wrong step).
That makes every algorithm observable and every number reproducible on a laptop CPU. The numbers show
how the algorithms behave; they are not scores of any real model.

| File in `src/m09_reasoning/` | What it does | Lesson |
|---|---|---|
| `problems.py` | Loads the problems; exact answer checks; the expression verifier for target puzzles | 9.1 |
| `prompts.py` | The system prompts and message formats (direct, CoT, think, next step, revise, propose, value) | 9.1 |
| `simulator.py` | The simulated reasoner: `SimConfig` knobs, slips, traps, self-checks | 9.1 |
| `reasoner.py` | The client: builds prompts, parses steps and answers, counts tokens (`Usage`) | 9.1 |
| `scaling.py` | Majority vote, best-of-N, sampling, revision loops and feedback sources | 9.2 |
| `verifiers.py` | Arithmetic re-check (exact); simulated ORM and PRM | 9.3 |
| `search.py` | PRM beam search, Tree of Thoughts (BFS and DFS), sampling until verified | 9.3 |
| `training.py` | STaR (with rationalisation and a strict filter); group advantages; GRPO-style toy | 9.4 |
| `experiments.py` | Every experiment behind the numbers in the lessons | all |
| `exercises.py` | Three functions for you to implement | 9.1, 9.2 |
| `cli.py` | `uv run lab <command>` | all |

## ⚙️ Setup

From the repository root:

```bash
cd modules/09-reasoning-models/lab
uv sync
uv run pytest
```

Expected summary (the time varies by machine):

```text
54 passed, 3 skipped
```

The 3 skipped tests check the exercises in step 10; they run as soon as you implement them. No API
key, network or GPU is needed. `uv run lab --help` lists the commands: `solve`, `scale`, `revise`,
`search`, `tot`, `star` and `rl`.

## 👣 Steps

### Step 1 — Set up and run the tests

Run the setup above. Then open `tests/test_simulator.py` and read
`test_without_errors_the_simulator_reproduces_every_reference_solution`. It proves that when the
simulator makes no mistake, it writes exactly the dataset's reference steps for all 160 problems. Every
error you will see later was injected on purpose and recorded.

### Step 2 — Meet the simulated reasoner

```bash
uv run lab solve --id r076
```

```text
r076 · hub_capacity · difficulty 5
A Hearth Hub can connect up to 64 HomeMesh devices. ...
Reference answer: 52

=== direct: 3 tokens (0 thinking), wrong
Answer: 12

=== cot: 85 tokens (0 thinking), wrong
Step 1: HomeMesh devices per room: 1 motion + 1 door + 5 bulbs = 7
Step 2: Rooms: 2 floors x 3 rooms = 6
Step 3: HomeMesh devices: 6 x 7 = 52
Next I need the following quantity.
Step 4: Cameras use Wi-Fi, so they do not count
I will keep two decimals for euro amounts.
Step 5: Free slots: 64 - 52 = 12
Answer: 12
[behind the curtain: step kinds trap, ok, slip, ok, ok]
...
=== think-medium: 329 tokens (254 thinking), correct
```

The `[behind the curtain]` line is the simulator's own record: step 1 fell into the trap (bulbs),
step 3 slipped. Read the think-low and think-medium outputs and find which mistake each one catches.
Then open `simulator.py` and read `SimConfig.slip_probability` and `SimConfig.trap_probability`: the
whole error model is in those two functions and the `write` methods of the four problem families.

### Step 3 — Compare direct, CoT and thinking

```bash
uv run lab solve
```

```text
mode          accuracy  diff 2  diff 3  diff 4  diff 5  tokens/answer  of which thinking
------------  --------  ------  ------  ------  ------  -------------  -----------------
direct           33.1%   44.7%   28.1%   40.0%   22.9%            6.5                0.0
cot              49.4%   57.9%   38.6%   63.3%   45.7%           72.7                0.0
think-low        57.5%   63.2%   49.1%   70.0%   54.3%          138.3               80.5
think-medium     68.1%   65.8%   54.4%   86.7%   77.1%          161.2              103.5
think-high       71.9%   71.1%   56.1%   86.7%   85.7%          194.1              136.4
```

Your turn: in `reasoner.py`, read `parse_solution` and find where thinking tokens are counted. Then
answer for Sam: which column would you put on a cost dashboard, and why not the visible answer length?

### Step 4 — Scale with parallel samples

```bash
uv run lab scale
```

```text
N   majority vote  best-of-N (ORM)  pass@N (oracle)  tokens/problem
--  -------------  ---------------  ---------------  --------------
1           61.8%            61.8%            61.8%            77.3
3           69.1%            82.7%            87.3%           230.9
5           76.4%            86.4%            92.7%           386.2
9           85.5%            86.4%            97.3%           695.9
15          89.1%            91.8%           100.0%         1,165.0
45          89.1%            93.6%           100.0%         3,515.6

At N=45: the most frequent answer is wrong on 12 of 110 problems; 12 of them are trap answers. Problems never solved: 0.
```

Read `majority_vote`, `answer_key` and `best_of_n` in `scaling.py` and `vote_stats` (which explains the
plateau). Your turn: implement `weighted_vote` and `adaptive_vote` in `exercises.py` (step 10).

### Step 5 — Revise with feedback

```bash
uv run lab revise
```

```text
feedback              round 0  round 1  round 2  round 3  tokens/problem  calls/problem  right->wrong
--------------------  -------  -------  -------  -------  --------------  -------------  ------------
self-critique           61.8%    65.5%    69.1%    66.4%           155.9           4.92             9
arithmetic re-check     61.8%    81.8%    85.5%    85.5%           101.6           1.38             0
process reward model    61.8%    80.9%    90.0%    86.4%           152.8           2.27             6
```

Read `revise_loop` and the three feedback functions in `scaling.py`. Notice that every feedback source
returns either `None` (stop) or a sentence naming a step; the loop does not care where it came from.

### Step 6 — Search with verifiers

```bash
uv run lab search --id r076
uv run lab search
```

```text
depth 1: 2 distinct candidates, keep 2
  keep 0.95  HomeMesh devices per room: 1 motion + 1 door = 2
  keep 0.46  HomeMesh devices per room: 1 motion + 1 door + 5 bulbs = 7
...
Answer 52 (correct), 39 calls, 468 generated tokens, 6391 prompt tokens
```

```text
method                         accuracy  generated tok  prompt tok  calls
-----------------------------  --------  -------------  ----------  -----
CoT, 1 sample                     61.8%           77.3       110.4    1.0
majority vote, N=9                85.5%          695.9       993.9    9.0
re-check filter + vote, N=9       86.4%          695.9       993.9    9.0
best-of-9, ORM                    86.4%          695.9       993.9    9.0
best-of-9, PRM                    95.5%          695.9       993.9    9.0
PRM beam search, 3x3              93.6%          274.3     3,055.8   20.5
pass@9 (needs the answer key)     97.3%          695.9       993.9    9.0

Verifier errors over all 4,950 samples (1,854 wrong, 801 of them traps):
verifier               wrong it lets through  right it rejects
---------------------  ---------------------  ----------------
arithmetic re-check                      572               238
ORM (score >= 0.5)                       571               518
PRM (min step >= 0.5)                     64               906

Reward hacking: best-of-N with an honest ORM vs one that also likes long answers (length bias 5.0)
N   honest ORM  length-biased ORM  tokens of pick (honest)  tokens of pick (biased)
--  ----------  -----------------  -----------------------  -----------------------
1        61.8%              61.8%                     77.3                     77.3
3        82.7%              73.6%                     74.1                     79.5
5        86.4%              74.5%                     72.6                     81.4
9        86.4%              77.3%                     73.8                     82.6
15       91.8%              79.1%                     76.4                     86.1
45       93.6%              80.9%                     77.5                     88.4
```

Read `step_is_consistent` in `verifiers.py` (the exact re-check), `SimulatedORM.score` and
`SimulatedPRM.step_scores`, and `prm_beam_search` in `search.py`.

### Step 7 — Tree of Thoughts on the puzzles

```bash
uv run lab tot --id r113
uv run lab tot
```

```text
=== DFS
try [2 4 1] (rated likely)
  prune [2 3] (rated impossible)
...
try [2 12 11/4] (rated likely)
  try [11/4 24] (rated sure)
    no last move reaches the target: backtrack
...
try [2 11 3] (rated likely)
  try [11 1] (rated sure)
found: 11 + ((12 / 4) - 2)  ·  26 calls, 201 generated tokens
```

```text
method                         solved  generated tok  prompt tok  calls  verifier rejections
-----------------------------  ------  -------------  ----------  -----  -------------------
CoT, 1 sample                   11/50           62.1        78.4    1.0                   39
sample until verified, max 5    30/50          218.6       263.7    3.4                  138
sample until verified, max 15   41/50          406.6       485.7    6.2                  268
ToT BFS, breadth 1              27/50           84.6       922.7   11.1                    0
ToT BFS, breadth 3              35/50          152.9     1,657.5   19.9                    0
ToT BFS, breadth 5              36/50          218.6     2,381.9   28.6                    0
ToT DFS with pruning            34/50           87.9     1,032.0   12.4                    0
```

Read `tot_bfs` and `tot_dfs` in `search.py`, then `verify_expression` in `problems.py`. "Verifier
rejections" counts answers the exact verifier refused: every failed sample for the sampling rows,
and false "I reached the target" claims for the tree rows.

### Step 8 — Train with STaR

```bash
uv run lab star
uv run lab star --no-rationalize
uv run lab star --strict
```

```text
round  test accuracy  energy  hub_capacity  order_total  target  kept (solved)  kept (hint)  unfaithful kept  tokens spent
-----  -------------  ------  ------------  -----------  ------  -------------  -----------  ---------------  ------------
0              51.2%   82.4%         38.9%        75.0%   20.0%             40           27                5         8,625
1              68.8%  100.0%         55.6%        80.0%   48.0%             51           21                1         7,508
2              71.2%  100.0%         61.1%        80.0%   52.0%             58           19                3         6,770
3              73.8%  100.0%         66.7%        85.0%   52.0%             56           21                2         7,178
4              72.5%  100.0%         61.1%        85.0%   52.0%              -            -                -             -
```

The last row of the other two runs: without rationalisation the loop ends at 62.5% (target puzzles
28.0%); with the strict filter it ends at 73.8% and keeps no unfaithful rationale. Read
`collect_rationales` and `finetune` in `training.py`. Only `finetune` is simulated; everything else is
the real loop.

### Step 9 — Reinforce with a verifiable reward

```bash
uv run lab rl
```

```text
step  train reward  test accuracy  tokens/answer  P(check step)  P(re-read)  groups with no signal
----  ------------  -------------  -------------  -------------  ----------  ---------------------
0            0.514          53.8%           77.9           0.10        0.10                     10
1            0.530          55.0%           79.8           0.11        0.19                     17
2            0.503          55.0%           81.2           0.09        0.38                     17
3            0.534          60.0%           90.4           0.38        0.70                     18
4            0.606          66.2%          108.8           0.95        0.77                     18
5            0.670          67.5%          112.5           0.98        0.93                     25
6            0.667          67.5%          112.6           0.99        0.93                     18
7            0.658          68.8%          113.0           0.99        0.94                     20
8                -          68.8%          113.0           0.99        0.94                      -
```

Read `group_advantages` and `grpo_toy` in `training.py`. The update is REINFORCE with group-relative
advantages; the real GRPO adds clipping and a penalty for drifting from a reference model. The two
parameters are the logits of "check each step" and "re-read the question".

### Step 10 — Your turn: three exercises

Open `src/m09_reasoning/exercises.py` and implement the three functions. `uv run pytest
tests/test_exercises.py` runs their tests (skipped until each function stops raising
`NotImplementedError`).

1. `pass_at_k(n, c, k)`: the unbiased pass@k estimator (lesson 9.1).
2. `weighted_vote(answers, scores)`: a vote where each sample counts with its verifier score.
3. `adaptive_vote(draw, max_n, lead)`: self-consistency that stops once the leader is safe.

Then use them: compare `adaptive_vote` with `lead=3, max_n=15` against the fixed N = 15 vote on the 110
numeric problems, counting samples used. Use `make_reasoner()` and `lambda: reasoner.solve(p).answer` as
the `draw` function.

<details><summary>Solution: pass_at_k</summary>

```python
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:  # every set of k samples contains a correct one
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
```

</details>

<details><summary>Solution: weighted_vote</summary>

```python
from collections import defaultdict
from m09_reasoning.scaling import answer_key

def weighted_vote(answers, scores):
    totals: dict[str, float] = defaultdict(float)
    first: dict[str, str] = {}
    for answer, score in zip(answers, scores):
        key = answer_key(answer)
        if key is None or answer is None:
            continue
        totals[key] += score
        first.setdefault(key, answer)
    if not totals:
        return None
    best = max(totals.values())
    return next(first[k] for k in first if totals[k] == best)
```

</details>

<details><summary>Solution: adaptive_vote</summary>

```python
from collections import Counter
from m09_reasoning.scaling import answer_key

def adaptive_vote(draw, max_n, lead):
    counts: Counter[str] = Counter()
    first: dict[str, str] = {}
    used = 0
    while used < max_n:
        answer = draw()
        used += 1
        key = answer_key(answer)
        if key is None or answer is None:
            continue
        counts[key] += 1
        first.setdefault(key, answer)
        top = counts.most_common(2)
        runner_up = top[1][1] if len(top) > 1 else 0
        if top[0][1] - runner_up >= lead:
            break
    if not counts:
        return None, used
    best = max(counts.values())
    return next(first[k] for k in first if counts[k] == best), used
```

</details>

## 💥 Break it

Each experiment changes one line. Run the command, compare with the table above, then undo the change
(`git checkout -- src` restores everything). The tests in `test_documented_numbers.py` will fail while
a change is in place: that is expected.

1. **Remove the traps.** In `simulator.py`, set the `order_total`, `energy` and `hub_capacity` values
   of `BASE_TRAP` to `0.0` and run `uv run lab scale`. What happens to the plateau?

   <details><summary>What to observe</summary>

   The plateau disappears: with only independent slips left, majority vote catches up with pass@N.
   Correlated errors were the whole reason the vote stalled.

   </details>

2. **Make self-critique paranoid.** In `SimConfig`, swap the values of `self_detect_slip` and
   `self_false_alarm`, then run `uv run lab revise`.

   <details><summary>What to observe</summary>

   The self-critique row barely moves from round 0 and the "right->wrong" count grows: a reviewer that flags
   correct steps more often than wrong ones sends good work back to be broken. The other two rows do
   not change, because they do not use the model's critique.

   </details>

3. **Turn up the reward hacking.** In `experiments.py`, change `HACK_BIAS` from `5.0` to `10.0`, then
   to `0.0`, and run `uv run lab search` each time.

   <details><summary>What to observe</summary>

   At `10.0` the biased column stays near single-sample accuracy whatever N is: the score now mostly
   measures length. At `0.0` the two columns are identical. Selection pressure goes wherever the
   verifier's preference points.

   </details>

4. **Let hints leak answers.** In `SimConfig`, set `hint_lucky` to `0.9` and run `uv run lab star`,
   then `uv run lab star --strict`.

   <details><summary>What to observe</summary>

   The default run keeps more unfaithful rationales (right answer copied from the hint, wrong steps),
   and the "unfaithful kept" column grows. The strict run still keeps none: checking steps, not only
   answers, is what protects the training data.

   </details>

5. **Groups of one.** Run `uv run lab rl --group 1`.

   <details><summary>What to observe</summary>

   Every group has one sample, so every advantage is zero: all 80 groups show "no signal" and the
   policy never moves. Group-relative methods need several samples per question that disagree.

   </details>

## 🚀 Extend it

1. **Combine verifiers.** Add a method to `run_search` in `experiments.py`: keep only the samples that
   pass `first_bad_step(...) is None`, then pick the best of those with the PRM.

   <details><summary>Hint</summary>

   Copy the `filtered` loop in `run_search`, but instead of `majority_vote` call `best_of_n(kept,
   [prm.score(p, a.steps) for a in kept])`. Add a `row(...)` for it and compare it with "best-of-9, PRM".
   Ask yourself whether the re-check removes anything the PRM would have kept.

   </details>

2. **Weighted vote in the scale table.** Add a column to `run_scale` and `cmd_scale` with your
   `weighted_vote` using ORM scores.

   <details><summary>Hint</summary>

   In the loop over `problems`, compute `weighted_vote([a.answer for a in subset], [orm.score(p, a) for a
   in subset])` and add it to a new field of `ScaleRow`. Expect it between the plain vote and best-of-N:
   it uses the verifier but still rewards frequent answers.

   </details>

3. **Cache the ToT evaluator.** BFS often rates the same set of numbers twice (reached by different
   moves). Add a cache in `tot_bfs` keyed by the sorted values of a state and count the calls saved.

   <details><summary>Hint</summary>

   Keep `seen: dict[tuple[Fraction, ...], str] = {}` inside `tot_bfs`; before calling
   `reasoner.value(...)`, look up `tuple(sorted(v for v, _ in child))`. Compare the "calls" column of
   `uv run lab tot` before and after. With a real model, every saved call is latency and money.

   </details>

4. **Let RL fight traps directly.** Give `grpo_toy` a third parameter per problem family that scales
   the trap probability, and learn it from the trap decisions in `sim.last_trace.trap_decisions`.

   <details><summary>Hint</summary>

   Add a `trap_factor` field to `SimConfig` and multiply it into `trap_probability`. For a trap
   probability `q = q0 * 2 * sigmoid(theta)`, the gradient of the log-probability is `1 - sigmoid(theta)`
   when the trap was taken and `-q * (1 - sigmoid(theta)) / (1 - q)` when it was avoided. Compare the
   tokens column: does the policy still learn to check when it can simply stop falling for traps?

   </details>

## ✅ Definition of Done

- [ ] `uv run pytest` passes (54 tests, plus the 3 exercise tests once implemented).
- [ ] You can explain what the simulated reasoner is, what it is not, and why the lab uses it.
- [ ] You can explain from `lab solve --id r076` which mistake each mode caught or missed.
- [ ] You can explain why majority vote plateaus and what best-of-N needs that voting does not.
- [ ] You can compare the three revision loops, including their right-to-wrong flips.
- [ ] You have traced one ToT search and one PRM beam search and can explain each decision in them.
- [ ] You have run STaR in its three variants and the RL toy, and can explain the zero-signal column.
- [ ] You have implemented at least two exercises and tried at least two Break it experiments.

## 🧯 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `uv: command not found` | Install uv from https://docs.astral.sh/uv/ and open a new terminal. |
| `Could not find the repo's datasets/ folder` | Run the lab inside the course repository, or set `COURSE_DATASETS_DIR` to the `datasets` folder. |
| `The simulator only knows the problems in datasets/reasoning/problems.jsonl` | The simulator only answers the dataset's questions. Use `make_reasoner()` with problems from `load_problems()`. |
| `The simulator can only continue solutions that it wrote itself` | `next_step`, `revise` and `critique` need steps the simulator produced. Do not edit the steps before passing them back. |
| Numbers differ from the lesson tables | You changed a knob in `SimConfig`, `experiments.py` or an algorithm. `git diff src` shows what; `git checkout -- src` restores it. |
| `test_documented_numbers.py` fails after your change | Expected while a Break it or Extend it change is in place. |

## 🔌 Using a real model

The provider layer can send the same prompts to a real model. **This was not run while writing the
lab:** Ollama was installed but not running, and no API key was used, so the commands below are
untested. Treat them as a starting point.

```bash
# Local model with Ollama (https://ollama.com): start the server and pull a model first.
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab solve --provider ollama --mode cot --limit 20
uv run lab solve --provider ollama --id r076 --mode cot

# Any OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp server, a hosted API).
LLM_PROVIDER=openai OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_MODEL=your-model \
  uv run lab solve --provider openai --mode cot --limit 20
```

What to expect and what does not carry over:

- Only `solve` accepts `--provider`. It sends the prompts from `prompts.py` and grades the `Answer:`
  line with the same exact checks; small models often ignore the format, which shows up as wrong
  answers.
- If a model writes its reasoning between `<think>` tags, `parse_solution` counts it as thinking
  tokens. Hosted reasoning models usually hide their reasoning; use the token counts in the provider's
  usage data instead (the `Completion` object keeps the raw response in `raw`).
- The simulated ORM and PRM, the self-critique, step-level beam search, STaR and the RL toy rely on the
  simulator's bookkeeping or on its knobs, so they only work with the default fake provider. To use
  `scaling.py` or `search.py` with a real model, build a `Reasoner(get_provider("ollama"))` and call the
  functions directly; voting, best-of-N with your own scorer, sampling until verified and Tree of
  Thoughts only need the chat interface.

---

[← Previous: 9.4 Training reasoning models](../04-training-reasoning-models.md) · [🏠 Module 09](../README.md) · [Next: Quiz →](../quiz.md)
