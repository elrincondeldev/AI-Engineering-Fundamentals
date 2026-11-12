🌐 English · [Español](CONTRIBUTING.es.md)

# Contributing

Thank you for helping make this course better. Typo fixes, clearer explanations, better exercises,
bug reports in the labs and translation improvements are all welcome.

## Ground rules

- **Be kind and constructive.** Assume good intent and explain the why behind a suggestion.
- **Original work only.** Do not paste text, code or figures from other courses, books or paid
  material. Link to free sources instead.
- **No invented facts.** Statistics, quotes and claims about real models, papers or companies need a
  public source. Fictional Hearthwise facts come only from [`datasets/`](datasets/README.md).
- **English is the source of truth.** Change the English page first, verify it, then update the
  Spanish page in the same pull request.
- Contributions are released under the [MIT License](LICENSE).

The full contract for authors is in [docs/course-conventions.md](docs/course-conventions.md): story
and cast, file names, lesson and lab templates, writing style, sources and language rules.

## How to propose a change

1. Open an issue for anything bigger than a small fix, so we can agree on the approach.
2. Fork the repository and create a branch: `fix/module-05-chunk-overlap`, `feat/module-08-new-exercise`
   or `i18n/module-03-es`.
3. Make the change. Keep pull requests small and focused on one module or one topic.
4. Run the checks below and paste their summary into the pull request.
5. Describe **what** you changed, **why**, and **how you verified it**.

## Checks to run locally

```bash
# The lab you touched (each lab is an independent uv project)
(cd modules/05-rag-retrieval/lab && uv sync && uv run pytest)

# Content checks (standard library only)
python3 datasets/generate.py --check
python3 tools/check_links.py
python3 tools/check_i18n.py
python3 tools/check_navigation.py
python3 tools/check_lessons.py
python3 tools/check_lessons.py --lang es

# Diagrams (needs Node.js 20+)
npm ci --prefix tools
node tools/check_mermaid.mjs
```

CI runs all of them on every pull request. `python3 tools/check_links.py --external` also checks
external URLs; run it when you add links.

## Changing a lesson

- Keep the template order and the emojis: the checkers rely on them.
- Lessons stay between 1,800 and 3,500 words of prose.
- **Every number you quote must come from a real lab run** and be asserted in that lab's
  `tests/test_documented_numbers.py`. If you change lab code and a number moves, update the test,
  the English page and the Spanish page together.

## Changing a lab

- A lab never imports code from another module; copy what you need.
- Tests must pass offline with the fake provider in under a minute on a laptop CPU.
- Add dependencies only when the standard library or `numpy` cannot do the job, and update
  `uv.lock` with `uv lock`.
- Generated files go to `lab/outputs/`, which Git ignores.

## Changing the datasets

Edit the generator in `datasets/_gen/`, run `python3 datasets/generate.py`, then
`python3 datasets/baselines.py`. Update the numbers in `datasets/README.md` (and its Spanish
version) and run every lab's tests, because many of them quote dataset results.

## Translating

- Translate from the verified English page; keep headings, code blocks, mermaid blocks and
  `<details>` blocks one-to-one (`tools/check_i18n.py` checks the counts).
- Code, identifiers and program output stay in English; prose and diagram labels are translated.
- Use neutral Spanish with *tú*, gender-neutral wording for the cast and the learner, and the English
  term in italics and parentheses the first time a technical term appears.

## Reporting a problem

Open an issue with the module and lesson, what you expected, what happened, and for labs the command,
the full output and your operating system.
