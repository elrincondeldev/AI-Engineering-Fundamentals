#!/usr/bin/env python3
"""Reference baselines for the shared datasets.

These numbers calibrate difficulty and are quoted in datasets/README.md. They
use only the standard library so anyone can reproduce them::

    python datasets/baselines.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORD = re.compile(r"\w+", re.UNICODE)


def jsonl(rel: str) -> list[dict]:
    return [json.loads(x) for x in (ROOT / rel).read_text(encoding="utf-8").splitlines() if x.strip()]


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


class TextExtractor(HTMLParser):
    """Keep visible article text; drop nav, footer, aside, cookie banners and hidden elements."""

    SKIP = {"nav", "footer", "aside", "script", "style", "header", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.depth_skip = 0
        self.parts: list[str] = []
        self.stack: list[bool] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        skip = tag in self.SKIP or "cookie" in (a.get("class") or "") or "display:none" in (a.get("style") or "")
        skip = skip or (a.get("class") or "") == "ads"
        if tag in ("br", "meta", "img", "input"):
            return
        self.stack.append(skip)
        if skip:
            self.depth_skip += 1

    def handle_endtag(self, tag):
        if tag in ("br", "meta", "img", "input") or not self.stack:
            return
        if self.stack.pop():
            self.depth_skip -= 1

    def handle_data(self, data):
        if self.depth_skip == 0 and data.strip():
            self.parts.append(data.strip())


def html_text(html: str) -> str:
    p = TextExtractor()
    p.feed(html)
    return " ".join(p.parts)


# --------------------------------------------------------------------------- #
# BM25                                                                        #
# --------------------------------------------------------------------------- #


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(map(len, docs)) / len(docs)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out.append(s)
        return out


# --------------------------------------------------------------------------- #
# Baselines                                                                   #
# --------------------------------------------------------------------------- #


def nb_train(rows: list[dict]) -> tuple:
    prior = Counter(r["intent"] for r in rows)
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r["intent"]].update(words(r["body"]))
    vocab = {w for c in counts.values() for w in c}
    return prior, counts, vocab


def nb_predict(model: tuple, text: str) -> str:
    prior, counts, vocab = model
    total = sum(prior.values())
    best, best_lp = "", -math.inf
    for label in sorted(prior):
        c = counts[label]
        denom = sum(c.values()) + len(vocab)
        lp = math.log(prior[label] / total) + sum(math.log((c[w] + 1) / denom) for w in words(text) if w in vocab)
        if lp > best_lp:
            best, best_lp = label, lp
    return best


def acc(model: tuple, rows: list[dict]) -> float:
    return sum(nb_predict(model, r["body"]) == r["intent"] for r in rows) / len(rows)


def tickets() -> dict:
    rows = list(csv.DictReader((ROOT / "tickets/tickets.csv").open(encoding="utf-8")))
    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)
    cut = int(0.8 * len(shuffled))
    rnd_train, rnd_test = shuffled[:cut], shuffled[cut:]
    tmp_train = [r for r in rows if r["created_at"] < "2025-09-01"]
    tmp_test = [r for r in rows if r["created_at"] >= "2025-09-01"]
    majority = Counter(r["intent"] for r in tmp_train).most_common(1)[0][0]
    leak = {}
    for r in tmp_train:
        leak[r["resolution_code"]] = r["intent"]
    cam = [r for r in tmp_test if r["product"] == "Lookout C2"]
    model_t = nb_train(tmp_train)
    bodies = Counter(r["body"] for r in rows)
    return {
        "rows": len(rows),
        "duplicate_bodies": sum(v - 1 for v in bodies.values() if v > 1),
        "majority_class_temporal_test": round(sum(r["intent"] == majority for r in tmp_test) / len(tmp_test), 3),
        "naive_bayes_random_split": round(acc(nb_train(rnd_train), rnd_test), 3),
        "naive_bayes_temporal_split": round(acc(model_t, tmp_test), 3),
        "naive_bayes_temporal_camera_tickets": round(acc(model_t, cam), 3),
        "leaky_resolution_code_temporal": round(sum(leak[r["resolution_code"]] == r["intent"] for r in tmp_test)
                                                / len(tmp_test), 3),
    }


def kb_chunks() -> list[tuple[str, str]]:
    chunks = []
    for meta in jsonl("kb/manifest.jsonl"):
        raw = (ROOT / meta["path"]).read_text(encoding="utf-8")
        if meta["format"] == "markdown":
            body = raw.split("---\n", 2)[2]
            parts = re.split(r"\n(?=#{1,3} )", body)
        else:
            parts = [html_text(raw)]
        for part in parts:
            if part.strip():
                chunks.append((meta["id"], part.strip()))
    return chunks


def rag() -> dict:
    chunks = kb_chunks()
    bm = BM25([words(t) for _, t in chunks])
    by_cat: dict[str, list[int]] = defaultdict(list)
    rr = []
    for q in jsonl("rag_eval/qa.jsonl"):
        if not q["relevant_doc_ids"]:
            continue
        scores = bm.scores(words(q["question"]))
        ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])
        docs: list[str] = []
        for i in ranked:
            if chunks[i][0] not in docs:
                docs.append(chunks[i][0])
        hit = int(any(d in q["relevant_doc_ids"] for d in docs[:5]))
        by_cat[q["category"]].append(hit)
        first = next((k for k, d in enumerate(docs) if d in q["relevant_doc_ids"]), None)
        rr.append(0.0 if first is None else 1 / (first + 1))
    all_hits = [h for v in by_cat.values() for h in v]
    return {
        "chunks": len(chunks),
        "answerable_questions": len(all_hits),
        "bm25_doc_recall_at_5": round(sum(all_hits) / len(all_hits), 3),
        "bm25_doc_mrr": round(sum(rr) / len(rr), 3),
        "recall_at_5_by_category": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_cat.items())},
    }


def preferences() -> dict:
    rows = jsonl("preferences/pairs.jsonl")
    out = {}
    for split in ("train", "test"):
        s = [r for r in rows if r["split"] == split]
        out[f"longer_is_better_{split}"] = round(sum(len(r["chosen"]) > len(r["rejected"]) for r in s) / len(s), 3)
    return out


def shingles(text: str, k: int = 5) -> set[str]:
    w = words(text)
    return {" ".join(w[i : i + k]) for i in range(max(1, len(w) - k + 1))}


def web() -> dict:
    pages = {p["url"]: p for p in jsonl("web/pages.jsonl")}
    labels = {x["url"]: x for x in jsonl("web/labels.jsonl")}
    texts = {u: html_text(p["html"]) for u, p in pages.items() if p["status"] == 200 and p["content_type"] == "text/html"}
    sh = {u: shingles(t) for u, t in texts.items()}

    def jac(a: str, b: str) -> float:
        x, y = sh[a], sh[b]
        return len(x & y) / len(x | y) if x | y else 0.0

    near = [jac(u, lab["near_duplicate_of"]) for u, lab in labels.items() if lab["near_duplicate_of"]]
    linked = {(u, lab["near_duplicate_of"]) for u, lab in labels.items() if lab["near_duplicate_of"]}
    linked |= {(u, lab["duplicate_of"]) for u, lab in labels.items() if lab["duplicate_of"]}
    linked |= {(b, a) for a, b in linked}
    urls = sorted(sh)
    unrelated_max = 0.0
    for i, a in enumerate(urls):
        for b in urls[i + 1:]:
            if (a, b) in linked or not (labels[a]["keep"] and labels[b]["keep"]):
                continue
            unrelated_max = max(unrelated_max, jac(a, b))
    kept_words = sum(len(words(texts[u])) for u, lab in labels.items() if lab["keep"] and u in texts)
    return {
        "pages": len(pages),
        "keep": sum(x["keep"] for x in labels.values()),
        "kept_words": kept_words,
        "near_duplicate_jaccard_min": round(min(near), 3),
        "kept_unrelated_pairs_max_jaccard": round(unrelated_max, 3),
    }


def agent_tasks() -> dict:
    pages = [p for p in jsonl("web/pages.jsonl") if p["status"] == 200 and p["content_type"] == "text/html"]
    texts = [html_text(p["html"]) for p in pages]
    bm = BM25([words(t) for t in texts])
    tasks = jsonl("agent_tasks/tasks.jsonl")
    hits = []
    for t in tasks:
        if not t["gold_urls"]:
            continue
        s = bm.scores(words(t["question"]))
        top = [pages[i]["url"] for i in sorted(range(len(pages)), key=lambda i: -s[i])[:5]]
        hits.append(any(u in t["gold_urls"] for u in top))
    return {"tasks": len(tasks), "answerable": len(hits), "bm25_gold_url_in_top5": round(sum(hits) / len(hits), 3)}


def reasoning() -> dict:
    rows = jsonl("reasoning/problems.jsonl")
    ok = 0
    for r in rows:
        if r["answer_type"] != "expression":
            continue
        used = sorted(int(x) for x in re.findall(r"\d+", r["answer"]))
        value = eval(re.sub(r"(\d+)", r"Fraction(\1)", r["answer"]), {"Fraction": Fraction})  # noqa: S307 (trusted)
        ok += used == r["numbers"] and value == r["target"]
    return {
        "problems": len(rows),
        "by_type": dict(sorted(Counter(r["type"] for r in rows).items())),
        "by_difficulty": dict(sorted(Counter(r["difficulty"] for r in rows).items())),
        "target_solutions_verified": ok,
    }


def main() -> None:
    report = {"tickets": tickets(), "rag": rag(), "preferences": preferences(), "web": web(),
              "agent_tasks": agent_tasks(), "reasoning": reasoning()}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
