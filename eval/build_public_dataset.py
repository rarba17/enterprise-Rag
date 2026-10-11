#!/usr/bin/env python3
"""Build Phase-2 public eval set from SQuAD v2 dev (free, has unanswerables).

Input:  eval/public/dev-v2.0.json  (official SQuAD-explorer mirror)
Output: eval/public/dataset_public.json  (same schema as eval/dataset.json)
        eval/public/contexts.json          (unique contexts for ingestion)

Sampling (deterministic, seed=42): 60 answerable + 20 unanswerable = 80 Q.
Keeps every Q's gold context so ingestion can embed exactly what's needed.

Usage:
    .venv/bin/python eval/build_public_dataset.py [--n-answerable 60 --n-unanswerable 20]
"""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUB = HERE / "public"
SRC = PUB / "dev-v2.0.json"
OUT_DS = PUB / "dataset_public.json"
OUT_CTX = PUB / "contexts.json"
REFUSAL = "I couldn't find this information in the documents."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-answerable", type=int, default=60)
    ap.add_argument("--n-unanswerable", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    raw = json.loads(SRC.read_text())["data"]
    answerable, unanswerable = [], []
    for art in raw:
        title = art.get("title", "")
        for para in art["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                item = {"question": qa["question"], "context": ctx, "title": title,
                        "id": qa["id"], "impossible": qa.get("is_impossible", False),
                        "answers": [x["text"] for x in qa.get("answers", [])]}
                (unanswerable if item["impossible"] else answerable).append(item)
    print(f"parsed: {len(answerable)} answerable, {len(unanswerable)} unanswerable")

    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    picked = answerable[:a.n_answerable] + unanswerable[:a.n_unanswerable]

    dataset, contexts = [], {}
    for i, p in enumerate(picked):
        ctx_id = f"PUBLIC-{abs(hash(p['context'])) % 10**8:08d}"
        contexts.setdefault(ctx_id, {"doc_id": ctx_id, "title": p["title"],
                                     "context": p["context"]})
        if p["impossible"]:
            dataset.append({"id": f"SQ-U{i:02d}", "type": "unanswerable",
                            "q": p["question"], "ground_truth": REFUSAL,
                            "expected_keyword": "couldn't find",
                            "expected_file": None, "squad_id": p["id"],
                            "ctx_id": ctx_id})
        else:
            gold = p["answers"][0] if p["answers"] else ""
            # retrieval check: first 40 chars of gold answer (verbatim in context)
            kw = gold[:40] if len(gold) > 40 else gold
            dataset.append({"id": f"SQ-F{i:02d}", "type": "factoid",
                            "q": p["question"], "ground_truth": gold,
                            "expected_keyword": kw,
                            "expected_file": None, "squad_id": p["id"],
                            "ctx_id": ctx_id,
                            "all_answers": p["answers"]})
    OUT_DS.write_text(json.dumps(dataset, indent=2))
    OUT_CTX.write_text(json.dumps(list(contexts.values()), indent=2))
    n_ans = sum(1 for d in dataset if d["type"] == "factoid")
    n_una = sum(1 for d in dataset if d["type"] == "unanswerable")
    print(f"wrote {OUT_DS} ({len(dataset)} Q: {n_ans} factoid + {n_una} unanswerable)")
    print(f"wrote {OUT_CTX} ({len(contexts)} unique contexts to ingest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
