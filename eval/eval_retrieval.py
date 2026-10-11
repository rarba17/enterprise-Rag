#!/usr/bin/env python3
"""Retrieval-only eval: HitRate@k / Recall@k / MRR. No LLM judge, $0.

Integration path uses YOUR retriever (app/query.py):
  - similarity_search_with_score (what /query/debug uses)
  - as_retriever(search_type="mmr", k=8, fetch_k=20) (what /query/ask uses)

Usage:
    .venv/bin/python eval/eval_retrieval.py                 # full run, needs DB + OPENAI_API_KEY
    .venv/bin/python -m pytest eval/eval_retrieval.py -v   # pure-metric unit tests (no DB)

Reads:  eval/dataset.json
Writes: eval/retrieval_results.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASET = HERE / "dataset.json"
PUBLIC_DATASET = HERE / "public" / "dataset_public.json"
OUT = HERE / "retrieval_results.json"
PUBLIC_OUT = HERE / "public" / "retrieval_results_public.json"


def resolve_dataset(name: str) -> tuple[Path, Path]:
    if name == "public":
        return PUBLIC_DATASET, PUBLIC_OUT
    return DATASET, OUT


# ---------------------------------------------------------------- pure metrics
def hit_at_k(rank: int | None, k: int) -> int:
    """1 if the gold chunk was retrieved at rank<=k (rank is 1-indexed)."""
    return 1 if (rank is not None and rank <= k) else 0


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(reciprocal_rank(r) for r in ranks) / len(ranks)


def hit_rate_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(hit_at_k(r, k) for r in ranks) / len(ranks)


def find_rank(docs_content: list[str], keyword: str) -> int | None:
    """1-indexed rank of first doc containing keyword (case-insensitive)."""
    kw = keyword.lower()
    for i, text in enumerate(docs_content, start=1):
        if kw in (text or "").lower():
            return i
    return None


# ------------------------------------------------------- integration (needs DB)
def _load_dataset(path: Path = DATASET):
    return json.loads(Path(path).read_text())


def run_integration(search_type: str = "mmr", k: int = 8, fetch_k: int = 20,
                    limit: int | None = None, verbose: bool = True,
                    dataset: Path = DATASET) -> dict:
    """Query the real pgvector store for every dataset item (skips unanswerable)."""
    try:
        from app.query import get_vectorstore
    except Exception as e:
        print(f"SKIP integration: cannot import app.query ({e})")
        return {}
    items = [d for d in _load_dataset(dataset) if d["type"] != "unanswerable"]
    if limit:
        items = items[:limit]
    try:
        vs = get_vectorstore()
    except Exception as e:
        print(f"SKIP integration: cannot connect to vectorstore ({e})")
        print("Hint: docker-compose up -d + OPENAI_API_KEY in .env, docs uploaded + COMPLETED.")
        return {}

    per_item, ranks = [], []
    for d in items:
        try:
            if search_type == "mmr":
                retr = vs.as_retriever(search_type="mmr",
                                       search_kwargs={"k": k, "fetch_k": fetch_k})
                docs = retr.invoke(d["q"])
                texts = [x.page_content for x in docs]
            else:
                scored = vs.similarity_search_with_score(d["q"], k=k)
                texts = [doc.page_content for doc, _ in scored]
        except Exception as e:
            print(f"  [{d['id']}] query failed: {e}")
            continue
        rank = find_rank(texts, d["expected_keyword"])
        ranks.append(rank)
        per_item.append({"id": d["id"], "type": d["type"], "rank": rank,
                         "hit@5": hit_at_k(rank, 5), "hit@8": hit_at_k(rank, 8)})
        if verbose:
            print(f"  [{d['id']}] rank={rank} q={d['q'][:60]}")
    summary = {
        "search_type": search_type, "k": k, "fetch_k": fetch_k, "n": len(ranks),
        "hit@1": round(hit_rate_at_k(ranks, 1), 3),
        "hit@5": round(hit_rate_at_k(ranks, 5), 3),
        "hit@8": round(hit_rate_at_k(ranks, 8), 3),
        "mrr": round(mean_reciprocal_rank(ranks), 3),
        "items": per_item,
    }
    return summary


def compare_modes(**kw) -> dict:
    """Ablation: similarity vs mmr on the same dataset. Proves MMR value."""
    return {
        "similarity": run_integration(search_type="similarity", verbose=False, **kw),
        "mmr": run_integration(search_type="mmr", verbose=False, **kw),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "public"], default="synthetic")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ds_path, out_path = resolve_dataset(a.dataset)
    sim = run_integration(search_type="similarity", dataset=ds_path, limit=a.limit)
    mmr = run_integration(search_type="mmr", dataset=ds_path, limit=a.limit)
    if not sim and not mmr:
        return 2  # skipped, DB unavailable
    report = {"dataset": a.dataset, "similarity": sim, "mmr": mmr}
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n==== Retrieval summary [{a.dataset}] ====")
    for mode, s in report.items():
        if not isinstance(s, dict) or not s:
            continue
        print(f"{mode:10s} n={s['n']} hit@1={s['hit@1']} hit@5={s['hit@5']} "
              f"hit@8={s['hit@8']} mrr={s['mrr']}")
    print(f"\nwrote {out_path}")
    # CI gate: factoid/paraphrase hit@5 on MMR should clear 0.8 on clean synthetic data
    # (public SQuAD data is noisier — gate is advisory there, enforced on synthetic)
    if mmr and a.dataset == "synthetic" and mmr["hit@5"] < 0.8:
        print("GATE FAIL: mmr hit@5 < 0.8 — investigate chunking (500/100) or k/fetch_k.")
        return 1
    print("GATE PASS: mmr hit@5 >= 0.8")
    return 0


# ------------------------------------------------------------- pytest (no DB)
def test_hit_at_k():
    assert hit_at_k(1, 5) == 1 and hit_at_k(6, 5) == 0 and hit_at_k(None, 5) == 0


def test_mrr():
    assert mean_reciprocal_rank([1, 2, None]) == (1 + 0.5 + 0) / 3


def test_find_rank_case_insensitive():
    assert find_rank(["hello World"], "world") == 1
    assert find_rank(["aaa", "has 42,500 here"], "42,500") == 2
    assert find_rank(["aaa"], "zzz") is None


def test_dataset_schema():
    data = _load_dataset(DATASET)
    assert len(data) == 30, f"expected 30 Q, got {len(data)}"
    for d in data:
        assert {"id", "type", "q", "ground_truth", "expected_keyword"} <= set(d)


def test_public_dataset_schema():
    if not PUBLIC_DATASET.exists():
        return  # built by eval/build_public_dataset.py in Phase 2
    data = _load_dataset(PUBLIC_DATASET)
    assert len(data) > 0
    for d in data:
        assert {"id", "type", "q", "ground_truth", "expected_keyword"} <= set(d)


if __name__ == "__main__":
    sys.exit(main())
