#!/usr/bin/env python3
"""End-to-end answer eval: CPU checks (always) + Groq free-tier judge (optional).

No local LLM, no VRAM. Peak RAM ~100MB without sentence-transformers,
~500MB with it (optional).

Flow per dataset item:
  1. GET /query/ask?question=...      (your RAG: MMR k=8 + gpt-4o)
  2. GET /query/debug?question=...    (retrieved chunks, for faithfulness audit)
  3. CPU score: keyword hit, token-F1 vs ground_truth, ROUGE-L, MiniLM cosine*
  4. Groq judge (if GROQ_API_KEY set): faithfulness 0-1 + correctness 0-1
     (*RAGAS-style, but without the heavy RAGAS framework*)

Usage:
    export GROQ_API_KEY=...            # free tier, judge only (optional but recommended)
    .venv/bin/python eval/eval_answers.py --email you@x.com --password secret
    .venv/bin/python -m pytest eval/eval_answers.py -v   # pure unit tests, no server

Reads:  eval/dataset.json
Writes: eval/answer_results.json, eval/report.md
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASET = HERE / "dataset.json"
RESULTS = HERE / "answer_results.json"
REPORT = HERE / "report.md"
CACHE = HERE / ".judge_cache.json"

BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")
GROQ_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------- CPU metrics
_WS = re.compile(r"\s+")


def normalize(t: str) -> str:
    return _WS.sub(" ", (t or "").lower().strip())


def tokens(t: str) -> list[str]:
    return normalize(t).split()


def token_f1(pred: str, gold: str) -> float:
    pt, gt = tokens(pred), tokens(gold)
    if not pt or not gt:
        return 0.0
    from collections import Counter
    cp, cg = Counter(pt), Counter(gt)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(pt), overlap / len(gt)
    return 2 * p * r / (p + r)


def _lcs(a: list[str], b: list[str]) -> int:
    # memory-efficient LCS for ROUGE-L (short answers, fine)
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1 without any dependency (fallback if rouge-score absent)."""
    pt, gt = tokens(pred), tokens(gold)
    if not pt or not gt:
        return 0.0
    l = _lcs(pt, gt)
    if l == 0:
        return 0.0
    p, r = l / len(pt), l / len(gt)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def keyword_hit(answer: str, keyword: str | None) -> int:
    if not keyword:
        return 1  # unanswerable items score via refusal check instead
    return 1 if keyword.lower() in (answer or "").lower() else 0


def refusal_ok(answer: str, item_type: str) -> int:
    """Unanswerable Qs must trigger the grounded refusal prompt."""
    if item_type != "unanswerable":
        return 1
    a = (answer or "").lower()
    return 1 if ("couldn't find" in a or "could not find" in a or "not in" in a) else 0


def minilm_cosine(a: str, b: str) -> float | None:
    """Optional semantic similarity on CPU (~80MB). None if dep missing."""
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        return None
    global _MINI
    try:
        _MINI
    except NameError:
        _MINI = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    try:
        import importlib

        torch = importlib.import_module("torch")
    except Exception:
        return None
    with torch.no_grad():
        e = _MINI.encode([a, b], convert_to_tensor=True, show_progress_bar=False)
    return float(util.cos_sim(e[0], e[1]).item())


# ------------------------------------------------------------- Groq judge
JUDGE_SYSTEM = (
    "You grade a RAG system. Return STRICT JSON only: "
    '{"faithfulness": 0.0-1.0, "correctness": 0.0-1.0, "reason": "short"}. '
    "faithfulness=1 means every claim in ANSWER is supported by CONTEXT "
    "(0 if it hallucinates or contradicts). "
    "correctness=1 means ANSWER matches GOLD. Unanswerable GOLD expects a refusal."
)


def _cache_load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _cache_save(c: dict) -> None:
    try:
        CACHE.write_text(json.dumps(c))
    except Exception:
        pass


def groq_judge(question: str, answer: str, context: str, gold: str,
               api_key: str | None) -> dict | None:
    """RAGAS-style faithfulness+correctness via Groq free tier. Cached, throttled."""
    if not api_key:
        return None
    key = hashlib.sha256(f"{GROQ_MODEL}|{question}|{answer}|{context[:2000]}".encode()).hexdigest()
    cache = _cache_load()
    if key in cache:
        return cache[key]
    prompt = (f"QUESTION: {question}\nGOLD: {gold}\nCONTEXT:\n{context[:4000]}\n"
              f"ANSWER: {answer}\nScore it.")
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=GROQ_MODEL, temperature=0, max_tokens=300,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content
    except ImportError:
        # fallback without groq package: raw REST
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": GROQ_MODEL, "temperature": 0, "max_tokens": 300,
                  "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                               {"role": "user", "content": prompt}]},
            timeout=60,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  judge call failed ({e}), continuing with CPU scores")
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
        res = {"faithfulness": float(out.get("faithfulness", 0)),
               "correctness": float(out.get("correctness", 0)),
               "reason": str(out.get("reason", ""))[:300]}
    except Exception:
        return None
    cache[key] = res
    _cache_save(cache)
    time.sleep(2)  # respect free-tier RPM (~30/min)
    return res


# ---------------------------------------------------------------- API helpers
def login(email: str, password: str) -> str:
    r = requests.post(f"{BASE_URL}/auth/login",
                      data={"username": email, "password": password}, timeout=15)
    # auto-register on first run for convenience
    if r.status_code in (400, 401, 404):
        requests.post(f"{BASE_URL}/auth/register",
                      json={"email": email, "password": password}, timeout=15)
        r = requests.post(f"{BASE_URL}/auth/login",
                          data={"username": email, "password": password}, timeout=15)
    r.raise_for_status()
    return r.json()["access_token"]


def ask(token: str, question: str) -> str:
    r = requests.get(f"{BASE_URL}/query/ask", params={"question": question},
                     headers={"Authorization": f"Bearer {token}"}, timeout=120)
    r.raise_for_status()
    return r.json().get("answer", "")


def debug_chunks(token: str, question: str) -> list[dict]:
    r = requests.get(f"{BASE_URL}/query/debug", params={"question": question},
                     headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------- runner
def evaluate(email: str, password: str, use_judge: bool,
             limit: int | None = None, no_minilm: bool = False,
             dataset_name: str = "synthetic") -> dict:
    ds_path = HERE / "public" / "dataset_public.json" if dataset_name == "public" else DATASET
    dataset = json.loads(Path(ds_path).read_text())
    if limit:
        dataset = dataset[:limit]
    api_key = os.getenv("GROQ_API_KEY") if use_judge else None
    if use_judge and not api_key:
        print("NOTE: --judge given but GROQ_API_KEY unset → CPU scores only.")
    token = login(email, password)
    print(f"logged in, scoring {len(dataset)} Qs (judge={'on' if api_key else 'off'})")

    rows = []
    for d in dataset:
        try:
            ans = ask(token, d["q"])
            chunks = debug_chunks(token, d["q"])
        except Exception as e:
            print(f"  [{d['id']}] API error: {e}")
            continue
        ctx = "\n\n".join(c.get("content", "") for c in chunks)
        cpu = {
            "keyword_hit": keyword_hit(ans, d.get("expected_keyword"))
            if d["type"] != "unanswerable" else refusal_ok(ans, d["type"]),
            "token_f1": round(token_f1(ans, d["ground_truth"]), 3),
            "rougeL": round(rouge_l(ans, d["ground_truth"]), 3),
        }
        if d["type"] == "unanswerable":
            cpu["refusal_ok"] = refusal_ok(ans, d["type"])
        if not no_minilm:
            s = minilm_cosine(ans, d["ground_truth"])
            cpu["minilm_cosine"] = round(s, 3) if s is not None else None
        judge = groq_judge(d["q"], ans, ctx, d["ground_truth"], api_key) if api_key else None
        rows.append({"id": d["id"], "type": d["type"], "q": d["q"],
                     "ground_truth": d["ground_truth"], "answer": ans,
                     "cpu": cpu, "judge": judge,
                     "n_chunks": len(chunks)})
        j = f" faith={judge['faithfulness']} corr={judge['correctness']}" if judge else ""
        print(f"  [{d['id']}] kw={cpu['keyword_hit']} f1={cpu['token_f1']}{j}")
    return {"model": GROQ_MODEL if api_key else "cpu-only", "dataset": dataset_name,
            "n": len(rows), "rows": rows}


def summarize(res: dict) -> str:
    rows = res["rows"]
    lines = [f"# RAG Eval Report ({res['model']})", "",
             f"Items scored: {len(rows)}", ""]
    for t in ("factoid", "paraphrase", "multihop", "unanswerable"):
        sub = [r for r in rows if r["type"] == t]
        if not sub:
            continue
        kw = sum(r["cpu"]["keyword_hit"] for r in sub) / len(sub)
        f1 = sum(r["cpu"]["token_f1"] for r in sub) / len(sub)
        lines.append(f"## {t} (n={len(sub)}): keyword/refusal rate={kw:.2f}, mean token-F1={f1:.2f}")
        if any(r["judge"] for r in sub):
            jf = sum((r["judge"] or {}).get("faithfulness", 0) for r in sub) / len(sub)
            jc = sum((r["judge"] or {}).get("correctness", 0) for r in sub) / len(sub)
            lines[-1] += f", judge faithfulness={jf:.2f}, correctness={jc:.2f}"
        # worst 2 per type for failure analysis
        worst = sorted(sub, key=lambda r: r["cpu"]["token_f1"])[:2]
        for r in worst:
            lines += ["", f"### {r['id']}: {r['q']}",
                      f"- gold: {r['ground_truth']}",
                      f"- got: {r['answer'][:400]}",
                      f"- cpu: {r['cpu']}" + (f", judge: {r['judge']}" if r["judge"] else "")]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default=os.getenv("EVAL_EMAIL", "eval@example.com"))
    ap.add_argument("--password", default=os.getenv("EVAL_PASSWORD", "evalpass123"))
    ap.add_argument("--judge", action="store_true", help="use Groq free-tier judge (needs GROQ_API_KEY)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-minilm", action="store_true", help="skip sentence-transformers (low RAM)")
    ap.add_argument("--dataset", choices=["synthetic", "public"], default="synthetic")
    a = ap.parse_args()
    try:
        res = evaluate(a.email, a.password, use_judge=a.judge,
                       limit=a.limit, no_minilm=a.no_minilm,
                       dataset_name=a.dataset)
    except requests.ConnectionError:
        print(f"Cannot reach {BASE_URL}. Start stack: docker-compose up -d && uvicorn app.main:app")
        return 2
    out_results = HERE / "public" / "answer_results_public.json" if a.dataset == "public" else RESULTS
    out_report = HERE / "public" / "report_public.md" if a.dataset == "public" else REPORT
    out_results.write_text(json.dumps(res, indent=2))
    out_report.write_text(summarize(res))
    print(f"\nwrote {out_results}\nwrote {out_report}")
    return 0


# ------------------------------------------------------------- pytest (no server)
def test_token_f1_exact():
    assert token_f1("hello world", "hello world") == 1.0
    assert token_f1("hello", "world") == 0.0


def test_rouge_l_partial():
    assert rouge_l("the cat sat", "the cat sat") == 1.0
    assert 0 < rouge_l("the cat sat", "the cat ran") < 1.0


def test_keyword_and_refusal():
    assert keyword_hit("budget is $42,500 ok", "42,500") == 1
    assert refusal_ok("I couldn't find this information in the documents.", "unanswerable") == 1
    assert refusal_ok("$42,500", "unanswerable") == 0


def test_dataset_has_30():
    assert len(json.loads(DATASET.read_text())) == 30


if __name__ == "__main__":
    sys.exit(main())
