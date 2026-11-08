import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from test import hybrid_search

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EVAL = os.path.join(_SCRIPT_DIR, "eval_set.json")


def load_cases(path=DEFAULT_EVAL):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _section_match(expected: str, section: str) -> bool:
    #eval
    if not expected:
        return True
    e, s = expected.lower().strip(), section.lower().strip()
    return e == s or e in s or s in e


def evaluate(alpha: float, k: int = 3):
    cases = load_cases()
    hits = 0
    mrr = 0.0
    details = []

    for c in cases:
        if c.get("must_refuse", False):
            continue

        results = hybrid_search(c["query"], page_filter=None, n_results=k, alpha=alpha)
        rank = None
        for i, (score, section, doc, page) in enumerate(results):
            page_ok = (not c["expected_page"]) or (c["expected_page"].lower() == page.lower())
            section_ok = _section_match(c["expected_section"], section)
            if page_ok and section_ok:
                rank = i + 1
                break
        if rank:
            hits += 1
            mrr += 1.0 / rank
        details.append((c["query"], rank))

    n = sum(1 for c in cases if not c.get("must_refuse", False))
    return {
        "recall@k": hits / n if n else 0.0,
        "mrr": mrr / n if n else 0.0,
        "n": n,
        "details": details,
    }


if __name__ == "__main__":
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        res = evaluate(alpha=alpha, k=3)
        print(f"\nalpha = {alpha}  recall@3 = {res['recall@k']:.3f}  MRR = {res['mrr']:.3f}  (n = {res['n']})")
        misses = [q for q, r in res["details"] if r is None]
        if misses:
            print(f"  Fallos ({len(misses)}):")
            for q in misses:
                print(f"  - {q}")