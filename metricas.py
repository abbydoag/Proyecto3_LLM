import json
import os
import time

from test import hybrid_search
from answer import generate_answer

REFUSAL_PHRASE = "I don't know based on the provided context."

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EVAL = os.path.join(_SCRIPT_DIR, "eval", "eval_set.json")

def load_cases(path=DEFAULT_EVAL):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def is_refusal(text: str) -> bool:
    return REFUSAL_PHRASE.lower() in text.lower()

def keyword_hit(answer: str, expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return False
    answer_l = answer.lower()
    return any(kw.lower() in answer_l for kw in expected_keywords)

def evaluate_full_pipeline(k=3, alpha=0.3):
    cases = load_cases()
    n = len(cases)
    latencies = []
    refusal_correct = 0
    hallucination = 0
    fidelity_hits = 0
    false_refusals = 0
    empty_results = 0

    print(f"Evaluando {n} casos con top_k={k}, alpha={alpha}\n")

    for c in cases:
        query = c["query"]
        must_refuse = c.get("must_refuse", False)
        expected_keywords = c.get("expected_keywords", [])

        t0 = time.time()
        results = hybrid_search(query, page_filter=None, n_results=k, alpha=alpha)
        chunk_texts = [doc for _, _, doc, _ in results]
        if not chunk_texts:
            empty_results += 1
            answer = REFUSAL_PHRASE
        else:
            answer = generate_answer(query, chunk_texts)
        latencies.append(time.time() - t0)

        if must_refuse:
            if is_refusal(answer):
                refusal_correct += 1
            else:
                hallucination += 1
                print(f"  [ALUCINACION] {query}\n    - {answer[:120]}")
        else:
            if expected_keywords and keyword_hit(answer, expected_keywords):
                fidelity_hits += 1
            elif is_refusal(answer):
                false_refusals += 1
                print(f"  [FALSO RECHAZO] {query}")
            else:
                print(f"  [POCO EXITO] {query}\n    - {answer[:120]}")
                if expected_keywords:
                    print(f"    keywords esperadas: {expected_keywords}")

    valid_answers = n - sum(1 for c in cases if c.get("must_refuse", False))
    must_refuse_total = n - valid_answers

    metrics = {
        "n_cases": n,
        "fidelity_rate": fidelity_hits / valid_answers if valid_answers else 0.0,
        "refusal_accuracy": refusal_correct / must_refuse_total if must_refuse_total else 0.0,
        "hallucination_rate": hallucination / must_refuse_total if must_refuse_total else 0.0,
        "false_refusal_rate": false_refusals / valid_answers if valid_answers else 0.0,
        "avg_latency_s": sum(latencies) / len(latencies),
        "empty_retrieval_cases": empty_results,
    }

    print("\n" + "=" * 50)
    print("METRICAS FINALES")
    print("=" * 50)
    for k_, v in metrics.items():
        print(f"  {k_:24} {v}")

    return metrics


if __name__ == "__main__":
    evaluate_full_pipeline(k=3, alpha=0.3)