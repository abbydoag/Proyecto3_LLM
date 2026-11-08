import sys
import os
import json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import answer
from test import collection, tokenize, model
from rank_bm25 import BM25Okapi

def load_cases(path="eval/eval_set.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

#no embeddings
def evaluate_bm25_only(k=3):
    cases = load_cases()
    hits = 0
    mrr = 0.0
    print("Baseline 1: Solo BM25")

    all_docs = collection.get()
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    bm25 = BM25Okapi([tokenize(d) for d in documents])

    for c in cases:
        if c.get("must_refuse", False):
            continue
        scores = bm25.get_scores(tokenize(c["query"]))
        order = sorted(range(len(documents)), key=lambda i: -scores[i])[:k]
        rank = None
        for r, idx in enumerate(order, 1):
            page = metadatas[idx]["page"]
            section = metadatas[idx]["section"]
            if c["expected_page"].lower() == page.lower():
                e = (c["expected_section"] or "").lower().strip()
                s = section.lower().strip()
                if not e or e == s or e in s or s in e:
                    rank = r
                    break
        if rank:
            hits += 1
            mrr += 1.0 / rank

    n = sum(1 for c in cases if not c.get("must_refuse", False))
    return {"recall@3": hits / n if n else 0.0,
            "mrr": mrr / n if n else 0.0,
            "n": n}

#no contexto
def evaluate_no_rag_llm(num_samples=5):
    cases = load_cases()
    gift_cases = [c for c in cases if c["expected_section"] in ["Love", "Hate", "Dislike"]][:num_samples]

    print(f"\nBaseline 2: LLM sin RAG ({num_samples} preguntas)")

    answer._load_model()

    for c in gift_cases:
        query = c["query"]

        prompt = (
            "You are an expert in Stardew Valley. "
            "Answer the following question using your own knowledge.\n"
            f"Question: {query}\n"
            "Answer:"
        )

        messages = [{"role": "user", "content": prompt}]
        text = answer._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = answer._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(answer._model.device)

        with torch.no_grad():
            out = answer._model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.3,
                pad_token_id=answer._tokenizer.eos_token_id,
            )

        response = answer._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        print(f"\nQuery: {query}")
        print(f"LLM sin RAG: {response[:250]}...")

if __name__ == "__main__":
    bm25_res = evaluate_bm25_only(k=3)
    print(f"\nResultado Baseline BM25:")
    print(f"- Recall@3: {bm25_res['recall@3']:.3f}")
    print(f"- MRR:      {bm25_res['mrr']:.3f}")

    print("\n" + "-" * 70)
    evaluate_no_rag_llm(num_samples=5)