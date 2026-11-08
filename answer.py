import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_tokenizer = None
_model = None

def _load_model():  
    global _tokenizer, _model
    if _model is None:
        print(f"Cargando modelo {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        _model.eval()
        print("Modelo cargado...")

def _clean_chunk(text: str) -> str:
    text = re.sub(r"\s*\|\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_prompt(query: str, chunks: list[str]) -> str:
    if not chunks:
        context = "(no context retrieved)"
    else:
        context = "\n\n".join(
            f"[{i+1}] {_clean_chunk(c)}" for i, c in enumerate(chunks)
        )

    return (
        "You are a Stardew Valley wiki assistant.\n"
        "Your ONLY source of truth is the CONTEXT below. You must NOT use outside knowledge.\n\n"
        "STRICT RULES:\n"
        "1. If the CONTEXT contains the answer, reply in 1-2 natural sentences.\n"
        "2. If the CONTEXT does not contain the answer, reply EXACTLY:\n"
        "   \"I don't know based on the provided context.\"\n"
        "3. NEVER repeat the context verbatim; paraphrase.\n"
        "4. Do NOT invent locations, items, or recipes.\n\n"
        "EXAMPLE:\n"
        "CONTEXT: [1] Copper Node: Copper Ore. Location: Mines any floor, Skull Cavern, Quarry.\n"
        "QUESTION: Where can I find copper ore?\n"
        "ANSWER: You can find copper ore in the Mines (any floor), the Skull Cavern, and the Quarry.\n\n"
        "EXAMPLE:\n"
        "CONTEXT: [1] Abigail loved gifts: Amethyst, Pumpkin.\n"
        "QUESTION: What is the capital of France?\n"
        "ANSWER: I don't know based on the provided context.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER (1-2 sentences, or the exact refusal phrase):"
    )

def generate_answer(query: str, chunks: list[str], max_new_tokens: int = 120) -> str:
    _load_model()
    prompt = build_prompt(query, chunks)
    messages = [{"role": "user", "content": prompt}]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=3072)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.8,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    response = _tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip()


if __name__ == "__main__":
    from test import hybrid_search

    q = "Where can i find copper ore?"
    results = hybrid_search(q, page_filter=None, n_results=3, alpha=0.3)
    chunk_texts = [doc for _, _, doc, _ in results]
    print("Chunks recuperados:")
    for i, (score, section, doc, page) in enumerate(results):
        print(f"  {i+1}. [{page} / {section}] score={score:.3f} {doc[:120]}...")
    print("\nRespuesta:")
    print(generate_answer(q, chunk_texts))