import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection(name="stardew")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "no", "not", "of", "on", "or", "she", "so",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your",
    "best", "way", "get", "do", "does", "can", "could", "would", "should",
    "about", "any", "some", "all", "much", "many", "more", "most",
}

def tokenize(text):
    tokens = re.findall(r"\w+", text.lower())
    stemmed = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if t.endswith("ing") and len(t) > 4:
            t = t[:-3]
        elif t.endswith("ed") and len(t) > 3:
            t = t[:-2]
        elif t.endswith("es") and len(t) > 3:
            t = t[:-2]
        elif t.endswith("s") and len(t) > 2 and not t.endswith("ss"):
            t = t[:-1]
        stemmed.append(t)
    return stemmed


def _stopword_ratio(text):
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in STOPWORDS) / len(words)


def min_max(scores):
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


PAGE_HINTS = {
    "Fish": ["fish", "catch", "crab", "pufferfish", "eel", "sardine", "tuna", "salmon", "legend"],
    "Mining": ["ore", "mining", "geode", "iridium", "copper", "diamond", "node", "pickaxe"],
    "Crops": ["crop", "seed", "grow", "plant", "harvest", "strawberry", "cauliflower"],
    "Museum": ["museum", "donate", "artifact", "mineral", "dwarvish", "translation"],
    "Volcano Dungeon": ["volcano", "forge", "cinder"],
    "Ginger Island": ["ginger", "island", "walnut", "parrot", "leo"],
    "Secret Notes": ["secret note", "special charm", "magnifying glass"],
    "The Mines": ["mines", "skull key", "elevator", "floor 120"],
    "Skull Cavern": ["skull cavern", "shaft", "iridium"],
    "The Sewers": ["sewers", "rusty key", "krobus"],
    "Blacksmith": ["clint", "upgrade", "tool"],
    "Community Center": ["bundle", "junimo", "community center"],
    "Marriage": ["marriage", "pendant", "bouquet", "propose"],
    "Animals": ["chicken", "cow", "goat", "rabbit", "pig", "wool", "milk", "egg"],
    "Friendship": ["gift", "love", "like", "hate", "friendship", "heart"],
    "Traveling Cart": ["traveling cart", "merchant"],
    "Festivals": ["night market", "mermaid", "spirit's eve", "festival"],
    "Desert Trader": ["desert trader", "barter"],
    "Island Trader": ["island trader"],
    "Oasis": ["oasis", "sandy"],
    "Casino": ["casino", "qi coin"],
    "Secret Woods": ["secret woods", "hardwood", "large stump"],
    "Trees": ["tree", "wood", "sap", "hardwood", "stump"],
}


def _guess_pages(query: str):
    q = query.lower()
    hits = []
    for page, words in PAGE_HINTS.items():
        if any(w in q for w in words):
            hits.append(page)
    return hits


def hybrid_search(query, page_filter=None, n_results=3, alpha=0.5):
    guessed_pages = set()
    if page_filter:
        guessed_pages.add(page_filter)
    else:
        guessed_pages = set(_guess_pages(query))

    all_docs = collection.get()
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    ids = all_docs["ids"]

    if not documents:
        return []

    bm25 = BM25Okapi([tokenize(d) for d in documents])
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_norm = min_max(list(bm25_scores))

    sem_results = collection.query(
        query_embeddings=model.encode([query], normalize_embeddings=True).tolist(),
        n_results=len(documents),
    )
    sem_map = {}
    for doc_id, dist in zip(sem_results["ids"][0], sem_results["distances"][0]):
        sem_map[doc_id] = max(0.0, 1.0 - dist)
    sem_scores = [sem_map.get(doc_id, 0.0) for doc_id in ids]
    sem_norm = min_max(sem_scores)

    combined = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        score = alpha * bm25_norm[i] + (1 - alpha) * sem_norm[i]

        if meta["page"] in guessed_pages:
            score += 0.05

        score -= 0.05 * _stopword_ratio(doc)

        combined.append((score, meta["section"], doc, meta["page"]))

    combined.sort(key=lambda x: -x[0])
    return combined[:n_results]


if __name__ == "__main__":
    queries = [
        "What gifts does Leah love?",
        "Where can I find copper ore?",
        "What gifts does Sebastian hate?",
        "Where does Penny live?",
        "Where can I catch a pufferfish?",
        "What is the best way to get a Pearl?",
        "What does the Mermaid Show do?",
        "How can I get an acient seed?"
    ]
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        print(f"\n{'#'*70}\n- ALPHA = {alpha}\n{'#'*70}")
        for q in queries:
            print(f"\nQuery: '{q}'")
            results = hybrid_search(q, page_filter=None, n_results=3, alpha=alpha)
            for i, (score, section, doc, page) in enumerate(results):
                print(f"  {i+1}. [{page}/{section}] score = {score:.4f}  {doc[:120]}...")