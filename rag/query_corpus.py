"""
query_corpus.py - Ask questions against the persistent medical reference corpus.
================================================================================

Opens the persistent Chroma DB built by build_corpus.py, retrieves the most
relevant chunks for each question, and asks Claude to answer GROUNDED in
those chunks (with citations). This is what the real RAG flow inside your
lab parser app will look like.

Usage:
    python3 query_corpus.py "what does low ferritin mean?"   # one-shot
    python3 query_corpus.py                                  # interactive

Prerequisite:
    export ANTHROPIC_API_KEY="your-key-here"
"""

import os
import sys
from pathlib import Path

try:
    import chromadb
    from chromadb.utils import embedding_functions
    import anthropic
except ImportError:
    print("Missing dependency. Run:  python3 -m pip install chromadb anthropic")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DB_DIR = Path("./chroma_db")
COLLECTION_NAME = "medical_reference"
API_KEY = os.environ.get("ANTHROPIC_API_KEY") or "PASTE_YOUR_KEY_HERE"
MODEL_NAME = "claude-haiku-4-5-20251001"
TOP_K = 4                # how many chunks to retrieve
DISTANCE_THRESHOLD = 1.4 # warn if best hit is further than this


# ---------------------------------------------------------------------------
# SYSTEM PROMPT (the augmentation rules)
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are a careful medical-literacy assistant for a
personal lab-report helper app, optimized for Indian adult patients. You are
NOT a doctor and provide NO medical advice.

Your task: answer the user's question using ONLY the reference passages
provided below. Rules:

1. After EVERY factual claim, cite the source in brackets in the format
   [source_file -> Section Title]. Example: [01_cbc.md -> Hemoglobin (Hb)].
2. If the passages do not contain enough information to answer, say so
   plainly. Do NOT make up information.
3. Do NOT recommend treatments, medications, supplements, foods to eat or
   avoid, exercise, or any health interventions.
4. Do NOT give a specific diagnosis for the user's situation. You MAY name
   recognized clinical PATTERNS that values fit (e.g., "microcytic anemia
   pattern") - these are descriptive, not diagnostic.
5. Keep the answer concise (a few sentences to a short paragraph). Don't
   pad with general health advice.

End every answer with this line on its own:
"This is general reference information, not medical advice. Discuss your
results with a qualified doctor."
"""


# ---------------------------------------------------------------------------
# RETRIEVE
# ---------------------------------------------------------------------------
def retrieve(collection, question: str, k: int = TOP_K):
    results = collection.query(query_texts=[question], n_results=k)
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "title": meta["title"],
            "source_file": meta["source_file"],
            "panel": meta["panel"],
            "distance": dist,
        })
    return hits


# ---------------------------------------------------------------------------
# AUGMENT + GENERATE
# ---------------------------------------------------------------------------
def ask_with_rag(question: str, hits: list) -> dict:
    context_block = "\n\n---\n\n".join(
        f"[Source: {h['source_file']} -> {h['title']}]\n{h['text']}"
        for h in hits
    )
    user_message = (
        f"REFERENCE PASSAGES:\n\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER (with citations after each claim):"
    )

    client = anthropic.Anthropic(api_key=API_KEY)
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        system=RAG_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    usage = response.usage
    cost_inr = (usage.input_tokens * 1.0 + usage.output_tokens * 5.0) / 1_000_000 * 85
    return {
        "answer": response.content[0].text.strip(),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_inr": cost_inr,
    }


# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------
def answer_question(collection, question: str):
    print("\n" + "=" * 76)
    print(f"Q: {question}")
    print("=" * 76)

    hits = retrieve(collection, question)
    best_distance = hits[0]["distance"] if hits else float("inf")

    print(f"\nRetrieved top {len(hits)} chunks (lower distance = more relevant):")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h['source_file']:40s}  ->  {h['title']:30s}  d={h['distance']:.4f}")

    if best_distance > DISTANCE_THRESHOLD:
        print(f"\n  ! Best distance ({best_distance:.4f}) exceeds threshold ({DISTANCE_THRESHOLD}).")
        print(f"  ! The corpus probably doesn't cover this topic. Claude should refuse.")

    print("\n--- Claude's grounded answer ---")
    result = ask_with_rag(question, hits)
    print(result["answer"])
    print(f"\n[tokens: {result['input_tokens']} in, {result['output_tokens']} out  "
          f"|  cost ~Rs {result['cost_inr']:.2f}]")
    print("=" * 76 + "\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    if not DB_DIR.exists():
        print(f"Error: Chroma DB not found at {DB_DIR.absolute()}.")
        print(f"Run first:  python3 build_corpus.py")
        sys.exit(1)

    if API_KEY == "PASTE_YOUR_KEY_HERE":
        print("No API key set. Run:  export ANTHROPIC_API_KEY=\"your-key\"")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    print(f"Loaded corpus: {collection.count()} chunks indexed.")

    if len(sys.argv) > 1:
        # One-shot mode: question from command line
        question = " ".join(sys.argv[1:])
        answer_question(collection, question)
    else:
        # Interactive mode
        print("Interactive mode. Type a question and press Enter. Ctrl+C to exit.\n")
        try:
            while True:
                try:
                    question = input("Q: ").strip()
                except EOFError:
                    break
                if not question:
                    continue
                if question.lower() in {"exit", "quit", "q"}:
                    break
                answer_question(collection, question)
        except KeyboardInterrupt:
            pass
        print("\nBye.")


if __name__ == "__main__":
    main()
