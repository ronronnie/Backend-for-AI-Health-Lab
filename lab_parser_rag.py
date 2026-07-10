"""
lab_parser_rag.py - Lab Parser with RAG-Grounded Explanations
=============================================================
This is the full pipeline that ties everything together:

  1. PARSE the lab report (reuses lab_parser.py's existing logic).
  2. For every ABNORMAL VALUE in the parsed JSON, query the medical
     reference corpus, retrieve the most relevant chunks, and ask Claude
     for a 2-4 sentence cited explanation.
  3. For every CLINICAL PATTERN detected (microcytic anemia, fatty liver,
     etc.), do the same.
  4. Save the enriched report as JSON and pretty-print the final output.

Each retrieved explanation includes citations like
[01_cbc.md -> Mean Corpuscular Volume (MCV)] so every claim is traceable
back to a specific section of your reference corpus.

PREREQUISITES
-------------
- lab_parser.py in the same folder as this script
- rag/chroma_db/ already built (run rag/build_corpus.py once)
- ANTHROPIC_API_KEY exported in your shell

USAGE
-----
    python3 lab_parser_rag.py /path/to/report.pdf
    python3 lab_parser_rag.py /path/to/report.jpg

EXPECTED COST
-------------
- Parsing the report:              ~Rs 1-2
- Enriching each abnormal value:   ~Rs 0.20-0.30 each
- Enriching each pattern:          ~Rs 0.20-0.30 each
- Total for a typical report:      ~Rs 3-5 (about 6-10 RAG calls)
"""

import json
import os
import sys
from pathlib import Path

try:
    import anthropic
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("Missing dependency. Run:  python3 -m pip install anthropic chromadb")
    sys.exit(1)

# Import the existing parser from the sibling file
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from lab_parser import parse_report, MODEL_NAME
except ImportError:
    print(f"Error: could not find lab_parser.py in {SCRIPT_DIR}.")
    print("This script must live in the same folder as lab_parser.py.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("ANTHROPIC_API_KEY") or "PASTE_YOUR_KEY_HERE"
DB_DIR = SCRIPT_DIR / "rag" / "chroma_db"
COLLECTION_NAME = "medical_reference"
TOP_K = 4   # how many corpus chunks to retrieve per explanation


# ---------------------------------------------------------------------------
# SYSTEM PROMPT - tightened for lab-explanation context.
# The "no advice" rules are stricter than the corpus query prompt because
# this output is going into an automated app shown to non-clinicians.
# ---------------------------------------------------------------------------
EXPLAIN_SYSTEM_PROMPT = """You are a careful medical-literacy assistant for
a personal lab-report tracker. You are NOT a doctor and do NOT give medical
advice.

Your task: explain a single lab value or clinical pattern using ONLY the
reference passages provided. Strict rules:

1. After every factual claim, cite the source in brackets in the format
   [source_file -> Section Title]. Example: [01_cbc.md -> Hemoglobin (Hb)].
2. Keep the explanation concise: 2-4 sentences. Do not pad with general
   health information.
3. Describe what the value or pattern can factually INDICATE in general
   terms. Do NOT speculate about THIS specific patient's situation.
4. Do NOT recommend treatments, medications, supplements, foods to eat or
   avoid, exercise, lifestyle changes, or any health interventions.
5. Do NOT direct the user to "investigate," "monitor," "evaluate," "check,"
   "consider," or "consult" anything specific. Those are clinician decisions.
   You may say "this pattern is commonly seen with X" but not "you should
   check for X."
6. If the passages do not adequately cover the question, say: "The
   reference corpus does not have specific information about this." Do not
   make things up to fill the gap.
"""


# ---------------------------------------------------------------------------
# RAG HELPERS
# ---------------------------------------------------------------------------
def load_collection():
    """Open the persistent Chroma collection built by build_corpus.py."""
    if not DB_DIR.exists():
        print(f"\nError: corpus database not found at {DB_DIR}.")
        print(f"Run first:  cd {SCRIPT_DIR / 'rag'} && python3 build_corpus.py\n")
        sys.exit(1)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def retrieve(collection, query: str, k: int = TOP_K):
    """Retrieve top-k chunks for a query."""
    results = collection.query(query_texts=[query], n_results=k)
    return [
        {
            "text": doc,
            "title": meta["title"],
            "source_file": meta["source_file"],
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def explain_with_rag(claude_client, query: str, hits: list):
    """Ask Claude for a grounded explanation, using only the retrieved chunks."""
    context = "\n\n---\n\n".join(
        f"[Source: {h['source_file']} -> {h['title']}]\n{h['text']}"
        for h in hits
    )
    user_msg = (
        f"REFERENCE PASSAGES:\n\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"EXPLANATION (with citations after each claim, 2-4 sentences):"
    )
    response = claude_client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        system=EXPLAIN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text.strip(), response.usage


# ---------------------------------------------------------------------------
# ENRICHMENT
# ---------------------------------------------------------------------------
def enrich_value(collection, claude_client, value: dict):
    """Generate a cited explanation for one abnormal value."""
    name = value.get("name", "")
    val = value.get("value", "")
    unit = value.get("unit", "")
    status = value.get("status", "")

    query = f"What does a {status} {name} value of {val} {unit} indicate?".strip()
    hits = retrieve(collection, query)
    explanation, usage = explain_with_rag(claude_client, query, hits)

    value["cited_explanation"] = explanation
    value["sources"] = [
        {"file": h["source_file"], "section": h["title"], "distance": round(h["distance"], 4)}
        for h in hits
    ]
    return usage


def enrich_pattern(collection, claude_client, pattern: str):
    """Generate a cited explanation for one detected clinical pattern."""
    query = f"What is {pattern}? What lab findings define it and what can it indicate?"
    hits = retrieve(collection, query)
    explanation, usage = explain_with_rag(claude_client, query, hits)

    return {
        "pattern": pattern,
        "explanation": explanation,
        "sources": [
            {"file": h["source_file"], "section": h["title"], "distance": round(h["distance"], 4)}
            for h in hits
        ],
    }, usage


# ---------------------------------------------------------------------------
# PRETTY PRINT
# ---------------------------------------------------------------------------
def pretty_print(parsed: dict):
    light_labels = {"green": "[GREEN]", "yellow": "[YELLOW]", "red": "[RED]"}
    summary = parsed.get("summary", {})
    light = summary.get("traffic_light", "?")
    headline = summary.get("headline", "")

    print("\n" + "=" * 76)
    print(f"{light_labels.get(light, '[?]')}  {headline}")
    print("=" * 76)

    patient = parsed.get("patient", {})
    print(f"\nPatient : {patient.get('name') or '-'}  "
          f"({patient.get('age') or '-'}, {patient.get('gender') or '-'})")
    print(f"Lab     : {patient.get('lab_name') or '-'}")
    print(f"Date    : {patient.get('report_date') or '-'}")
    print(f"Panel   : {parsed.get('test_panel', '-')}")

    # Abnormal values with RAG explanations
    abnormal = [v for v in parsed.get("values", []) if v.get("status") not in ("normal", "")]
    if abnormal:
        print("\n" + "=" * 76)
        print(f"ABNORMAL VALUES WITH GROUNDED EXPLANATIONS  ({len(abnormal)})")
        print("=" * 76)
        for v in abnormal:
            print(f"\n  {v.get('name', '?')}: {v.get('value', '?')} {v.get('unit', '')}  "
                  f"({v.get('status', '?')}, ref: {v.get('reference_range', '?')})")
            print(f"  {'-' * 70}")
            ce = v.get("cited_explanation", "(no explanation generated)")
            # Wrap the explanation lines to a reasonable width
            for line in ce.split("\n"):
                print(f"  {line}")

    # Clinical pattern explanations
    pattern_explanations = summary.get("pattern_explanations", []) or []
    if pattern_explanations:
        print("\n" + "=" * 76)
        print(f"CLINICAL PATTERNS WITH GROUNDED EXPLANATIONS  ({len(pattern_explanations)})")
        print("=" * 76)
        for pe in pattern_explanations:
            print(f"\n  {pe['pattern']}")
            print(f"  {'-' * 70}")
            for line in pe["explanation"].split("\n"):
                print(f"  {line}")

    next_steps = summary.get("next_steps", "")
    if next_steps:
        print("\n" + "=" * 76)
        print("NEXT STEPS")
        print("=" * 76)
        print(f"\n  {next_steps}")

    print(f"\nNote: {parsed.get('disclaimer', '')}\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage:  python3 lab_parser_rag.py <path_to_report.pdf_or_image>")
        sys.exit(1)

    if API_KEY == "PASTE_YOUR_KEY_HERE":
        print("Set API key:  export ANTHROPIC_API_KEY=\"your-key\"")
        sys.exit(1)

    report_path = sys.argv[1]

    # ---- STEP 1: Parse the report (existing logic) ----
    print("=" * 76)
    print("STEP 1: PARSE THE LAB REPORT")
    print("=" * 76)
    parsed = parse_report(report_path)

    # ---- STEP 2: Load the corpus ----
    print("\n" + "=" * 76)
    print("STEP 2: LOAD THE MEDICAL REFERENCE CORPUS")
    print("=" * 76)
    collection = load_collection()
    print(f"Loaded corpus: {collection.count()} chunks indexed.")

    claude_client = anthropic.Anthropic(api_key=API_KEY)

    # ---- STEP 3: Enrich abnormal values ----
    print("\n" + "=" * 76)
    print("STEP 3: ENRICH ABNORMAL VALUES")
    print("=" * 76)
    abnormal_values = [v for v in parsed.get("values", []) if v.get("status") not in ("normal", "")]
    print(f"Found {len(abnormal_values)} abnormal value(s).")

    total_in, total_out = 0, 0
    for i, value in enumerate(abnormal_values, 1):
        print(f"\n  [{i}/{len(abnormal_values)}] {value.get('name', '?')}: "
              f"{value.get('value', '?')} {value.get('unit', '')} "
              f"({value.get('status', '?')})")
        usage = enrich_value(collection, claude_client, value)
        total_in += usage.input_tokens
        total_out += usage.output_tokens
        sources = [s["section"] for s in value["sources"]]
        print(f"      Cited from: {', '.join(sources)}")

    # ---- STEP 4: Enrich clinical patterns ----
    print("\n" + "=" * 76)
    print("STEP 4: ENRICH CLINICAL PATTERNS")
    print("=" * 76)
    patterns = parsed.get("summary", {}).get("patterns_detected", []) or []
    print(f"Found {len(patterns)} pattern(s).")

    pattern_explanations = []
    for i, pattern in enumerate(patterns, 1):
        print(f"\n  [{i}/{len(patterns)}] {pattern}")
        pe, usage = enrich_pattern(collection, claude_client, pattern)
        pattern_explanations.append(pe)
        total_in += usage.input_tokens
        total_out += usage.output_tokens
        sources = [s["section"] for s in pe["sources"]]
        print(f"      Cited from: {', '.join(sources)}")

    parsed["summary"]["pattern_explanations"] = pattern_explanations

    # ---- STEP 5: Cost summary ----
    cost_inr = (total_in * 1.0 + total_out * 5.0) / 1_000_000 * 85
    print("\n" + "=" * 76)
    print(f"RAG ENRICHMENT COMPLETE")
    print(f"  Total RAG calls: {len(abnormal_values) + len(patterns)}")
    print(f"  Tokens used:     {total_in} in, {total_out} out")
    print(f"  RAG cost:        ~Rs {cost_inr:.2f}  (excludes the initial parse)")
    print("=" * 76)

    # ---- STEP 6: Save the enriched JSON ----
    out_path = Path(report_path).expanduser().with_suffix(".enriched.json")
    out_path.write_text(json.dumps(parsed, indent=2))
    print(f"\nEnriched JSON saved to:  {out_path}")

    # ---- STEP 7: Pretty-print the final result ----
    pretty_print(parsed)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}\n")
        sys.exit(1)
