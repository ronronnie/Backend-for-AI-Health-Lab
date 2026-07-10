"""
build_corpus.py - Ingest the medical reference corpus into a persistent Chroma DB.
==================================================================================

What this script does (a real RAG pipeline, not a toy):

  1. WALK the corpus/ folder for .md files (one file per panel).
  2. CHUNK each file at H2 (##) headings - each test / pattern becomes
     one self-contained chunk. This is "structural chunking", the cleanest
     strategy when source content has consistent section boundaries.
  3. EMBED every chunk with Chroma's default embedding model.
  4. STORE everything in a PERSISTENT Chroma database on disk
     (./chroma_db/). Persistent means: built once, used forever - subsequent
     runs of query_corpus.py just open the existing DB, they don't re-embed.
  5. ATTACH metadata to every chunk: which file it came from, which panel,
     which section heading. This is what makes citations possible later.

Usage:
    python3 build_corpus.py                  # uses ./corpus/
    python3 build_corpus.py /path/to/corpus  # custom corpus folder

Run this ONCE after creating or updating your corpus. Then use
query_corpus.py to ask questions.
"""

import re
import sys
from pathlib import Path

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("Missing dependency. Run:  python3 -m pip install chromadb")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CORPUS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./corpus")
DB_DIR = Path("./chroma_db")
COLLECTION_NAME = "medical_reference"


# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------
def parse_markdown_chunks(file_path: Path):
    """
    Split a markdown file into chunks at H2 (##) headings. Each chunk is one
    self-contained section (one test or one clinical pattern).

    Why this strategy? Because our reference content is structured: every
    section has a heading and a small set of fields underneath. Splitting at
    H2 means each chunk is meaningful on its own and roughly the same size.

    In real-world RAG with unstructured documents, you'd use a "recursive
    character splitter" (e.g. ~500-1000 chars with overlap). For our well-
    structured markdown, heading-based splits produce better retrieval.
    """
    text = file_path.read_text(encoding="utf-8")
    sections = re.split(r"\n## ", text)
    if len(sections) < 2:
        return []  # file has no ## headings - probably just a README

    chunks = []
    # sections[0] is the file intro (before first H2) - we skip it
    for section in sections[1:]:
        lines = section.split("\n", 1)
        if len(lines) < 2:
            continue
        title = lines[0].strip()
        body = lines[1].strip()
        # Re-attach the heading so the chunk is fully self-describing
        chunk_text = f"## {title}\n\n{body}"
        chunks.append({"title": title, "text": chunk_text})
    return chunks


def slugify(s: str) -> str:
    """Make a safe id fragment from a heading."""
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[-\s]+", "_", s)
    return s[:40]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    if not CORPUS_DIR.exists():
        print(f"Error: corpus folder not found: {CORPUS_DIR.absolute()}")
        print(f"Create it and add .md files, then re-run.")
        sys.exit(1)

    md_files = sorted([f for f in CORPUS_DIR.glob("*.md") if not f.name.startswith("00_")])
    if not md_files:
        print(f"No .md files found in {CORPUS_DIR.absolute()}")
        sys.exit(1)

    print(f"Found {len(md_files)} content file(s) in {CORPUS_DIR}/")
    for f in md_files:
        print(f"  - {f.name}")

    # Initialize a PERSISTENT Chroma client. Data lives on disk under DB_DIR.
    DB_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    ef = embedding_functions.DefaultEmbeddingFunction()

    # Wipe and recreate the collection so each build starts clean. (In a
    # production system you might do incremental updates instead.)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"\nDeleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )
    print(f"Created fresh collection '{COLLECTION_NAME}'.\n")

    # Walk, chunk, collect.
    all_ids, all_texts, all_metas = [], [], []
    for md_file in md_files:
        chunks = parse_markdown_chunks(md_file)
        panel = md_file.stem.split("_", 1)[-1].replace("_", " ").title()
        print(f"  {md_file.name:40s}  -> {len(chunks):3d} chunks  ({panel})")

        for i, c in enumerate(chunks):
            chunk_id = f"{md_file.stem}__{i:02d}__{slugify(c['title'])}"
            all_ids.append(chunk_id)
            all_texts.append(c["text"])
            all_metas.append({
                "source_file": md_file.name,
                "panel": panel,
                "title": c["title"],
            })

    print(f"\nTotal chunks: {len(all_ids)}")
    print("Embedding and storing (this may take 10-30 seconds for ~80 chunks)...")

    # Chroma will embed all texts in one call and write them to disk.
    collection.add(ids=all_ids, documents=all_texts, metadatas=all_metas)

    print(f"\nDone. Persistent Chroma DB at:  {DB_DIR.absolute()}")
    print(f"Total chunks in collection:     {collection.count()}")
    print(f"\nNext step: ask questions with")
    print(f"  python3 query_corpus.py \"what does low ferritin mean?\"")


if __name__ == "__main__":
    main()
