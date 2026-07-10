"""
backend.py - FastAPI service wrapping the lab parser + RAG pipeline.
====================================================================
This is a thin HTTP layer over your existing code:

  - POST /api/parse        Upload a PDF or image of a lab report.
                           Returns the enriched JSON (parsed values +
                           cited explanations + pattern explanations).

  - GET  /api/health       Quick liveness check. Verifies the corpus
                           database is loaded and reachable.

Why FastAPI? Because it's the lightest way to put HTTP in front of
existing Python. Async support, automatic Swagger docs at /docs, and
file-upload handling are all free.

PREREQUISITES
-------------
This file lives at  ~/Desktop/AI_Health_Lab/backend.py
and imports your existing files in the same folder:
  - lab_parser.py        (the parser)
  - lab_parser_rag.py    (parser + RAG enrichment)
  - rag/chroma_db/       (the persistent corpus)

INSTALL
-------
    python3 -m pip install fastapi "uvicorn[standard]" python-multipart

RUN (during development)
------------------------
    cd ~/Desktop/AI_Health_Lab
    export ANTHROPIC_API_KEY="your-key-here"
    uvicorn backend:app --reload --port 8000

You should then be able to hit:
    http://localhost:8000/api/health          (browser or curl)
    http://localhost:8000/docs                (interactive Swagger UI)

TEST WITH CURL
--------------
    curl -X POST http://localhost:8000/api/parse \\
         -F "file=@/path/to/your/report.pdf"
"""

import os
import sys
import tempfile
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    print('Missing dependency. Run:  python3 -m pip install fastapi "uvicorn[standard]" python-multipart')
    sys.exit(1)

# Make sure we can import the existing parser + RAG modules
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import anthropic
    import chromadb
    from chromadb.utils import embedding_functions
    from lab_parser import parse_report, MODEL_NAME
    from lab_parser_rag import (
        load_collection,
        enrich_value,
        enrich_pattern,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("This file must be placed alongside lab_parser.py and lab_parser_rag.py")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# CORS origins. Localhost is always allowed for dev. In production, set the
# CORS_ORIGINS env var to a comma-separated list of your deployed frontend
# URLs, e.g.  CORS_ORIGINS="https://parentcare.vercel.app"
ALLOWED_ORIGINS = [
    "http://localhost:3000",      # Next.js dev server
    "http://127.0.0.1:3000",
]
ALLOWED_ORIGINS += [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()
]

# Simple per-IP rate limit for the public demo, so nobody can drain the
# Anthropic budget. Overridable via env. Window is a rolling number of seconds.
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "3600"))


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ParentCare Backend",
    description="Parses lab reports and enriches them with RAG-grounded explanations.",
    version="0.1.0",
)

# CORS: allow the Next.js dev server (and later your Vercel domain) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Lazy-load the corpus and Claude client on first request (so startup is fast
# and we don't crash if Chroma is misconfigured until someone actually uses it)
_collection = None
_claude_client = None


def get_collection():
    global _collection
    if _collection is None:
        _collection = load_collection()
    return _collection


def get_claude_client():
    global _claude_client
    if _claude_client is None:
        if not API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
        _claude_client = anthropic.Anthropic(api_key=API_KEY)
    return _claude_client


# ---------------------------------------------------------------------------
# RATE LIMITING (in-memory, per client IP)
# ---------------------------------------------------------------------------
# Good enough for a single-instance free-tier demo. State is per-process, so it
# resets on restart and isn't shared across multiple workers — that's fine here.
_request_log: "defaultdict[str, deque]" = defaultdict(deque)


def _client_ip(request: Request) -> str:
    # On Render/most PaaS the app sits behind a proxy, so the real client IP is
    # the first entry in X-Forwarded-For, not request.client.host.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time.time()
    hits = _request_log[ip]
    # Drop timestamps outside the rolling window.
    while hits and hits[0] <= now - RATE_LIMIT_WINDOW_SECONDS:
        hits.popleft()
    if len(hits) >= RATE_LIMIT_MAX:
        retry_after = int(hits[0] + RATE_LIMIT_WINDOW_SECONDS - now)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit reached ({RATE_LIMIT_MAX} uploads per "
                f"{RATE_LIMIT_WINDOW_SECONDS // 60} min). Try again in "
                f"{max(retry_after, 1)}s."
            ),
            headers={"Retry-After": str(max(retry_after, 1))},
        )
    hits.append(now)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    """Quick liveness + readiness check."""
    try:
        n = get_collection().count()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(e)},
        )
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "corpus_chunks": n,
        "api_key_set": bool(API_KEY),
    }


@app.post("/api/parse")
async def parse_endpoint(request: Request, file: UploadFile = File(...)):
    """
    Accept a PDF or image upload. Runs the full pipeline:
      1. Parse the report (lab_parser.py).
      2. Enrich abnormal values with cited RAG explanations.
      3. Enrich detected patterns with cited RAG explanations.

    Returns the enriched JSON.
    """
    enforce_rate_limit(request)

    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY not set on the server.",
        )

    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use PDF, JPG, PNG, or WEBP.",
        )

    # Persist the upload to a temp file so the existing parse_report() can read it
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read upload: {e}")

    try:
        # ---- Step 1: parse the report ----
        parsed = parse_report(tmp_path)

        # ---- Step 2: enrich abnormal values ----
        collection = get_collection()
        claude_client = get_claude_client()

        abnormal = [v for v in parsed.get("values", []) if v.get("status") not in ("normal", "")]
        total_in, total_out = 0, 0
        for value in abnormal:
            usage = enrich_value(collection, claude_client, value)
            total_in += usage.input_tokens
            total_out += usage.output_tokens

        # ---- Step 3: enrich patterns ----
        patterns = parsed.get("summary", {}).get("patterns_detected", []) or []
        pattern_explanations = []
        for pattern in patterns:
            pe, usage = enrich_pattern(collection, claude_client, pattern)
            pattern_explanations.append(pe)
            total_in += usage.input_tokens
            total_out += usage.output_tokens

        parsed.setdefault("summary", {})["pattern_explanations"] = pattern_explanations

        # ---- Step 4: attach a small usage summary ----
        rag_cost_inr = (total_in * 1.0 + total_out * 5.0) / 1_000_000 * 85
        parsed["meta"] = {
            "rag_calls": len(abnormal) + len(patterns),
            "rag_input_tokens": total_in,
            "rag_output_tokens": total_out,
            "rag_estimated_cost_inr": round(rag_cost_inr, 2),
        }

        return parsed

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# DEV ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=True)
