# ParentCare Backend — Deploy Guide

FastAPI service that parses lab reports and enriches them with RAG-grounded,
cited explanations. Pairs with the ParentCare Next.js frontend (deployed
separately on Vercel).

## What's in here

- `backend.py` — FastAPI app (`/api/parse`, `/api/health`)
- `lab_parser.py`, `lab_parser_rag.py` — parsing + RAG enrichment
- `rag/chroma_db/` — the prebuilt ChromaDB corpus (committed, ~2 MB)
- `Dockerfile` — container that bundles the corpus and pre-warms the embedder
- `render.yaml` — Render Blueprint for one-click setup

## Deploy on Render (free tier)

1. Push this repo to GitHub (see below).
2. Render dashboard → **New → Blueprint** → pick this repo. It reads
   `render.yaml` and creates the `parentcare-backend` Docker service.
3. When prompted, set env vars:
   - `ANTHROPIC_API_KEY` — your Anthropic key
   - `CORS_ORIGINS` — your Vercel URL, e.g. `https://parentcare.vercel.app`
     (comma-separate multiple; localhost is always allowed)
4. Deploy. When it's live, verify:
   `curl https://<your-service>.onrender.com/api/health`
   → should return `{"status":"ok","corpus_chunks":<n>, ...}`
5. Copy that base URL into the frontend's `NEXT_PUBLIC_BACKEND_URL` on Vercel.

> Note: Render's free tier sleeps after inactivity, so the first request after
> idle takes ~30–60s to wake. Fine for a demo.

## Environment variables

| Var | Required | Default | Purpose |
|-----|----------|---------|---------|
| `ANTHROPIC_API_KEY` | yes | — | Claude API access |
| `CORS_ORIGINS` | prod | — | Comma-separated allowed frontend origins |
| `RATE_LIMIT_MAX` | no | `10` | Max uploads per IP per window |
| `RATE_LIMIT_WINDOW_SECONDS` | no | `3600` | Rate-limit window |
| `PORT` | no | `8000` | Injected by Render automatically |

## Run locally

```bash
python3 -m pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"
uvicorn backend:app --reload --port 8000
# health: http://localhost:8000/api/health   docs: http://localhost:8000/docs
```

Or with Docker:

```bash
docker build -t parentcare-backend .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY="your-key" parentcare-backend
```
