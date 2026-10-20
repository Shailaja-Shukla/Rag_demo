"""
app.py — Flask web application for the Disneyland Review RAG system.

Routes
------
GET  /              → Chat UI
POST /ingest        → Kick off background ingestion (returns immediately)
GET  /ingest/status → Live progress for the ingestion thread
POST /chat          → Ask a question; works even during partial ingestion
GET  /health        → Health-check endpoint
"""

import logging

from flask import Flask, jsonify, render_template, request

from config import load_config
from ingestion import PROGRESS, run_ingestion_background
from retrieval import RAGPipeline

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── App init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
cfg = load_config()

# Lazy singleton — rebuilt whenever ingestion finishes
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline | None:
    """
    Return a RAGPipeline if the ChromaDB collection exists (even partially).
    Returns None if no data has been upserted yet.
    Rebuilds the singleton every time ingestion completes.
    """
    global _pipeline

    # Re-create the pipeline once ingestion has just finished
    if PROGRESS["done"] and _pipeline is None:
        try:
            _pipeline = RAGPipeline(cfg)
        except RuntimeError:
            pass   # collection still not ready (edge case)

    # During ingestion, rebuild on every call so we pick up new chunks
    if PROGRESS["running"]:
        try:
            return RAGPipeline(cfg)
        except RuntimeError:
            return None   # no chunks upserted yet

    return _pipeline


def _reset_pipeline() -> None:
    global _pipeline
    _pipeline = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/ingest")
def ingest():
    """
    Start background ingestion and return immediately.
    The client polls /ingest/status for live progress.
    """
    if PROGRESS["running"]:
        return jsonify({"status": "already_running", **PROGRESS}), 202

    _reset_pipeline()
    run_ingestion_background(cfg)
    return jsonify({"status": "started"}), 202


@app.get("/ingest/status")
def ingest_status():
    """Return the current ingestion progress snapshot."""
    pct = 0
    if PROGRESS["total_chunks"] > 0:
        pct = round(PROGRESS["upserted_chunks"] / PROGRESS["total_chunks"] * 100, 1)
    return jsonify({**PROGRESS, "percent": pct}), 200


@app.post("/chat")
def chat():
    """
    Accept a user question and return an answer with citations.
    Works with partial data while ingestion is still running.

    Request body  : {"question": "<user question>"}
    Response body : {"answer": "...", "citations": [...], "sub_queries": [...],
                     "partial": true/false}
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "question field is required"}), 400

    pipeline = _get_pipeline()
    if pipeline is None:
        return jsonify({
            "error": "⏳ No data available yet — ingestion hasn't started or "
                     "no chunks have been uploaded. Please wait a moment and retry."
        }), 503

    try:
        result = pipeline.answer(question)
        result["partial"] = PROGRESS["running"]   # flag so UI can show a notice
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Chat failed for question: %s", question)
        return jsonify({"error": "An error occurred. Please try again."}), 500


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=cfg.port, debug=False)
