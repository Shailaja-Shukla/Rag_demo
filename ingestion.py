"""
ingestion.py — Data ingestion pipeline for Disneyland Reviews.

Key design choices
──────────────────
* Parallel embedding   : N worker threads each embed one batch concurrently,
                         cutting wall-clock embed time by ~4-8x.
* Streaming upsert     : each batch is upserted into ChromaDB immediately after
                         it is embedded, so /chat can query partial data while
                         ingestion is still running.
* Background execution : run_ingestion_background() spawns a daemon thread so
                         Flask is never blocked.
* Progress tracking    : a module-level PROGRESS dict is updated after every
                         batch; Flask reads it via /ingest/status.

Metadata stored per chunk
──────────────────────────
  review_id, rating, year_month, location, branch, chunk_index, chunk_count
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread
from typing import Any

import chromadb
import pandas as pd
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams

from config import Config, load_config

logger = logging.getLogger(__name__)

# ── Progress state (read by /ingest/status) ───────────────────────────────────
PROGRESS: dict = {
    "running":        False,
    "done":           False,
    "error":          None,
    "total_chunks":   0,
    "embedded_chunks": 0,
    "upserted_chunks": 0,
    "total_rows":     0,
    "collection_name": "",
}
_progress_lock = Lock()


def _update_progress(**kwargs) -> None:
    with _progress_lock:
        PROGRESS.update(kwargs)


# ── Text helpers ──────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - chunk_overlap
    return chunks


def _build_chunk_text(row: pd.Series, chunk: str) -> str:
    """Prefix every chunk with its column values so text search hits metadata too."""
    return (
        f"Branch: {row['Branch']}\n"
        f"Rating: {row['Rating']}\n"
        f"Year_Month: {row['Year_Month']}\n"
        f"Reviewer_Location: {row['Reviewer_Location']}\n"
        f"Review: {chunk}"
    )


def _sanitize_id(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(value))


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _make_embed_client(cfg: Config) -> Embeddings:
    return Embeddings(
        model_id=cfg.embedding_model,
        credentials=Credentials(url=cfg.watsonx_url, api_key=cfg.watsonx_api_key),
        params={EmbedParams.TRUNCATE_INPUT_TOKENS: 512},
        project_id=cfg.watsonx_project_id,
    )


def _embed_batch_with_retry(
    cfg: Config, texts: list[str], retries: int = 3, backoff: float = 2.0
) -> list[list[float]]:
    """Embed one batch; retry up to `retries` times on transient errors."""
    client = _make_embed_client(cfg)   # each thread gets its own client
    for attempt in range(retries):
        try:
            return client.embed_documents(texts)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            logger.warning("Embed attempt %d failed (%s); retrying in %.1fs", attempt + 1, exc, wait)
            time.sleep(wait)
    return []   # unreachable


# ── Core pipeline ─────────────────────────────────────────────────────────────

BATCH_SIZE      = 50    # chunks per embedding request
EMBED_WORKERS   = 4     # parallel embedding threads
MAX_ROWS        = 2000  # cap: only ingest the first N rows of the CSV


def _run_pipeline(cfg: Config) -> None:
    """
    Full ingestion pipeline — runs inside a background thread.
    Updates PROGRESS throughout so the UI can poll progress.
    """
    _update_progress(
        running=True, done=False, error=None,
        total_chunks=0, embedded_chunks=0, upserted_chunks=0,
    )

    try:
        # 1. Load CSV ──────────────────────────────────────────────────────────
        csv_path = os.path.join(cfg.data_path, "DisneylandReviews.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at: {csv_path}")

        df = pd.read_csv(csv_path, encoding="latin-1")
        required = {"Review_ID", "Rating", "Year_Month", "Reviewer_Location",
                    "Review_Text", "Branch"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        df.dropna(subset=["Review_Text"], inplace=True)
        df["Review_Text"] = df["Review_Text"].astype(str)
        df = df.head(MAX_ROWS).reset_index(drop=True)
        logger.info("Using %d rows (capped at MAX_ROWS=%d)", len(df), MAX_ROWS)
        _update_progress(total_rows=len(df))

        # 2. Build all chunks ──────────────────────────────────────────────────
        all_ids: list[str]       = []
        all_texts: list[str]     = []
        all_metadatas: list[dict] = []

        for _, row in df.iterrows():
            chunks = _split_text(row["Review_Text"], cfg.chunk_size, cfg.chunk_overlap)
            for idx, chunk in enumerate(chunks):
                all_ids.append(f"{_sanitize_id(row['Review_ID'])}_chunk_{idx}")
                all_texts.append(_build_chunk_text(row, chunk))
                all_metadatas.append({
                    "review_id":   str(row["Review_ID"]),
                    "rating":      int(row["Rating"]),
                    "year_month":  str(row["Year_Month"]),
                    "location":    str(row["Reviewer_Location"]),
                    "branch":      str(row["Branch"]),
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                })

        total = len(all_ids)
        _update_progress(total_chunks=total, collection_name=cfg.chroma_collection)
        logger.info("Built %d chunks", total)

        # 3. Prepare ChromaDB (drop + recreate for clean ingest) ───────────────
        chroma = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
        try:
            chroma.delete_collection(cfg.chroma_collection)
        except Exception:
            pass
        collection = chroma.create_collection(
            name=cfg.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

        # 4. Parallel embed + streaming upsert ─────────────────────────────────
        # Slice chunks into batches, submit all to a thread pool,
        # then upsert each batch as soon as its embeddings come back.
        batches: list[tuple[int, int]] = [
            (s, min(s + BATCH_SIZE, total))
            for s in range(0, total, BATCH_SIZE)
        ]

        upsert_lock = Lock()   # protect collection.upsert from concurrent writes

        def embed_and_upsert(start: int, end: int) -> int:
            texts  = all_texts[start:end]
            ids    = all_ids[start:end]
            metas  = all_metadatas[start:end]
            vecs   = _embed_batch_with_retry(cfg, texts)

            with upsert_lock:
                collection.upsert(ids=ids, embeddings=vecs,
                                  documents=texts, metadatas=metas)

            done = end - start
            with _progress_lock:
                PROGRESS["embedded_chunks"] += done
                PROGRESS["upserted_chunks"] += done
            logger.info("Upserted batch %d–%d (%d chunks done so far)",
                        start, end - 1, PROGRESS["upserted_chunks"])
            return done

        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
            futures = {
                pool.submit(embed_and_upsert, s, e): (s, e)
                for s, e in batches
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    s, e = futures[future]
                    logger.error("Batch %d–%d failed: %s", s, e, exc)

        logger.info("=== Ingestion complete: %d chunks ===", PROGRESS["upserted_chunks"])
        _update_progress(running=False, done=True)

    except Exception as exc:
        logger.exception("Ingestion pipeline error")
        _update_progress(running=False, done=False, error=str(exc))


# ── Public API ────────────────────────────────────────────────────────────────

def run_ingestion_background(cfg: Config | None = None) -> None:
    """
    Start the ingestion pipeline in a daemon background thread.
    Returns immediately; poll PROGRESS for status.
    """
    if cfg is None:
        cfg = load_config()

    if PROGRESS["running"]:
        logger.warning("Ingestion already running — ignoring duplicate request")
        return

    thread = Thread(target=_run_pipeline, args=(cfg,), daemon=True, name="ingestion")
    thread.start()
    logger.info("Ingestion thread started")


def run_ingestion(cfg: Config | None = None) -> dict:
    """
    Synchronous wrapper kept for backward compatibility / CLI use.
    Blocks until ingestion is complete.
    """
    if cfg is None:
        cfg = load_config()
    _run_pipeline(cfg)
    return {
        "total_rows":     PROGRESS["total_rows"],
        "total_chunks":   PROGRESS["total_chunks"],
        "collection_name": PROGRESS["collection_name"],
    }
