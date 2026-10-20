"""
retrieval.py — Query retrieval pipeline with a Planner Agent.

Flow
────
  1. PlannerAgent   : breaks the user question into focused sub-queries
                      (one per relevant dimension: text, rating, branch, location, date).
  2. HybridRetriever: for each sub-query, searches ChromaDB by:
                        a) semantic vector similarity (text path)
                        b) metadata filter (structured path)
                      then merges and deduplicates results.
  3. Reranker        : scores every candidate chunk against the original question,
                       keeps top-K.
  4. Generator       : feeds the reranked context to the IBM watsonx LLM and
                       returns the answer with citation metadata.
"""

import json
import logging
import re
from typing import Any

import chromadb
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference
from ibm_watsonx_ai.metanames import (
    EmbedTextParamsMetaNames as EmbedParams,
    GenTextParamsMetaNames as GenParams,
)

from config import Config, load_config

logger = logging.getLogger(__name__)


# ── LLM factory ──────────────────────────────────────────────────────────────

def _make_llm(cfg: Config) -> ModelInference:
    credentials = Credentials(url=cfg.watsonx_url, api_key=cfg.watsonx_api_key)
    params = {
        GenParams.MAX_NEW_TOKENS: cfg.max_new_tokens,
        GenParams.TEMPERATURE:    cfg.temperature,
        GenParams.TOP_P:          cfg.top_p,
    }
    return ModelInference(
        model_id=cfg.llm_model,
        credentials=credentials,
        params=params,
        project_id=cfg.watsonx_project_id,
    )


def _make_embed_client(cfg: Config) -> Embeddings:
    credentials = Credentials(url=cfg.watsonx_url, api_key=cfg.watsonx_api_key)
    return Embeddings(
        model_id=cfg.embedding_model,
        credentials=credentials,
        params={EmbedParams.TRUNCATE_INPUT_TOKENS: 512},
        project_id=cfg.watsonx_project_id,
    )


# ── 1. Planner Agent ─────────────────────────────────────────────────────────

_PLANNER_PROMPT = """\
You are a query-planning assistant for a Disneyland review search engine.
The database has these columns: Review_ID, Rating (1-5), Year_Month, Reviewer_Location, Review_Text, Branch.

Given the user question below, decompose it into at most 4 focused sub-queries that together
cover all aspects of the question. Each sub-query must be a plain English sentence.

Additionally, if any sub-query maps to a structured metadata filter, output it as a JSON object
with keys from: rating (int), year_month (str), location (str), branch (str).
Use null for filters that do not apply.

Return ONLY a JSON array like:
[
  {{"text": "<sub-query 1>", "filter": {{"rating": null, "year_month": null, "location": null, "branch": null}}}},
  {{"text": "<sub-query 2>", "filter": {{"rating": 5, "year_month": null, "location": "Australia", "branch": null}}}}
]

User question: {question}
"""


class PlannerAgent:
    """Decomposes a user question into sub-queries with optional metadata filters."""

    def __init__(self, llm: ModelInference) -> None:
        self._llm = llm

    def plan(self, question: str) -> list[dict]:
        prompt = _PLANNER_PROMPT.format(question=question)
        raw = self._llm.generate_text(prompt=prompt)
        return self._parse(raw, question)

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(raw: str, original_question: str) -> list[dict]:
        """Extract the JSON array from the LLM output; fall back gracefully."""
        # Find the first [...] block in the response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                plans = json.loads(match.group())
                if isinstance(plans, list) and plans:
                    return plans
            except json.JSONDecodeError:
                pass

        logger.warning("Planner parse failed – falling back to single sub-query")
        return [
            {
                "text": original_question,
                "filter": {
                    "rating": None,
                    "year_month": None,
                    "location": None,
                    "branch": None,
                },
            }
        ]


# ── 2. Hybrid Retriever ───────────────────────────────────────────────────────

def _build_where_clause(filter_dict: dict) -> dict | None:
    """Convert a planner filter dict into a ChromaDB $and / single-field where clause."""
    conditions = []

    if filter_dict.get("rating") is not None:
        conditions.append({"rating": {"$eq": int(filter_dict["rating"])}})
    if filter_dict.get("year_month"):
        conditions.append({"year_month": {"$eq": str(filter_dict["year_month"])}})
    if filter_dict.get("location"):
        # case-insensitive substring is not natively supported; use exact match
        conditions.append({"location": {"$eq": str(filter_dict["location"])}})
    if filter_dict.get("branch"):
        conditions.append({"branch": {"$eq": str(filter_dict["branch"])}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class HybridRetriever:
    """
    Searches ChromaDB using both semantic vectors and metadata filters.
    Returns a deduplicated list of result dicts.
    """

    def __init__(self, collection: chromadb.Collection, embed_client: Embeddings) -> None:
        self._col = collection
        self._embed = embed_client

    def retrieve(self, sub_queries: list[dict], top_k: int) -> list[dict]:
        seen_ids: set[str] = set()
        results: list[dict] = []

        for sq in sub_queries:
            text   = sq.get("text", "")
            filter_ = sq.get("filter", {})
            where  = _build_where_clause(filter_)

            # a) Semantic / text path
            vector = self._embed.embed_documents([text])[0]
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [vector],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where

            try:
                resp = self._col.query(**query_kwargs)
                self._collect(resp, results, seen_ids)
            except Exception as exc:
                logger.warning("Semantic query failed for '%s': %s", text, exc)

            # b) Metadata-only path (pure filter, no vector) – only when filter exists
            if where:
                try:
                    meta_resp = self._col.get(
                        where=where,
                        include=["documents", "metadatas"],
                        limit=top_k,
                    )
                    self._collect_get(meta_resp, results, seen_ids)
                except Exception as exc:
                    logger.warning("Metadata query failed for '%s': %s", text, exc)

        return results

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _collect(
        resp: dict, out: list[dict], seen: set[str]
    ) -> None:
        for doc, meta, dist, doc_id in zip(
            resp["documents"][0],
            resp["metadatas"][0],
            resp["distances"][0],
            resp["ids"][0],
        ):
            if doc_id not in seen:
                seen.add(doc_id)
                out.append({"id": doc_id, "text": doc, "metadata": meta, "score": 1 - dist})

    @staticmethod
    def _collect_get(resp: dict, out: list[dict], seen: set[str]) -> None:
        for doc, meta, doc_id in zip(
            resp["documents"], resp["metadatas"], resp["ids"]
        ):
            if doc_id not in seen:
                seen.add(doc_id)
                out.append({"id": doc_id, "text": doc, "metadata": meta, "score": 0.5})


# ── 3. Reranker ───────────────────────────────────────────────────────────────

_RERANK_PROMPT = """\
Score the relevance of the following passage to the question on a scale of 0–10.
Reply with ONLY a single integer.

Question: {question}

Passage:
{passage}

Score:"""


class Reranker:
    """LLM-based pointwise reranker; keeps the top-K highest-scoring chunks."""

    def __init__(self, llm: ModelInference) -> None:
        self._llm = llm

    def rerank(self, question: str, candidates: list[dict], top_k: int) -> list[dict]:
        scored: list[tuple[float, dict]] = []

        for chunk in candidates:
            prompt = _RERANK_PROMPT.format(
                question=question, passage=chunk["text"][:600]
            )
            try:
                raw = self._llm.generate_text(prompt=prompt).strip()
                score = float(re.search(r"\d+(?:\.\d+)?", raw).group())  # type: ignore[union-attr]
            except Exception:
                score = chunk["score"] * 10   # fall back to retrieval score

            scored.append((score, chunk))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


# ── 4. Generator ──────────────────────────────────────────────────────────────

_ANSWER_PROMPT = """\
You are a helpful assistant that answers questions about Disneyland customer reviews.
Use ONLY the context passages provided below. Cite your sources using [review_id].

Context:
{context}

Question: {question}

Answer:"""


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        parts.append(
            f"[{i}] review_id={meta.get('review_id')} | "
            f"branch={meta.get('branch')} | rating={meta.get('rating')} | "
            f"location={meta.get('location')}\n{c['text']}"
        )
    return "\n\n".join(parts)


def _build_citations(chunks: list[dict]) -> list[dict]:
    return [
        {
            "review_id": c["metadata"].get("review_id"),
            "branch":    c["metadata"].get("branch"),
            "rating":    c["metadata"].get("rating"),
            "location":  c["metadata"].get("location"),
            "year_month": c["metadata"].get("year_month"),
            "score":     round(c["score"], 4),
        }
        for c in chunks
    ]


# ── Public entry point ────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Assembles the full RAG pipeline.

    Usage
    -----
    pipeline = RAGPipeline(cfg)
    result   = pipeline.answer("Which branch has the best food?")
    # result = {"answer": "...", "citations": [...], "sub_queries": [...]}
    """

    def __init__(self, cfg: Config | None = None) -> None:
        if cfg is None:
            cfg = load_config()
        self._cfg = cfg

        llm          = _make_llm(cfg)
        embed_client = _make_embed_client(cfg)

        chroma_client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)

        existing = [c.name for c in chroma_client.list_collections()]
        if cfg.chroma_collection not in existing:
            raise RuntimeError("not_ingested")

        collection    = chroma_client.get_collection(cfg.chroma_collection)

        self._planner   = PlannerAgent(llm)
        self._retriever = HybridRetriever(collection, embed_client)
        self._reranker  = Reranker(llm)
        self._llm       = llm

    def answer(self, question: str) -> dict:
        # 1. Plan
        sub_queries = self._planner.plan(question)
        logger.info("Sub-queries: %s", [sq["text"] for sq in sub_queries])

        # 2. Retrieve
        candidates = self._retriever.retrieve(sub_queries, top_k=self._cfg.top_k)
        logger.info("Retrieved %d candidates", len(candidates))

        # 3. Rerank
        top_chunks = self._reranker.rerank(question, candidates, top_k=5)
        logger.info("Reranked to %d chunks", len(top_chunks))

        # 4. Generate
        context = _format_context(top_chunks)
        prompt  = _ANSWER_PROMPT.format(context=context, question=question)
        answer  = self._llm.generate_text(prompt=prompt)

        return {
            "answer":      answer.strip(),
            "citations":   _build_citations(top_chunks),
            "sub_queries": [sq["text"] for sq in sub_queries],
        }
