"""
config.py — Centralised configuration loader.
Reads all values from config.env and exposes them as a typed Config object.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv("config.env")


@dataclass(frozen=True)
class Config:
    watsonx_url: str
    watsonx_api_key: str
    watsonx_project_id: str
    llm_model: str
    embedding_model: str
    data_path: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    top_p: float
    temperature: float
    max_new_tokens: int
    port: int
    chroma_collection: str = "disney_reviews"
    chroma_persist_dir: str = "./chroma_db"


def load_config() -> Config:
    return Config(
        watsonx_url=os.environ["WATSONX_URL"],
        watsonx_api_key=os.environ["WATSONX_API_KEY"],
        watsonx_project_id=os.environ["WATSONX_PROJECT_ID"],
        llm_model=os.environ["WATSONX_LLM_MODEL"],
        embedding_model=os.environ["WATSONX_EMBEDDING_MODEL"],
        data_path=os.environ.get("PDF_PATH", "data"),
        top_k=int(os.environ.get("TOP_K", "10")),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "64")),
        top_p=float(os.environ.get("TOP_P", "0.9")),
        temperature=float(os.environ.get("TEMPERATURE", "0.2")),
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "512")),
        port=int(os.environ.get("PORT", "5001")),
    )
