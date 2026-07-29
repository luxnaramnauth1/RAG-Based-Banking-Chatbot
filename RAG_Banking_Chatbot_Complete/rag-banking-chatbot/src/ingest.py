"""
Ingestion pipeline: load the banking knowledge base, chunk it, embed it,
and persist a vector index to disk so the chatbot can load it instantly
on future runs without re-embedding.

Run this once (or whenever data/banking_kb.json changes):
    python src/ingest.py
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def load_documents(path: str) -> list[Document]:
    """Load the banking KB JSON and convert each entry into a LlamaIndex Document."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    documents = []
    for r in records:
        text = f"{r['title']}\n\n{r['content']}"
        doc = Document(
            text=text,
            doc_id=r["id"],
            metadata={
                "id": r["id"],
                "category": r["category"],
                "title": r["title"],
            },
        )
        documents.append(doc)
    return documents


def build_index() -> VectorStoreIndex:
    print(f"Loading knowledge base from {config.DATA_PATH} ...")
    documents = load_documents(config.DATA_PATH)
    print(f"Loaded {len(documents)} documents.")

    print(f"Loading embedding model: {config.EMBED_MODEL_NAME} ...")
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    Settings.node_parser = splitter

    print("Building vector index (embedding all chunks)...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    os.makedirs(config.STORAGE_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=config.STORAGE_DIR)
    print(f"Index persisted to {config.STORAGE_DIR}")
    return index


def load_or_build_index() -> VectorStoreIndex:
    """Load a previously persisted index if it exists, otherwise build one."""
    from llama_index.core import load_index_from_storage

    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    if os.path.exists(config.STORAGE_DIR):
        print(f"Loading existing index from {config.STORAGE_DIR} ...")
        storage_context = StorageContext.from_defaults(persist_dir=config.STORAGE_DIR)
        return load_index_from_storage(storage_context)
    else:
        return build_index()


if __name__ == "__main__":
    build_index()
