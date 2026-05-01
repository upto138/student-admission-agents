"""
RAG Tool - Retrieval-Augmented Generation using ChromaDB + Azure OpenAI Embeddings

The Tool is registered with Azure AI Agent as a FunctionTool
It allow Researcher Agent query the knowledge base saved from previous crawls
"""
from __future__ import annotations

import json
import logging
import os
from typing import Callable
# from typing import Optional

from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# ── Lazy-load ChromaDB to avoid import error when it's not installed ──
_chroma_client = None
_collection = None
_embedding_client = None

COLLECTION_NAME = "admission_knowledge"
TOP_K = 5  # default numeber of document returned

def _get_embedding_client() -> AzureOpenAI:
    """Initalize Azure OpenAI client for embedding"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
    return _embedding_client


def _get_collection():
    """Initalize ChromaDB collection"""
    global _chroma_client, _collection
    if _collection is None:
        try:
            import chromadb

            persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
            _chroma_client = chromadb.PersistentClient(path=persist_dir)
            _collection = _chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Admission knowledge base for Vietnamese universities"},
            )
            logger.info(f"ChromaDB collection '{COLLECTION_NAME}' loaded. Count: {_collection.count()}")
        except ImportError:
            logger.warning("chromadb not installed. RAG tool will return empty results.")
            return None
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            return None
    return _collection


def embed_text(text: str) -> list[float]:
    """Creating embedding vector from text using Azure OpenAI."""
    client = _get_embedding_client()
    deployment = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "embedding-admission")

    response = client.embeddings.create(
        input=text,
        model=deployment,
    )
    return response.data[0].embedding


def query_knowledge_base(query: str, top_k: int = TOP_K) -> str:
    """
    Query knowledge base to find information related to the query

    Args:
        query: The question or keyword to search for
        top_k: The maximum number of related document to return

    Returns:
        a JSON string containing a list of related documents
    """
    collection = _get_collection()

    if collection is None or collection.count() == 0:
        return json.dumps(
            {
                "query": query,
                "results": [],
                "note": "Knowledge base is empty. Please ingest data first.",
            },
            ensure_ascii=False,
        )

    try:
        query_embedding = embed_text(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])
        documents = documents[0] if documents else []
        metadatas = results.get("metadatas", [[]])
        metadatas = metadatas[0] if metadatas else []
        distances = results.get("distances", [[]])
        distances = distances[0] if distances else []

        formatted = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted.append(
                {
                    "content": doc,
                    "metadata": meta,
                    "relevance_score": round(1 - dist, 4),  # convert distance to similarity
                }
            )

        return json.dumps({"query": query, "results": formatted}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return json.dumps({"query": query, "results": [], "error": str(e)}, ensure_ascii=False)


def add_to_knowledge_base(content: str, source_url: str, university: str = "", year: str = "2026") -> str:
    """
    Add new information to the knowledge base.
    Called by ingestion pipeline (not the direct agent).

    Args:
        content: The text content to be saved.
        source_url: The source URL of the information.
        university: University Name (optional).
        year: Year of enrollment.

    Returns:
        Result information
    """
    collection = _get_collection()
    if collection is None:
        return json.dumps({"success": False, "error": "ChromaDB not available"})

    try:
        embedding = embed_text(content)
        doc_id = f"{source_url}_{year}_{hash(content) % 100000}"

        collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[{"source_url": source_url, "university": university, "year": year}],
        )

        return json.dumps({"success": True, "doc_id": doc_id, "collection_count": collection.count()})

    except Exception as e:
        logger.error(f"Failed to add to knowledge base: {e}")
        return json.dumps({"success": False, "error": str(e)})


# ── Defind function schema for registration with Azure AI Agent ───

QUERY_KB_DEFINITION = {
    "name": "query_knowledge_base",
    "description": (
        "Query the internal knowledge base to retrieve addmission stored information"
        "Use RAG with ChromaDB. "
        "Use when you need information about addmission scores application requirements and selection methods from previous years"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Questions or keywords to search wihini the knowledge base",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of related documents to retrieve (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "query_knowledge_base": query_knowledge_base,
}