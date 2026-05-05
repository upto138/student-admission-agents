"""
RAG Tool - Retrieval-Augmented Generation

Backend is selected via the RAG_BACKEND variable in .env:
    - "chroma"  → ChromaDB local (dev/test)
    - "azure"   → Azure AI Search (production, pre-indexed on Blob Storage)

This tool is registered with FoundryChatClient agent as a @tool decorator
It allow Researcher Agent query the knowledge base saved from previous crawls
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from agent_framework import tool
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# ── Lazy-load ChromaDB to avoid import error when it's not installed ──
_chroma_client = None
_collection = None
_embedding_client = None

COLLECTION_NAME = "admission_knowledge"
TOP_K = 5  # default number of document returned

# =============================================================================
# Shared: Azure OpenAI Embedding client
# =============================================================================

_embedding_client: Optional[AzureOpenAI] = None

def _get_embedding_client() -> AzureOpenAI:
    """Initialize Azure OpenAI client for embedding"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
    return _embedding_client

def embed_text(text: str) -> list[float]:
    """Creating embedding vector from text using Azure OpenAI."""
    client = _get_embedding_client()
    deployment = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "embedding-admission")

    response = client.embeddings.create(
        input=text,
        model=deployment,
    )
    return response.data[0].embedding

# =============================================================================
# Abstract Backend Interface
# =============================================================================

class RAGBackend(ABC):
    """A common interface for all RAG backends."""

    @abstractmethod
    def query(self, query: str, query_embedding: list[float], top_k: int) -> list[dict]:
        """
        Return list[dict] with schema:
          {"content": str, "metadata": dict, "relevance_score": float}
        """
        ...

    @abstractmethod
    def add(self, content: str, embedding: list[float], metadata: dict, doc_id: str) -> None:
        """Add document to knowledge base."""
        ...

    @abstractmethod
    def count(self) -> int:
        """The number of documents currently in the knowledge base."""
        ...

# =============================================================================
# Backend 1: ChromaDB (local dev/test)
# =============================================================================

class ChromaBackend(RAGBackend):
    """
    Use ChromaDB PersistentClient for local storage.
    No internet connection required, suitable for mocking data during development.
    """

    def __init__(self):
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Admission knowledge base for Vietnamese universities"},
            )
            logger.info(
                f"[ChromaBackend] Collection '{COLLECTION_NAME}' ready. "
                f"Count: {self._collection.count()}"
            )
            return self._collection
        except ImportError:
            logger.warning("[ChromaBackend] chromadb not installed.")
            return None
        except Exception as e:
            logger.error(f"[ChromaBackend] Init failed: {e}")
            return None

    def query(self, query: str, query_embedding: list[float], top_k: int) -> list[dict]:
        collection = self._get_collection()
        if collection is None or collection.count() == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        return [
            {
                "content": doc,
                "metadata": meta,
                "relevance_score": round(1 - dist, 4),
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def add(self, content: str, embedding: list[float], metadata: dict, doc_id: str) -> None:
        collection = self._get_collection()
        if collection is None:
            raise RuntimeError("ChromaDB collection not available")
        collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def count(self) -> int:
        collection = self._get_collection()
        return collection.count() if collection else 0


# =============================================================================
# Backend 2: Azure AI Search (production)
# =============================================================================

class AzureSearchBackend(RAGBackend):
    """
    Use Azure AI Search with pre-indexed data from Blob Storage.
    Search by the already indexed vector field "embedding"

    The index already has the following fields:
      - id          (Edm.String, key)
      - content     (Edm.String, retrievable)
      - embedding   (Collection(Edm.Single), searchable, dimensions=1536)
      - source_url  (Edm.String, retrievable, filterable)
      - university  (Edm.String, retrievable, filterable)
      - year        (Edm.String, retrievable, filterable)
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from azure.search.documents import SearchClient
            from azure.core.credentials import AzureKeyCredential

            endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
            api_key = os.environ["AZURE_SEARCH_API_KEY"]
            index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME", "admission-knowledge")

            self._client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key),
            )
            logger.info(f"[AzureSearchBackend] Connected to index '{index_name}'")
            return self._client
        except KeyError as e:
            logger.error(f"[AzureSearchBackend] Missing env var: {e}")
            return None
        except Exception as e:
            logger.error(f"[AzureSearchBackend] Init failed: {e}")
            return None

    def query(self, query: str, query_embedding: list[float], top_k: int) -> list[dict]:
        client = self._get_client()
        if client is None:
            return []

        try:
            from azure.search.documents.models import VectorizedQuery

            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="embedding",         # field name vector in index
            )

            results = client.search(
                search_text=None,           # pure vector search
                vector_queries=[vector_query],
                select=["id", "content", "source_url", "university", "year"],
                top=top_k,
            )

            output = []
            for r in results:
                score = r.get("@search.score", 0.0)
                output.append({
                    "content": r.get("content", ""),
                    "metadata": {
                        "source_url": r.get("source_url", ""),
                        "university": r.get("university", ""),
                        "year": r.get("year", ""),
                    },
                    # Azure returns a cosine score 0-1, normalize to the same scale
                    "relevance_score": round(min(score, 1.0), 4),
                })
            return output

        except Exception as e:
            logger.error(f"[AzureSearchBackend] Query failed: {e}")
            return []

    def add(self, content: str, embedding: list[float], metadata: dict, doc_id: str) -> None:
        """
        Upload document to Azure AI Search.
        This is usually called by the ingestion pipeline, not by a direct agent
        """
        client = self._get_client()
        if client is None:
            raise RuntimeError("Azure Search client not available")

        document = {
            "id": doc_id,
            "content": content,
            "embedding": embedding,
            **metadata,
        }
        client.upload_documents(documents=[document])
        logger.info(f"[AzureSearchBackend] Uploaded doc: {doc_id}")

    # Todo: Ko lấy được thì khỏi cho vào abstraction class trên
    def count(self) -> int:
        """Azure Search does not have a direct API count - return -1 to indicate 'unknown'."""
        return -1


# =============================================================================
# Backend Factory — selected by RAG_BACKEND env var
# =============================================================================

_backend_instance: Optional[RAGBackend] = None

def _get_backend() -> RAGBackend:
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    backend_name = os.environ.get("RAG_BACKEND", "chroma").lower().strip()

    if backend_name == "azure":
        _backend_instance = AzureSearchBackend()
        logger.info("[RAG] Using backend: Azure AI Search (production)")
    else:
        _backend_instance = ChromaBackend()
        logger.info("[RAG] Using backend: ChromaDB (local dev)")

    return _backend_instance


# =============================================================================
# The Tool is registered with Researcher Agent
# =============================================================================

@tool(approval_mode="never_require")
def query_knowledge_base(query: str, top_k: int = TOP_K) -> str:
    """
    Query knowledge base to find information related to the query

    Args:
        query: The question or keyword to search for
        top_k: The maximum number of related document to return

    Returns:
        a JSON string containing a list of related documents
    """
    backend = _get_backend()

    # Azure backend return -1 so a separate handle is needed
    count = backend.count()
    if count == 0:
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
        results = backend.query(query=query, query_embedding=query_embedding, top_k=top_k)
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[RAG] query_knowledge_base failed: {e}")
        return json.dumps({"query": query, "results": [], "error": str(e)}, ensure_ascii=False)



# =============================================================================
# Ingestion helper — use for ingestion pipeline
# =============================================================================

def add_to_knowledge_base(
    content: str,
    source_url: str,
    university: str = "",
    year: str = "2026",
) -> str:
    """
    Add document to knowledge base (ChromaDB or Azure AI Search).
    This is usually called by the ingestion pipeline, not by a direct agent
    """
    backend = _get_backend()
    try:
        embedding = embed_text(content)
        url_hash = hashlib.sha256(source_url.encode("utf-8")).digest()
        safe_url = base64.urlsafe_b64encode(url_hash).decode("ascii").rstrip("=")
        content_hash = abs(hash(content)) % 100000
        doc_id = f"doc_{safe_url}_{year}_{content_hash}"
        metadata = {"source_url": source_url, "university": university, "year": year}
        backend.add(content=content, embedding=embedding, metadata=metadata, doc_id=doc_id)
        return json.dumps({"success": True, "doc_id": doc_id}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[RAG] add_to_knowledge_base failed: {e}")
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


ALL_RAG_TOOLS = [query_knowledge_base]