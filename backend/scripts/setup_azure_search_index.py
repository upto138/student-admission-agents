"""
Create an Azure AI Search index with a vector field.
Run this once before ingesting data.

Usage:
    python scripts/setup_azure_search_index.py
"""
import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

def create_index():
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    api_key = os.environ["AZURE_SEARCH_API_KEY"]
    index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME", "admission-knowledge")

    # text-embedding-ada-002 → 1536 dims
    # text-embedding-3-small → 1536 dims
    # text-embedding-3-large → 3072 dims
    # Adjust according to the model you are using
    EMBEDDING_DIMENSIONS = 3072

    fields = [
        SimpleField(name="id",          type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source_url",  type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="university",  type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="year",        type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-algo")],
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

    client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
    result = client.create_or_update_index(index)
    print(f"Index '{result.name}' created/updated successfully.")

if __name__ == "__main__":
    create_index()